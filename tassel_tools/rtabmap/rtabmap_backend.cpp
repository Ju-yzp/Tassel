#include "rtabmap/rtabmap_backend.h"

#include "frond_end/feature_manager.h"

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Transform.h>

#include <cmath>
#include <exception>
#include <opencv2/features2d.hpp>
#include <stdexcept>
#include <utility>

namespace tassel_tools {
namespace {

rtabmap::Transform toRtabmapTransform(const Sophus::SE3d& pose) {
    const Eigen::Matrix3d& rotation = pose.rotationMatrix();
    const Eigen::Vector3d& translation = pose.translation();
    return rtabmap::Transform(
        static_cast<float>(rotation(0, 0)), static_cast<float>(rotation(0, 1)),
        static_cast<float>(rotation(0, 2)), static_cast<float>(translation.x()),
        static_cast<float>(rotation(1, 0)), static_cast<float>(rotation(1, 1)),
        static_cast<float>(rotation(1, 2)), static_cast<float>(translation.y()),
        static_cast<float>(rotation(2, 0)), static_cast<float>(rotation(2, 1)),
        static_cast<float>(rotation(2, 2)), static_cast<float>(translation.z()));
}

void validateOptions(const RtabmapBackendOptions& options) {
    if (options.queue_capacity == 0) {
        throw std::invalid_argument("RTAB-Map queue capacity must be positive");
    }
    if (!std::isfinite(options.linear_variance) || options.linear_variance <= 0.0 ||
        !std::isfinite(options.angular_variance) || options.angular_variance <= 0.0) {
        throw std::invalid_argument("RTAB-Map odometry variances must be finite and positive");
    }
}

}  // namespace

struct RtabmapBackend::Impl {
    struct Frame {
        tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
        cv::Mat image;
        Sophus::SE3d camera_pose;
        std::vector<tassel_core::ObservedLandmark> landmarks;
    };

    Impl(
        const cv::Mat& camera_matrix, const cv::Mat& distortion, const cv::Size& image_size,
        bool equidistant, const Sophus::SE3d& imu_to_camera, RtabmapBackendOptions options)
        : imu_to_camera(std::move(imu_to_camera)), options(std::move(options)) {
        validateOptions(this->options);
        if (camera_matrix.type() != CV_64FC1 || camera_matrix.rows != 3 ||
            camera_matrix.cols != 3 || image_size.width <= 0 || image_size.height <= 0) {
            throw std::invalid_argument("Invalid RTAB-Map camera calibration");
        }
        if (distortion.type() != CV_64FC1 || distortion.empty()) {
            throw std::invalid_argument("RTAB-Map distortion must be non-empty CV_64FC1");
        }
        // RTAB-Map 固定要求 1xN；这里只改变存储布局，不改变 YAML 中的系数顺序。
        cv::Mat rtabmap_distortion = distortion.reshape(1, 1).clone();
        if (equidistant) {
            if (rtabmap_distortion.total() != 4) {
                throw std::invalid_argument(
                    "Equidistant RTAB-Map calibration requires 4 coefficients");
            }
            // RTAB-Map 以 [k1,k2,0,0,k3,k4] 六项布局标识 equidistant 模型。
            rtabmap_distortion = cv::Mat::zeros(1, 6, CV_64FC1);
            rtabmap_distortion.at<double>(0, 0) = distortion.at<double>(0);
            rtabmap_distortion.at<double>(0, 1) = distortion.at<double>(1);
            rtabmap_distortion.at<double>(0, 4) = distortion.at<double>(2);
            rtabmap_distortion.at<double>(0, 5) = distortion.at<double>(3);
        }
        cv::Mat projection = cv::Mat::zeros(3, 4, CV_64FC1);
        camera_matrix.copyTo(projection.colRange(0, 3));
        camera_model = rtabmap::CameraModel(
            "tassel", image_size, camera_matrix, rtabmap_distortion, cv::Mat::eye(3, 3, CV_64FC1),
            projection, rtabmap::Transform::getIdentity());

        rtabmap::ParametersMap parameters;
        parameters.emplace(rtabmap::Parameters::kRGBDEnabled(), "true");
        parameters.emplace(rtabmap::Parameters::kMemIncrementalMemory(), "true");
        parameters.emplace(rtabmap::Parameters::kOptimizerStrategy(), "3");
        parameters.emplace(rtabmap::Parameters::kRtabmapImagesAlreadyRectified(), "false");
        rtabmap.init(parameters, this->options.database_path, false);
        descriptor = cv::ORB::create();
        worker = std::thread([this]() { run(); });
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
        }
        condition.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        rtabmap.close();
    }

    void submit(
        tassel_utils::FrameId frame_id, const cv::Mat& image, const Sophus::SE3d& world_to_imu,
        const std::vector<tassel_core::ObservedLandmark>& landmarks) {
        if (frame_id == tassel_utils::kInvalidFrameId || image.empty() ||
            !world_to_imu.matrix().allFinite()) {
            throw std::invalid_argument("Invalid RTAB-Map frame");
        }
        Frame frame{frame_id, image.clone(), world_to_imu * imu_to_camera, landmarks};
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopped) {
                throw std::logic_error("RTAB-Map backend is stopped");
            }
            if (worker_error) {
                std::rethrow_exception(worker_error);
            }
            // 后端积压时保留最新关键帧，避免回环处理反向阻塞实时 VIO。
            if (frames.size() == options.queue_capacity) {
                frames.pop_front();
            }
            frames.push_back(std::move(frame));
        }
        condition.notify_one();
    }

    void run() {
        try {
            while (true) {
                Frame frame;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    condition.wait(lock, [this]() { return stopped || !frames.empty(); });
                    if (frames.empty()) {
                        return;
                    }
                    frame = std::move(frames.front());
                    frames.pop_front();
                }
                const double timestamp = tassel_utils::frameIdToSeconds(frame.frame_id);
                rtabmap::SensorData data(frame.image, camera_model, next_node_id++, timestamp);
                std::vector<cv::KeyPoint> keypoints;
                keypoints.reserve(frame.landmarks.size());
                for (size_t i = 0; i < frame.landmarks.size(); ++i) {
                    cv::KeyPoint keypoint(frame.landmarks[i].pixel, 7.0f);
                    keypoint.class_id = static_cast<int>(i);
                    keypoints.push_back(keypoint);
                }
                cv::Mat descriptors;
                descriptor->compute(frame.image, keypoints, descriptors);
                std::vector<cv::Point3f> camera_points;
                camera_points.reserve(keypoints.size());
                const Sophus::SE3d camera_to_world = frame.camera_pose;
                for (cv::KeyPoint& keypoint : keypoints) {
                    if (keypoint.class_id < 0 ||
                        keypoint.class_id >= static_cast<int>(frame.landmarks.size())) {
                        throw std::logic_error("ORB changed the retained landmark index");
                    }
                    const Eigen::Vector3d point =
                        camera_to_world.inverse() * frame.landmarks[keypoint.class_id].world_point;
                    if (!point.allFinite() || point.z() <= 0.0) {
                        throw std::logic_error("Retained landmark is invalid in camera frame");
                    }
                    camera_points.emplace_back(
                        static_cast<float>(point.x()), static_cast<float>(point.y()),
                        static_cast<float>(point.z()));
                    keypoint.class_id = frame.landmarks[keypoint.class_id].feature_id;
                }
                data.setFeatures(keypoints, camera_points, descriptors);
                rtabmap.process(
                    data, toRtabmapTransform(frame.camera_pose),
                    static_cast<float>(options.linear_variance),
                    static_cast<float>(options.angular_variance));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            worker_error = std::current_exception();
            stopped = true;
        }
    }

    Sophus::SE3d imu_to_camera;
    RtabmapBackendOptions options;
    rtabmap::CameraModel camera_model;
    rtabmap::Rtabmap rtabmap;
    cv::Ptr<cv::ORB> descriptor;
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<Frame> frames;
    std::thread worker;
    int next_node_id = 1;
    std::exception_ptr worker_error;
    bool stopped = false;
};

RtabmapBackend::RtabmapBackend(
    const cv::Mat& camera_matrix, const cv::Mat& distortion, const cv::Size& image_size,
    bool equidistant, const Sophus::SE3d& imu_to_camera, RtabmapBackendOptions options)
    : impl_(std::make_unique<Impl>(
          camera_matrix, distortion, image_size, equidistant, imu_to_camera, std::move(options))) {}

RtabmapBackend::~RtabmapBackend() = default;

void RtabmapBackend::submit(
    tassel_utils::FrameId frame_id, const cv::Mat& image, const Sophus::SE3d& world_to_imu,
    const std::vector<tassel_core::ObservedLandmark>& landmarks) {
    impl_->submit(frame_id, image, world_to_imu, landmarks);
}

}  // namespace tassel_tools
