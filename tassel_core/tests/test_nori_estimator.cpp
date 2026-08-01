// 真实 Nori 硬件的纯 C++ VIO 入口，不依赖 ROS 或 rosbag。

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>

#include "cam/camera_factory.h"
#include "nori/nori_decoder.h"
#include "nori/nori_device.h"
#include "estimator/estimator.h"
#include "frond_end/feature_manager.h"
#include "frond_end/feature_tracker.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/types.h"
#include "viewer/viewer.h"

namespace {

constexpr int kCaptureWidth = 4000;
constexpr int kCaptureHeight = 1200;
constexpr double kNoriSyncDelay = 0.002671414577930159;

struct MonoFrame {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    cv::Mat image;
};

struct SyncedFrame {
    MonoFrame mono;
    std::vector<tassel_utils::IMUMeasurement> imu;
};

struct LatestImage {
    cv::Mat image;
    double timestamp = -1.0;
};

struct TrackedFrame {
    MonoFrame mono;
    std::vector<tassel_utils::IMUMeasurement> imu;
    std::unordered_map<int, tassel_core::FeaturePerFrame> features;
};

class EncodedQueue {
public:
    void push(std::vector<uchar> bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        frames_.push_back(std::move(bytes));
        cv_.notify_all();
    }

    bool waitPop(std::vector<uchar>& bytes) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stopped_ || !frames_.empty(); });
        if (frames_.empty()) {
            return false;
        }
        bytes = std::move(frames_.front());
        frames_.pop_front();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frames_.size();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::vector<uchar>> frames_;
    bool stopped_ = false;
};

class TrackedQueue {
public:
    void push(TrackedFrame frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        frames_.push_back(std::move(frame));
        cv_.notify_all();
    }

    bool waitPop(TrackedFrame& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stopped_ || !frames_.empty(); });
        if (frames_.empty()) {
            return false;
        }
        frame = std::move(frames_.front());
        frames_.pop_front();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frames_.size();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<TrackedFrame> frames_;
    bool stopped_ = false;
};

class NoriSync {
public:
    void push(MonoFrame mono, const std::vector<tassel_utils::IMUMeasurement>& imu) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        for (const auto& sample : imu) {
            if (imus_.empty() || sample.timestamp > imus_.back().timestamp) {
                imus_.push_back(sample);
            }
        }
        frames_.push_back(std::move(mono));
        cv_.notify_all();
    }

    bool waitPop(SyncedFrame& output) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
            if (frames_.empty()) {
                return stopped_;
            }
            const double end = tassel_utils::frameIdToSeconds(frames_.front().frame_id) +
                               kNoriSyncDelay;
            return stopped_ || (!imus_.empty() && imus_.back().timestamp >= end);
        });
        if (frames_.empty()) {
            return false;
        }
        const double end =
            tassel_utils::frameIdToSeconds(frames_.front().frame_id) + kNoriSyncDelay;
        if (imus_.empty() || imus_.back().timestamp < end) {
            return false;
        }
        output.mono = std::move(frames_.front());
        frames_.pop_front();
        output.imu.clear();
        if (has_boundary_) {
            output.imu.push_back(boundary_);
        }
        for (const auto& sample : imus_) {
            if (sample.timestamp >= end) {
                break;
            }
            if (!has_boundary_ || sample.timestamp > boundary_.timestamp) {
                output.imu.push_back(sample);
            }
        }
        boundary_ = interpolate(end);
        if (output.imu.empty() || output.imu.back().timestamp < boundary_.timestamp) {
            output.imu.push_back(boundary_);
        }
        // 相邻图像包共享该插值样本，保证预积分区间连续且不重复积分。
        while (!imus_.empty() && imus_.front().timestamp <= end) {
            imus_.pop_front();
        }
        has_boundary_ = true;
        lock.unlock();
        cv_.notify_all();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frames_.size();
    }

private:
    tassel_utils::IMUMeasurement interpolate(double timestamp) const {
        auto upper = std::lower_bound(
            imus_.begin(), imus_.end(), timestamp,
            [](const auto& sample, double value) { return sample.timestamp < value; });
        if (upper != imus_.end() && upper->timestamp == timestamp) {
            return *upper;
        }
        const auto* before = upper != imus_.begin() ? &*std::prev(upper)
                                                    : (has_boundary_ ? &boundary_ : nullptr);
        const auto* after = upper != imus_.end() ? &*upper : nullptr;
        if (!before || !after || after->timestamp <= before->timestamp) {
            throw std::logic_error("Cannot interpolate Nori IMU boundary");
        }
        const double alpha =
            (timestamp - before->timestamp) / (after->timestamp - before->timestamp);
        return {before->acc + alpha * (after->acc - before->acc),
                before->gyro + alpha * (after->gyro - before->gyro), timestamp};
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<MonoFrame> frames_;
    std::deque<tassel_utils::IMUMeasurement> imus_;
    tassel_utils::IMUMeasurement boundary_{};
    bool has_boundary_ = false;
    bool stopped_ = false;
};

void scaleTrackingConfiguration(tassel_tools::Parameters& params, double scale) {
    if (!std::isfinite(scale) || scale <= 0.0 || scale > 1.0) {
        throw std::invalid_argument("tracking_scale must be finite and in (0, 1]");
    }
    if (params.cam_intrinsic.type() != CV_64FC1 || params.cam_intrinsic.rows != 3 ||
        params.cam_intrinsic.cols != 3) {
        throw std::logic_error("Nori camera intrinsics must be a 3x3 double matrix");
    }

    // 缩放后的图像、相机模型和所有像素域判据必须使用同一坐标布局。
    const int source_cols = params.cols;
    const int source_rows = params.rows;
    params.cols = std::max(1, cvRound(source_cols * scale));
    params.rows = std::max(1, cvRound(source_rows * scale));
    const double scale_x = static_cast<double>(params.cols) / source_cols;
    const double scale_y = static_cast<double>(params.rows) / source_rows;
    params.cam_intrinsic.at<double>(0, 0) *= scale_x;
    params.cam_intrinsic.at<double>(1, 1) *= scale_y;
    params.cam_intrinsic.at<double>(0, 2) =
        scale_x * (params.cam_intrinsic.at<double>(0, 2) + 0.5) - 0.5;
    params.cam_intrinsic.at<double>(1, 2) =
        scale_y * (params.cam_intrinsic.at<double>(1, 2) + 0.5) - 0.5;
    params.per_grid_cols = std::max(1, cvRound(params.per_grid_cols * scale));
    params.per_grid_rows = std::max(1, cvRound(params.per_grid_rows * scale));
    params.edge_x = std::max(0, cvRound(params.edge_x * scale));
    params.edge_y = std::max(0, cvRound(params.edge_y * scale));
    params.mask_radius *= scale;
    params.max_square_move_dist *= scale * scale;
    params.reproj_err_thres *= scale;
    params.reproj_huber_thres *= scale;
    params.parallax_threshold *= scale;
    params.visual_factor_weight /= scale;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string config = argc > 1 ? argv[1] : "config/tassel.yaml";
    const std::string device = argc > 2 ? argv[2] : "/dev/video4";
    const int max_frames = argc > 3 ? std::stoi(argv[3]) : 0;
    const double tracking_scale = argc > 4 ? std::stod(argv[4]) : 1.0;
    if (max_frames < 0) {
        throw std::invalid_argument("max_frames must not be negative");
    }

    tassel_tools::Parameters params(config);
    const int source_width = params.cols;
    const int source_height = params.rows;
    scaleTrackingConfiguration(params, tracking_scale);
    if (params.camera_model != "equi") {
        throw std::invalid_argument("Nori input requires an equidistant source calibration");
    }
    auto camera = tassel_core::CameraFactory::create(
        params.camera_model, params.cam_intrinsic, params.cam_distort, params.cols, params.rows);
    const tassel_core::CameraBase* camera_ptr = camera.get();
    std::cout << "[nori] source=" << source_width << 'x' << source_height << " tracking="
              << params.cols << 'x' << params.rows << " scale=" << tracking_scale << '\n';
    tassel_core::FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    // Nori 直接在原始鱼眼图上跟踪；观测点再由 equi 模型转换到归一化坐标。
    tracker.setCamera(
        std::move(camera), params.per_grid_rows, params.per_grid_cols, params.edge_y, params.edge_x,
        params.mask_radius, params.min_feature_num);
    auto state = std::make_shared<tassel_core::State>(static_cast<int>(params.max_frame_count) + 1);
    auto feature_manager = std::make_shared<tassel_core::FeatureManager>(
        params.reproj_err_thres, params.min_landmark_observations, params.parallax_threshold,
        params.keyframe_min_connection_ratio, params.min_depth, params.max_depth);
    tassel_core::Estimator estimator(params, state, feature_manager);
    estimator.setCamera(camera_ptr);
    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("odom");
    rclcpp::QoS image_qos(rclcpp::KeepLast(1));
    image_qos.best_effort().durability_volatile();
    viewer->createImagePublisher("mono/image", image_qos);
    viewer->createOdometryPublisher("imu_link", "vio/odometry");
    viewer->createOdometryPublisher(
        "camera_optical_frame", "vio/camera_odometry", rclcpp::QoS(10), false);
    viewer->publishStaticTransform(
        "imu_link", "camera_optical_frame", params.tic,
        Eigen::Quaterniond(params.ric).normalized());
    viewer->createPathPublisher("vio/path", rclcpp::QoS(10), params.viewer_path_max_poses);
    viewer->createPathPublisher(
        "ground_truth/path", rclcpp::QoS(10), params.viewer_path_max_poses);
    viewer->createCompressedImagePublisher("optimization/visual_window", image_qos);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(viewer);
    std::atomic_bool stop_executor{false};
    std::thread spin_thread([&]() {
        while (rclcpp::ok() && !stop_executor.load()) {
            executor.spin_some(std::chrono::milliseconds(5));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::mutex latest_image_mutex;
    LatestImage latest_image;
    std::atomic_bool stop_image_publisher{false};
    std::thread image_publish_thread([&]() {
        constexpr auto kPeriod = std::chrono::milliseconds(33);
        auto next_tick = std::chrono::steady_clock::now();
        double last_timestamp = -1.0;
        while (rclcpp::ok() && !stop_image_publisher.load()) {
            LatestImage image;
            {
                std::lock_guard<std::mutex> lock(latest_image_mutex);
                image.image = latest_image.image.clone();
                image.timestamp = latest_image.timestamp;
            }
            if (!image.image.empty() && image.timestamp > last_timestamp) {
                viewer->publishImage(
                    "mono/image", "camera_optical_frame", image.image, image.timestamp);
                last_timestamp = image.timestamp;
            }
            next_tick += kPeriod;
            std::this_thread::sleep_until(next_tick);
        }
    });
    estimator.setPoseCallback([&viewer, &params](double timestamp, const Sophus::SE3d& pose) {
        viewer->publishOdometry(
            "vio/odometry", pose.translation(), pose.unit_quaternion(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), timestamp);
        const Sophus::SE3d camera_pose = pose * Sophus::SE3d(params.ric, params.tic);
        viewer->publishOdometry(
            "vio/camera_odometry", camera_pose.translation(), camera_pose.unit_quaternion(),
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), timestamp);
        viewer->publishPath("vio/path", pose.translation(), pose.unit_quaternion(), timestamp);
        std::cout << "[pose] t=" << timestamp << " p=" << pose.translation().transpose() << '\n';
    });
    estimator.setVisualFactorCallback(
        [&viewer](double /*timestamp*/, const std::vector<int>& visual_factors_per_frame) {
            viewer->publishVisualFactorWindow(
                "optimization/visual_window", visual_factors_per_frame);
        });

    tassel_hardware::NoriDeviceConfig nori_config;
    nori_config.width = kCaptureWidth;
    nori_config.height = kCaptureHeight;
    nori_config.fps = 30;
    tassel_hardware::NoriDevice nori(device, nori_config);
    const int bytes_per_line = nori.bytesPerLine();
    std::cout << "[nori] capture=4000x1200 YUYV@30 bytes_per_line=" << bytes_per_line << '\n';

    EncodedQueue encoded_queue;
    NoriSync sync;
    std::atomic_bool stop_pipeline{false};
    const auto pipeline_start = std::chrono::steady_clock::now();
    std::atomic_size_t captured_count{0};
    std::atomic_size_t decoded_count{0};
    std::atomic_size_t tracked_count{0};
    std::atomic_llong decode_time_us{0};
    std::atomic_llong tracking_time_us{0};
    std::atomic_llong sensor_interval_us{0};
    std::exception_ptr capture_error;
    std::thread capture_thread([&]() {
        try {
            for (int captured = 0;
                 rclcpp::ok() && !stop_pipeline.load() &&
                 (max_frames == 0 || captured < max_frames);
                 ++captured) {
                tassel_hardware::NoriCapture capture;
                while (rclcpp::ok() && !stop_pipeline.load() && !nori.tryRead(capture)) {
                    usleep(1000);
                }
                if (!rclcpp::ok() || stop_pipeline.load()) {
                    break;
                }
                encoded_queue.push(std::move(capture.bytes));
                ++captured_count;
            }
        } catch (...) {
            capture_error = std::current_exception();
            stop_pipeline = true;
        }
        encoded_queue.stop();
    });

    std::exception_ptr decode_error;
    std::thread decode_thread([&]() {
        try {
            tassel_hardware::NoriDecoder decoder;
            std::vector<uchar> bytes;
            tassel_utils::FrameId previous_exposure_end = tassel_utils::kInvalidFrameId;
            while (rclcpp::ok() && !stop_pipeline.load() && encoded_queue.waitPop(bytes)) {
                const auto decode_start = std::chrono::steady_clock::now();
                const size_t required_bytes =
                    static_cast<size_t>(bytes_per_line) * kCaptureHeight;
                if (bytes.size() < required_bytes) {
                    throw std::runtime_error("Incomplete Nori YUYV frame");
                }
                const cv::Mat yuyv(
                    kCaptureHeight, kCaptureWidth, CV_8UC2, bytes.data(),
                    bytes_per_line);
                cv::Mat image;
                cv::extractChannel(yuyv, image, 0);
                tassel_hardware::NoriFrameTiming timing;
                std::vector<tassel_utils::IMUMeasurement> measurements;
                if (!decoder.decode(image, timing, measurements)) {
                    continue;
                }
                if (previous_exposure_end != tassel_utils::kInvalidFrameId) {
                    sensor_interval_us =
                        (timing.exposure_end - previous_exposure_end) / 1000;
                }
                previous_exposure_end = timing.exposure_end;
                cv::Mat right_gray = image.colRange(0, image.cols / 2);
                if (source_width != right_gray.cols || source_height != right_gray.rows) {
                    throw std::logic_error("Nori source calibration does not match the mono image");
                }
                if (params.edge_x > 0) {
                    // 左侧为时间戳/IMU 编码条带，不得作为光流纹理参与跟踪。
                    const int source_edge_x = cvRound(params.edge_x / tracking_scale);
                    right_gray.colRange(0, std::min(source_edge_x, right_gray.cols)).setTo(0);
                }
                cv::Mat tracking_image;
                if (tracking_scale < 1.0) {
                    cv::resize(
                        right_gray, tracking_image, cv::Size(params.cols, params.rows), 0.0, 0.0,
                        cv::INTER_AREA);
                } else {
                    tracking_image = right_gray.clone();
                }
                sync.push({timing.exposure_end, std::move(tracking_image)}, measurements);
                decode_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - decode_start)
                                      .count();
                ++decoded_count;
            }
        } catch (...) {
            decode_error = std::current_exception();
            stop_pipeline = true;
            encoded_queue.stop();
        }
        sync.stop();
    });

    TrackedQueue tracked_queue;
    std::exception_ptr tracking_error;
    std::thread tracking_thread([&]() {
        try {
            SyncedFrame synced;
            while (rclcpp::ok() && sync.waitPop(synced)) {
                const auto tracking_start = std::chrono::steady_clock::now();
                auto features = tracker.monoTracking(synced.mono.image);
                cv::Mat tracking_image = synced.mono.image.clone();
                tracker.drawTrackingResult(tracking_image);
                {
                    std::lock_guard<std::mutex> lock(latest_image_mutex);
                    latest_image.image = std::move(tracking_image);
                    latest_image.timestamp =
                        tassel_utils::frameIdToSeconds(synced.mono.frame_id);
                }
                for (auto& [id, feature] : features) {
                    (void)id;
                    feature.sync_delay = kNoriSyncDelay;
                }
                tracked_queue.push(
                    {std::move(synced.mono), std::move(synced.imu), std::move(features)});
                tracking_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - tracking_start)
                                        .count();
                ++tracked_count;
            }
        } catch (...) {
            tracking_error = std::current_exception();
            stop_pipeline = true;
            encoded_queue.stop();
            sync.stop();
        }
        tracked_queue.stop();
    });

    int processed = 0;
    TrackedFrame packet;
    while (rclcpp::ok() && tracked_queue.waitPop(packet)) {
        const auto frame_id = packet.mono.frame_id;
        estimator.processMeasurement(frame_id, packet.features, packet.imu, kNoriSyncDelay);
        ++processed;
        if (processed % 10 == 0) {
            const double elapsed = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - pipeline_start)
                                       .count();
            const size_t decoded = decoded_count.load();
            const size_t tracked = tracked_count.load();
            const long long interval_us = sensor_interval_us.load();
            std::cout << "[nori] frames=" << processed << " imu=" << packet.imu.size()
                      << " features=" << packet.features.size()
                      << " raw_queue=" << encoded_queue.size() << " sync_queue=" << sync.size()
                      << " tracking_queue=" << tracked_queue.size() << " rates="
                      << captured_count.load() / elapsed << '/' << decoded / elapsed << '/'
                      << tracked / elapsed << '/' << processed / elapsed
                      << " sensor_fps="
                      << (interval_us > 0 ? 1e6 / static_cast<double>(interval_us) : 0.0)
                      << " decode_ms="
                      << (decoded > 0 ? decode_time_us.load() / (1000.0 * decoded) : 0.0)
                      << " tracking_ms="
                      << (tracked > 0 ? tracking_time_us.load() / (1000.0 * tracked) : 0.0)
                      << '\n';
        }
    }
    stop_pipeline = true;
    encoded_queue.stop();
    sync.stop();
    capture_thread.join();
    decode_thread.join();
    tracking_thread.join();
    stop_image_publisher = true;
    image_publish_thread.join();
    stop_executor = true;
    executor.cancel();
    spin_thread.join();
    executor.remove_node(viewer);
    viewer.reset();
    rclcpp::shutdown();
    if (capture_error) {
        std::rethrow_exception(capture_error);
    }
    if (decode_error) {
        std::rethrow_exception(decode_error);
    }
    if (tracking_error) {
        std::rethrow_exception(tracking_error);
    }
    return 0;
}
