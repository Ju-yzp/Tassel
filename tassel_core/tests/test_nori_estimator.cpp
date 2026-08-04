// 真实 Nori 硬件的纯 C++ VIO 入口，不依赖 ROS 或 rosbag。

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "cam/camera_factory.h"
#include "estimator/estimator.h"
#include "frond_end/feature_manager.h"
#include "frond_end/feature_tracker.h"
#include "nori/nori_decoder.h"
#include "nori/nori_device.h"
#include "nori/nori_sync.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/types.h"
#include "viewer/viewer.h"

namespace {

constexpr int kCaptureWidth = 4000;
constexpr int kCaptureHeight = 1200;
constexpr double kNoriSyncDelay = 0.002671414577930159;

using MonoFrame = tassel_hardware::NoriImageFrame;
using SyncedFrame = tassel_hardware::NoriSyncedFrame;

struct LatestImage {
    cv::Mat image;
    double timestamp = -1.0;
};

struct TrackedFrame {
    MonoFrame mono;
    std::vector<tassel_utils::IMUMeasurement> imu;
    std::unordered_map<int, tassel_core::FeaturePerFrame> features;
};

enum class MonoHalf {
    First,
    Second,
};

MonoHalf parseMonoHalf(const std::string& value) {
    if (value == "first") {
        return MonoHalf::First;
    }
    if (value == "second") {
        return MonoHalf::Second;
    }
    throw std::invalid_argument("mono_half must be either 'first' or 'second'");
}

const char* monoHalfName(MonoHalf half) { return half == MonoHalf::First ? "first" : "second"; }

cv::Mat buildValidMask(int rows, int cols) { return cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255)); }

cv::Mat loadValidMask(const std::string& mask_path, int rows, int cols) {
    if (mask_path.empty() || mask_path == "auto") {
        return buildValidMask(rows, cols);
    }
    cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        throw std::runtime_error("Failed to read valid mask image: " + mask_path);
    }
    if (mask.rows != rows || mask.cols != cols) {
        throw std::invalid_argument("Valid mask image size does not match the tracking image");
    }
    return mask;
}

template <typename T>
class PipelineQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        values_.push_back(std::move(value));
        cv_.notify_all();
    }

    bool waitPop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stopped_ || !values_.empty(); });
        if (values_.empty()) {
            return false;
        }
        value = std::move(values_.front());
        values_.pop_front();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return values_.size();
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
    std::deque<T> values_;
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
    params.mask_radius *= scale;
    params.max_square_move_dist *= scale * scale;
    params.reproj_err_thres *= scale;
    params.reproj_huber_thres *= scale;
    params.visual_factor_weight /= scale;
}

cv::Mat resizePreviewImage(const cv::Mat& image, double scale) {
    if (!std::isfinite(scale) || scale <= 0.0 || scale > 1.0) {
        throw std::invalid_argument("preview_scale must be finite and in (0, 1]");
    }
    if (std::abs(scale - 1.0) < 1e-9) {
        return image;
    }
    const int width = std::max(1, cvRound(image.cols * scale));
    const int height = std::max(1, cvRound(image.rows * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string config = argc > 1 ? argv[1] : "config/tassel.yaml";
    const std::string device = argc > 2 ? argv[2] : "/dev/video4";
    const int max_frames = argc > 3 ? std::stoi(argv[3]) : 0;
    const double tracking_scale = argc > 4 ? std::stod(argv[4]) : 1.0;
    const int visualization_stride = argc > 5 ? std::stoi(argv[5]) : 1;
    const MonoHalf mono_half = argc > 6 ? parseMonoHalf(argv[6]) : MonoHalf::First;
    const std::string dump_mask_path = argc > 8 ? argv[8] : "";
    const double preview_scale = argc > 9 ? std::stod(argv[9]) : 0.5;
    const double preview_hz = argc > 10 ? std::stod(argv[10]) : 10.0;
    if (max_frames < 0) {
        throw std::invalid_argument("max_frames must not be negative");
    }
    if (visualization_stride < 0) {
        throw std::invalid_argument("visualization_stride must be non-negative");
    }
    if (!std::isfinite(preview_scale) || preview_scale <= 0.0 || preview_scale > 1.0) {
        throw std::invalid_argument("preview_scale must be finite and in (0, 1]");
    }
    if (!std::isfinite(preview_hz) || preview_hz <= 0.0) {
        throw std::invalid_argument("preview_hz must be positive and finite");
    }

    tassel_tools::Parameters params(config);
    const std::string mask_path = argc > 7 ? argv[7] : params.valid_mask_path;
    const int source_width = params.cols;
    const int source_height = params.rows;
    scaleTrackingConfiguration(params, tracking_scale);
    if (params.camera_model != "equi") {
        throw std::invalid_argument("Nori input requires an equidistant source calibration");
    }
    auto camera = tassel_core::CameraFactory::create(
        params.camera_model, params.cam_intrinsic, params.cam_distort, params.cols, params.rows);
    const tassel_core::CameraBase* camera_ptr = camera.get();
    std::cout << "[nori] source=" << source_width << 'x' << source_height
              << " tracking=" << params.cols << 'x' << params.rows << " scale=" << tracking_scale
              << " visualization_stride=" << visualization_stride
              << " preview_scale=" << preview_scale << " preview_hz=" << preview_hz
              << " mono_half=" << monoHalfName(mono_half) << " mask_path=" << mask_path << '\n';
    tassel_core::FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    // Nori 直接在原始鱼眼图上跟踪；观测点再由 equi 模型转换到归一化坐标。
    tracker.setCamera(
        std::move(camera), params.per_grid_rows, params.per_grid_cols, params.mask_radius,
        params.min_feature_num);
    cv::Mat valid_mask = loadValidMask(mask_path, params.rows, params.cols);
    tracker.setValidMask(valid_mask);
    if (!dump_mask_path.empty()) {
        if (!cv::imwrite(dump_mask_path, valid_mask)) {
            throw std::runtime_error("Failed to write valid mask image: " + dump_mask_path);
        }
        std::cout << "[nori] mask written to " << dump_mask_path << '\n';
    }
    auto state = std::make_shared<tassel_core::State>(static_cast<int>(params.max_frame_count) + 1);
    auto feature_manager = std::make_shared<tassel_core::FeatureManager>(
        params.reproj_err_thres, params.min_landmark_observations, params.parallax_threshold,
        params.keyframe_min_connection_ratio, params.min_depth, params.max_depth);
    tassel_core::Estimator estimator(params, state, feature_manager);
    estimator.setCamera(camera_ptr);
    // ROS must be initialized before constructing the node and its publishers.
    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("odom");
    rclcpp::QoS image_qos(rclcpp::KeepLast(1));
    image_qos.best_effort().durability_volatile();
    viewer->createImagePublisher("mono/image", image_qos);
    viewer->createCompressedImagePublisher("mono/image/compressed", image_qos);
    viewer->createOdometryPublisher("imu_link", "vio/odometry");
    viewer->createOdometryPublisher(
        "camera_optical_frame", "vio/camera_odometry", rclcpp::QoS(10), false);
    viewer->publishStaticTransform(
        "imu_link", "camera_optical_frame", params.tic,
        Eigen::Quaterniond(params.ric).normalized());
    viewer->createPathPublisher("vio/path", rclcpp::QoS(10), params.viewer_path_max_poses);
    viewer->createPathPublisher("ground_truth/path", rclcpp::QoS(10), params.viewer_path_max_poses);
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
    const bool publish_tracking_image = visualization_stride > 0;
    std::thread image_publish_thread([&]() {
        if (!publish_tracking_image) {
            return;
        }
        const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / preview_hz));
        auto next_tick = std::chrono::steady_clock::now();
        double last_timestamp = -1.0;
        while (rclcpp::ok() && !stop_image_publisher.load()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_tick) {
                std::this_thread::sleep_until(next_tick);
            } else if (now - next_tick > period) {
                // RViz/传输落后时只保留最新预览，不补发历史周期。
                next_tick = now;
            }
            LatestImage image;
            {
                std::lock_guard<std::mutex> lock(latest_image_mutex);
                image.image = latest_image.image.clone();
                image.timestamp = latest_image.timestamp;
            }
            if (!image.image.empty() && image.timestamp > last_timestamp) {
                // 预览缩放只降低发布带宽，不改变追踪/估计使用的图像坐标。
                cv::Mat preview_image = resizePreviewImage(image.image, preview_scale);
                viewer->publishImage(
                    "mono/image", "camera_optical_frame", preview_image, image.timestamp);
                viewer->publishCompressedImage(
                    "mono/image/compressed", "camera_optical_frame", preview_image, "jpeg",
                    image.timestamp);
                last_timestamp = image.timestamp;
            }
            next_tick += period;
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

    PipelineQueue<std::vector<uchar>> encoded_queue;
    tassel_hardware::NoriSync sync(kNoriSyncDelay);
    std::atomic_bool stop_pipeline{false};
    std::mutex prediction_mutex;
    std::condition_variable prediction_cv;
    std::optional<tassel_core::TrackingPredictionSnapshot> prediction_snapshot;
    const auto pipeline_start = std::chrono::steady_clock::now();
    std::atomic_size_t captured_count{0};
    std::atomic_size_t decoded_count{0};
    std::atomic_size_t decoded_imu_count{0};
    std::atomic_size_t last_decoded_imu_count{0};
    std::atomic<double> last_decoded_imu_first{0.0};
    std::atomic<double> last_decoded_imu_last{0.0};
    std::atomic_size_t tracked_count{0};
    std::atomic_llong decode_time_us{0};
    std::atomic_llong tracking_time_us{0};
    std::atomic_llong sensor_interval_us{0};
    std::exception_ptr capture_error;
    std::thread capture_thread([&]() {
        try {
            for (int captured = 0; rclcpp::ok() && !stop_pipeline.load() &&
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
                const size_t required_bytes = static_cast<size_t>(bytes_per_line) * kCaptureHeight;
                if (bytes.size() < required_bytes) {
                    throw std::runtime_error("Incomplete Nori YUYV frame");
                }
                const cv::Mat yuyv(
                    kCaptureHeight, kCaptureWidth, CV_8UC2, bytes.data(), bytes_per_line);
                cv::Mat image;
                cv::extractChannel(yuyv, image, 0);
                tassel_hardware::NoriFrameTiming timing;
                std::vector<tassel_utils::IMUMeasurement> measurements;
                if (!decoder.decode(image, timing, measurements)) {
                    continue;
                }
                last_decoded_imu_count = measurements.size();
                decoded_imu_count += measurements.size();
                if (!measurements.empty()) {
                    last_decoded_imu_first = measurements.front().timestamp;
                    last_decoded_imu_last = measurements.back().timestamp;
                }
                if (previous_exposure_end != tassel_utils::kInvalidFrameId) {
                    sensor_interval_us = (timing.exposure_end - previous_exposure_end) / 1000;
                }
                previous_exposure_end = timing.exposure_end;
                // 硬件左右目在拼接图中的顺序不由类型表达，运行入口必须显式选择半幅。
                const int half_cols = image.cols / 2;
                const int x0 = mono_half == MonoHalf::First ? 0 : half_cols;
                cv::Mat mono_gray = image.colRange(x0, x0 + half_cols);
                if (source_width != mono_gray.cols || source_height != mono_gray.rows) {
                    throw std::logic_error("Nori source calibration does not match the mono image");
                }
                cv::Mat tracking_image;
                if (tracking_scale < 1.0) {
                    cv::resize(
                        mono_gray, tracking_image, cv::Size(params.cols, params.rows), 0.0, 0.0,
                        cv::INTER_AREA);
                } else {
                    tracking_image = mono_gray.clone();
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

    PipelineQueue<TrackedFrame> tracked_queue;
    std::exception_ptr tracking_error;
    std::thread tracking_thread([&]() {
        try {
            SyncedFrame synced;
            tassel_utils::FrameId previous_frame_id = tassel_utils::kInvalidFrameId;
            while (rclcpp::ok() && sync.waitPop(synced)) {
                const auto tracking_start = std::chrono::steady_clock::now();
                std::optional<tassel_core::TrackingPredictionSnapshot> snapshot;
                {
                    std::unique_lock<std::mutex> lock(prediction_mutex);
                    if (prediction_snapshot && previous_frame_id != tassel_utils::kInvalidFrameId) {
                        // 闭环启用后，预测快照必须对应追踪器的上一图像帧；等待期间采集与解码继续运行。
                        prediction_cv.wait(lock, [&]() {
                            return stop_pipeline.load() || !rclcpp::ok() ||
                                   (prediction_snapshot &&
                                    prediction_snapshot->source_frame_id >= previous_frame_id);
                        });
                        if (prediction_snapshot &&
                            prediction_snapshot->source_frame_id == previous_frame_id) {
                            snapshot = prediction_snapshot;
                        }
                    }
                }
                if (stop_pipeline.load() || !rclcpp::ok()) {
                    break;
                }
                std::unordered_map<int, cv::Point2f> predicted_pixels;
                if (snapshot) {
                    predicted_pixels = tassel_core::predictLandmarkPixelsFromSnapshot(
                        *snapshot, synced.mono.frame_id, synced.imu, kNoriSyncDelay, *camera_ptr,
                        params);
                }
                auto features = tracker.monoTracking(synced.mono.image, predicted_pixels);
                previous_frame_id = synced.mono.frame_id;
                const size_t tracked_index = tracked_count.load();
                if (publish_tracking_image &&
                    tracked_index % static_cast<size_t>(visualization_stride) == 0) {
                    cv::Mat tracking_image = synced.mono.image.clone();
                    tracker.drawTrackingResult(tracking_image);
                    {
                        std::lock_guard<std::mutex> lock(latest_image_mutex);
                        latest_image.image = std::move(tracking_image);
                        latest_image.timestamp =
                            tassel_utils::frameIdToSeconds(synced.mono.frame_id);
                    }
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
            prediction_cv.notify_all();
        }
        tracked_queue.stop();
    });

    std::exception_ptr estimator_error;
    std::thread estimator_thread([&]() {
        try {
            int processed = 0;
            TrackedFrame packet;
            while (rclcpp::ok() && tracked_queue.waitPop(packet)) {
                const auto frame_id = packet.mono.frame_id;
                estimator.processMeasurement(frame_id, packet.features, packet.imu, kNoriSyncDelay);
                if (auto snapshot = estimator.makeTrackingPredictionSnapshot()) {
                    {
                        std::lock_guard<std::mutex> lock(prediction_mutex);
                        prediction_snapshot = std::move(snapshot);
                    }
                    prediction_cv.notify_all();
                }
                ++processed;
                if (processed % 10 == 0) {
                    const double elapsed = std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - pipeline_start)
                                               .count();
                    const size_t decoded = decoded_count.load();
                    const size_t tracked = tracked_count.load();
                    const long long interval_us = sensor_interval_us.load();
                    const double synced_imu_first =
                        packet.imu.empty() ? 0.0 : packet.imu.front().timestamp;
                    const double synced_imu_last =
                        packet.imu.empty() ? 0.0 : packet.imu.back().timestamp;
                    std::cout << "[nori] frames=" << processed << " imu=" << packet.imu.size()
                              << " imu_span=" << synced_imu_first << ':' << synced_imu_last
                              << " decoded_imu_last=" << last_decoded_imu_count.load()
                              << " decoded_imu_span=" << last_decoded_imu_first.load() << ':'
                              << last_decoded_imu_last.load() << " decoded_imu_avg="
                              << (decoded > 0
                                      ? static_cast<double>(decoded_imu_count.load()) / decoded
                                      : 0.0)
                              << " features=" << packet.features.size()
                              << " raw_queue=" << encoded_queue.size()
                              << " sync_queue=" << sync.size()
                              << " tracking_queue=" << tracked_queue.size()
                              << " rates=" << captured_count.load() / elapsed << '/'
                              << decoded / elapsed << '/' << tracked / elapsed << '/'
                              << processed / elapsed << " sensor_fps="
                              << (interval_us > 0 ? 1e6 / static_cast<double>(interval_us) : 0.0)
                              << " decode_ms="
                              << (decoded > 0 ? decode_time_us.load() / (1000.0 * decoded) : 0.0)
                              << " tracking_ms="
                              << (tracked > 0 ? tracking_time_us.load() / (1000.0 * tracked) : 0.0)
                              << '\n';
                }
            }
        } catch (...) {
            estimator_error = std::current_exception();
            stop_pipeline = true;
            encoded_queue.stop();
            sync.stop();
            tracked_queue.stop();
            prediction_cv.notify_all();
        }
    });
    estimator_thread.join();
    if (estimator_error) {
        std::rethrow_exception(estimator_error);
    }
    stop_pipeline = true;
    encoded_queue.stop();
    sync.stop();
    prediction_cv.notify_all();
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
