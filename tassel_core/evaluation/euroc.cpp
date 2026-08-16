// =============================================================================
// euroc.cpp
//
// 目的：
//   EuRoC MAV Machine Hall 单目序列的离线集成入口。
//
// 用法：
//   test_euroc [config.yaml] [sequence_dir] [replay_hz] [rtabmap_database]
//
// 示例：
//   unzip dataset/machine_hall/MH_01_easy/MH_01_easy.zip -d dataset/machine_hall/MH_01_easy
//   ./build/tassel_core/test_euroc config/euroc.yaml dataset/machine_hall/MH_01_easy 20
// =============================================================================

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rclcpp/executors/single_threaded_executor.hpp>

#include "cam/camera_factory.h"
#include "estimator/estimator.h"
#include "evaluation/trajectory_evaluator.h"
#include "frond_end/feature_manager.h"
#include "frond_end/feature_tracker.h"
#include "parameters/parameters.h"
#include "rtabmap/rtabmap_backend.h"
#include "state/state.h"
#include "viewer/viewer.h"

namespace fs = std::filesystem;

namespace {

// 20 Hz 轨迹评估只允许在不超过一个图像周期的连续真值之间插值。
constexpr double kMaxGroundTruthInterval = 0.05;

struct ImageEntry {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    std::string timestamp_ns;
    std::string filename;
};

struct MonoFrame {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    fs::path left_path;
};

struct MonoObservation {
    tassel_utils::FrameId timestamp = tassel_utils::kInvalidFrameId;
    cv::Mat left_img;

    double get_timestamp() const { return tassel_utils::frameIdToSeconds(timestamp); }
};

using tassel_core::evaluation::PosePair;
using tassel_core::evaluation::TimedPose;

struct LatestDisplayImages {
    cv::Mat mono;
    cv::Mat tracking;
    double timestamp = -1.0;
};

struct SyncedPacket {
    std::shared_ptr<MonoObservation> mono;
    std::vector<tassel_utils::IMUMeasurement> imu_slice;
    double sync_delay = 0.0;
};

class BlockingDatasetSync {
public:
    void pushMono(std::shared_ptr<MonoObservation> mono) {
        if (!mono) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            mono_queue_.push_back(std::move(mono));
        }
        cv_.notify_one();
    }

    void pushImu(const tassel_utils::IMUMeasurement& imu) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            imu_queue_.push_back(imu);
        }
        cv_.notify_one();
    }

    void closeMono() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            mono_done_ = true;
        }
        cv_.notify_all();
    }

    void closeImu() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            imu_done_ = true;
        }
        cv_.notify_all();
    }

    bool waitPop(SyncedPacket& packet, double sync_delay) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
            if (mono_queue_.empty()) {
                return mono_done_;
            }
            const double sync_ts = mono_queue_.front()->get_timestamp() + sync_delay;
            return imu_done_ || (!imu_queue_.empty() && imu_queue_.back().timestamp >= sync_ts);
        });

        if (mono_queue_.empty()) {
            return false;
        }
        const double sync_ts = mono_queue_.front()->get_timestamp() + sync_delay;
        if (imu_queue_.empty() || imu_queue_.back().timestamp < sync_ts) {
            std::cerr << "[EuRoC] missing IMU coverage for image t="
                      << mono_queue_.front()->get_timestamp() << " sync t=" << sync_ts
                      << " sync_delay=" << sync_delay << "\n";
            return false;
        }

        packet.mono = std::move(mono_queue_.front());
        mono_queue_.pop_front();
        packet.sync_delay = sync_delay;

        packet.imu_slice.clear();
        if (has_boundary_) {
            packet.imu_slice.push_back(boundary_imu_);
        }

        const double prev_ts = has_boundary_ ? boundary_imu_.timestamp : -1.0;
        for (const auto& imu : imu_queue_) {
            if (imu.timestamp >= sync_ts) {
                break;
            }
            if (prev_ts < 0.0 || imu.timestamp > prev_ts) {
                packet.imu_slice.push_back(imu);
            }
        }

        auto boundary = interpolateBoundary(sync_ts);
        if (packet.imu_slice.empty() || packet.imu_slice.back().timestamp < boundary.timestamp) {
            packet.imu_slice.push_back(boundary);
        }

        while (!imu_queue_.empty() && imu_queue_.front().timestamp <= sync_ts) {
            imu_queue_.pop_front();
        }
        boundary_imu_ = boundary;
        has_boundary_ = true;
        return true;
    }

private:
    tassel_utils::IMUMeasurement interpolateBoundary(double ts) const {
        auto it = std::lower_bound(
            imu_queue_.begin(), imu_queue_.end(), ts,
            [](const tassel_utils::IMUMeasurement& imu, double value) {
                return imu.timestamp < value;
            });

        if (it != imu_queue_.end() && it->timestamp == ts) {
            return *it;
        }

        const tassel_utils::IMUMeasurement* p0 = nullptr;
        const tassel_utils::IMUMeasurement* p1 = nullptr;
        if (it != imu_queue_.begin()) {
            p0 = &(*std::prev(it));
        } else if (has_boundary_) {
            p0 = &boundary_imu_;
        }
        if (it != imu_queue_.end()) {
            p1 = &(*it);
        }

        if (p0 && p1 && p1->timestamp > p0->timestamp) {
            tassel_utils::IMUMeasurement out;
            out.timestamp = ts;
            double alpha = (ts - p0->timestamp) / (p1->timestamp - p0->timestamp);
            out.acc = p0->acc + alpha * (p1->acc - p0->acc);
            out.gyro = p0->gyro + alpha * (p1->gyro - p0->gyro);
            return out;
        }

        if (p0) {
            auto out = *p0;
            out.timestamp = ts;
            return out;
        }
        if (p1) {
            auto out = *p1;
            out.timestamp = ts;
            return out;
        }
        return {};
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<MonoObservation>> mono_queue_;
    std::deque<tassel_utils::IMUMeasurement> imu_queue_;
    bool mono_done_ = false;
    bool imu_done_ = false;
    bool has_boundary_ = false;
    tassel_utils::IMUMeasurement boundary_imu_;
};

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        field.erase(field.begin(), std::find_if(field.begin(), field.end(), [](unsigned char ch) {
                        return !std::isspace(ch);
                    }));
        field.erase(
            std::find_if(
                field.rbegin(), field.rend(), [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            field.end());
        fields.push_back(field);
    }
    return fields;
}

double nsToSec(const std::string& value) { return std::stod(value) * 1e-9; }

std::vector<ImageEntry> loadImageCsv(const fs::path& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image csv: " + csv_path.string());
    }

    std::vector<ImageEntry> entries;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto fields = splitCsvLine(line);
        if (fields.size() < 2) {
            continue;
        }
        entries.push_back(
            {static_cast<tassel_utils::FrameId>(std::stoll(fields[0])), fields[0], fields[1]});
    }
    return entries;
}

std::vector<tassel_utils::IMUMeasurement> loadImuCsv(const fs::path& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open IMU csv: " + csv_path.string());
    }

    std::vector<tassel_utils::IMUMeasurement> measurements;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto fields = splitCsvLine(line);
        if (fields.size() < 7) {
            continue;
        }
        tassel_utils::IMUMeasurement m;
        m.timestamp = nsToSec(fields[0]);
        m.gyro << std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3]);
        m.acc << std::stod(fields[4]), std::stod(fields[5]), std::stod(fields[6]);
        measurements.push_back(m);
    }
    return measurements;
}

std::vector<TimedPose> loadGroundTruthCsv(const fs::path& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ground truth csv: " + csv_path.string());
    }

    std::vector<TimedPose> poses;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto fields = splitCsvLine(line);
        if (fields.size() < 8) {
            continue;
        }

        Eigen::Vector3d position(std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3]));
        Eigen::Quaterniond orientation(
            std::stod(fields[4]), std::stod(fields[5]), std::stod(fields[6]), std::stod(fields[7]));
        orientation.normalize();
        poses.push_back({nsToSec(fields[0]), Sophus::SE3d(orientation, position)});
    }
    return poses;
}

void reportTrajectoryError(const std::vector<PosePair>& poses, size_t skipped_truth_samples) {
    if (poses.empty()) {
        std::cout << "[EuRoC] no synchronized ground-truth poses for evaluation"
                  << " skipped_truth_samples=" << skipped_truth_samples
                  << " max_truth_interval=" << kMaxGroundTruthInterval << " s\n";
        return;
    }

    const auto error = tassel_core::evaluation::evaluateTrajectory(poses);
    std::ostringstream report;
    report << std::setprecision(12)
           << "[EuRoC] trajectory evaluation (single global yaw+translation alignment): "
           << "samples=" << poses.size() << " skipped_truth_samples=" << skipped_truth_samples
           << " max_truth_interval=" << kMaxGroundTruthInterval << " s"
           << " time_range=[" << poses.front().timestamp << ", " << poses.back().timestamp << "] s"
           << " ATE_RMSE=" << error.position_rmse << " m"
           << " terminal_position_error=" << error.terminal_position_error << " m"
           << " rotation_RMSE=" << error.rotation_rmse << " rad";
    std::cout << report.str() << "\n";
}

fs::path resolveSequenceDir(const fs::path& sequence_dir) {
    if (fs::exists(sequence_dir / "mav0" / "cam0" / "data.csv")) {
        return sequence_dir;
    }

    const fs::path nested_sequence_dir = sequence_dir / sequence_dir.filename();
    if (fs::exists(nested_sequence_dir / "mav0" / "cam0" / "data.csv")) {
        return nested_sequence_dir;
    }

    return sequence_dir;
}

std::vector<MonoFrame> makeMonoFrames(const fs::path& sequence_dir) {
    const fs::path cam0_dir = sequence_dir / "mav0" / "cam0";
    auto left_entries = loadImageCsv(cam0_dir / "data.csv");

    std::vector<MonoFrame> frames;
    frames.reserve(left_entries.size());
    for (const auto& left : left_entries) {
        frames.push_back({left.frame_id, cam0_dir / "data" / left.filename});
    }
    return frames;
}

tassel_core::Camera initializeCamera(const tassel_tools::Parameters& params) {
    return tassel_core::CameraFactory::create(
        params.camera_model, params.cam_intrinsic, params.cam_distort, params.cols, params.rows);
}

void publishDisplayImages(
    const std::shared_ptr<tassel_tools::Viewer>& viewer, const LatestDisplayImages& images) {
    if (images.mono.empty() || images.tracking.empty()) {
        return;
    }
    viewer->publishImage("mono/image", "camera_optical_frame", images.mono, images.timestamp);
    viewer->publishImage(
        "tracking/image", "camera_optical_frame", images.tracking, images.timestamp);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tassel_core;

    const fs::path config_path =
        (argc >= 2) ? fs::path(argv[1]) : fs::path("/home/adrewn/Tassel/config/euroc.yaml");
    const fs::path sequence_dir = (argc >= 3) ? fs::path(argv[2])
                                              : fs::path(
                                                    "/home/adrewn/Tassel/datasets/"
                                                    "machine_hall/MH_01_easy");
    const double replay_hz = (argc >= 4) ? std::stod(argv[3]) : 20.0;
    const std::string rtabmap_database = (argc >= 5) ? argv[4] : "";
    if (!std::isfinite(replay_hz) || replay_hz <= 0.0) {
        throw std::invalid_argument("replay_hz must be finite and positive");
    }
    constexpr bool kHeadless = false;

    const fs::path resolved_sequence_dir = resolveSequenceDir(sequence_dir);
    if (!fs::exists(resolved_sequence_dir / "mav0" / "cam0" / "data.csv")) {
        std::cerr << "[EuRoC] sequence is not extracted: " << sequence_dir << "\n"
                  << "        expected mav0/cam0/data.csv directly under the sequence directory "
                     "or under a same-named child directory\n";
        return 0;
    }

    tassel_tools::Parameters params(config_path.string());
    auto frames = makeMonoFrames(resolved_sequence_dir);
    auto imu_measurements = loadImuCsv(resolved_sequence_dir / "mav0" / "imu0" / "data.csv");
    fs::path ground_truth_path =
        resolved_sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv";
    if (!fs::exists(ground_truth_path)) {
        ground_truth_path = resolved_sequence_dir / "mav0" / "mocap0" / "data.csv";
    }
    auto ground_truth = loadGroundTruthCsv(ground_truth_path);

    if (frames.empty() || imu_measurements.empty()) {
        std::cerr << "[EuRoC] empty image or IMU stream under " << resolved_sequence_dir << "\n";
        return 1;
    }

    auto camera = initializeCamera(params);

    rclcpp::init(argc, argv);
    std::shared_ptr<tassel_tools::Viewer> viewer;
    if (!kHeadless) {
        viewer = std::make_shared<tassel_tools::Viewer>("odom");
        rclcpp::QoS image_qos(rclcpp::KeepLast(1));
        image_qos.best_effort().durability_volatile();
        viewer->createImagePublisher("mono/image", image_qos);
        viewer->createImagePublisher("tracking/image", image_qos);
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
        viewer->createVector3Publisher("vio/ba");
        viewer->createVector3Publisher("vio/bg");
    }

    rclcpp::executors::SingleThreadedExecutor executor;
    if (viewer) {
        executor.add_node(viewer);
    }
    std::atomic_bool stop_executor{false};
    std::thread spin_thread;
    if (!kHeadless) {
        spin_thread = std::thread([&]() {
            while (rclcpp::ok() && !stop_executor.load()) {
                executor.spin_some(std::chrono::milliseconds(5));
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    const CameraBase* camera_ptr = camera.get();
    FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    tracker.setCamera(
        std::move(camera), params.per_grid_rows, params.per_grid_cols, params.mask_radius,
        params.min_feature_num);

    auto state = std::make_shared<State>(static_cast<int>(params.max_frame_count) + 1);
    auto feature_manager = std::make_shared<FeatureManager>(
        params.reproj_err_thres, params.min_landmark_observations, params.parallax_threshold,
        params.keyframe_min_connection_ratio, params.min_depth, params.max_depth);

    Estimator estimator(params, state, feature_manager);
    state->camera = camera_ptr;
    estimator.setCamera(camera_ptr);
    std::unique_ptr<tassel_tools::RtabmapBackend> rtabmap_backend;
    if (!rtabmap_database.empty()) {
        tassel_tools::RtabmapBackendOptions options;
        options.database_path = rtabmap_database;
        rtabmap_backend = std::make_unique<tassel_tools::RtabmapBackend>(
            params.cam_intrinsic, params.cam_distort, cv::Size(params.cols, params.rows),
            params.camera_model == "equi", Sophus::SE3d(params.ric, params.tic), options);
    }
    const size_t frame_limit = frames.size();
    std::optional<Sophus::SE3d> ground_truth_alignment;
    std::vector<PosePair> evaluated_poses;
    evaluated_poses.reserve(frame_limit);
    size_t skipped_truth_samples = 0;
    std::optional<Sophus::SE3d> latest_optimized_pose;
    std::optional<std::array<double, 9>> latest_current_speed_bias;
    std::optional<std::array<double, 9>> latest_linearized_speed_bias;
    estimator.setPoseCallback([&viewer, &state, &ground_truth, &ground_truth_alignment,
                               &evaluated_poses, &skipped_truth_samples, &params,
                               &latest_optimized_pose, &latest_current_speed_bias,
                               &latest_linearized_speed_bias](double ts, const Sophus::SE3d& pose) {
        latest_optimized_pose = pose;
        const FrameState& frame = state->frames[state->latest_active_frame_index];
        latest_current_speed_bias = frame.param_speed_bias;
        latest_linearized_speed_bias = frame.linearized_speed_bias;
        if (viewer) {
            // Odometry twist 必须表达在 child_frame_id；V 是世界系速度，gyro-Bg 是 IMU 系角速度。
            const Eigen::Vector3d body_velocity = frame.rot_w_i.transpose() * frame.vel_w;
            const Eigen::Vector3d body_angular_velocity = frame.imu_gyro - frame.gyro_bias;
            viewer->publishOdometry(
                "vio/odometry", pose.translation(), pose.unit_quaternion(), body_velocity,
                body_angular_velocity, ts);
            const Sophus::SE3d camera_pose = pose * Sophus::SE3d(params.ric, params.tic);
            const Eigen::Vector3d camera_velocity =
                params.ric.transpose() * (body_velocity + body_angular_velocity.cross(params.tic));
            const Eigen::Vector3d camera_angular_velocity =
                params.ric.transpose() * body_angular_velocity;
            viewer->publishOdometry(
                "vio/camera_odometry", camera_pose.translation(), camera_pose.unit_quaternion(),
                camera_velocity, camera_angular_velocity, ts);
            viewer->publishPath("vio/path", pose.translation(), pose.unit_quaternion(), ts);
        }
        // R/P/V 位于图像时间加同步延迟的 IMU 时刻；视觉因子另行补偿 time_delay-sync_delay。
        const double evaluation_ts = ts + frame.image_sync_delay;
        const auto truth = tassel_core::evaluation::interpolatePose(
            ground_truth, evaluation_ts, kMaxGroundTruthInterval);
        if (truth) {
            evaluated_poses.push_back({evaluation_ts, pose, *truth});
            if (!ground_truth_alignment) {
                // 仅用于 Foxglove 叠加显示；正式评估在所有样本收集后统一对齐。
                ground_truth_alignment = pose * truth->inverse();
            }
            const Sophus::SE3d aligned_truth = *ground_truth_alignment * *truth;
            if (viewer) {
                viewer->publishPath(
                    "ground_truth/path", aligned_truth.translation(),
                    aligned_truth.unit_quaternion(), ts);
            }
        } else {
            ++skipped_truth_samples;
        }
        const Eigen::Vector3d& velocity = state->frames[state->latest_active_frame_index].vel_w;
        if (viewer) {
            viewer->publishVector3(
                "vio/ba", state->frames[state->latest_active_frame_index].accel_bias, ts);
            viewer->publishVector3(
                "vio/bg", state->frames[state->latest_active_frame_index].gyro_bias, ts);
        }
        std::cout << "[pose] t=" << ts << " p=" << pose.translation().transpose()
                  << " |V|=" << velocity.norm() << "\n";
    });
    if (viewer) {
        estimator.setVisualFactorCallback(
            [&viewer](double ts, const std::vector<int>& visual_factors_per_frame) {
                viewer->publishVisualFactorWindow(
                    "optimization/visual_window", visual_factors_per_frame, ts);
            });
    }
    BlockingDatasetSync sync;

    std::atomic_bool imu_done{false};
    std::atomic_bool stop_reader{false};
    std::atomic_bool stop_image_publisher{false};
    std::atomic_size_t produced{0};
    std::atomic_size_t produced_imu{0};
    std::mutex latest_image_mutex;
    LatestDisplayImages latest_images;
    std::mutex loaded_mono_mutex;
    std::condition_variable loaded_mono_cv;
    std::deque<std::shared_ptr<MonoObservation>> loaded_mono_queue;
    bool mono_load_done = false;
    constexpr size_t kMaxLoadedMonoFrames = 30;
    const size_t preload_target = std::min(frame_limit, kMaxLoadedMonoFrames);
    std::vector<std::shared_ptr<MonoObservation>> preloaded_mono;
    if (kHeadless) {
        // Headless 离线入口预读图像，避免数据读取线程影响确定性的回放顺序。
        preloaded_mono.reserve(frame_limit);
        for (size_t i = 0; i < frame_limit; ++i) {
            const auto& frame = frames[i];
            cv::Mat left_img = cv::imread(frame.left_path.string(), cv::IMREAD_GRAYSCALE);
            if (left_img.empty()) {
                std::cerr << "[EuRoC] failed to read mono image at t="
                          << tassel_utils::frameIdToSeconds(frame.frame_id) << "\n";
                break;
            }

            auto mono_msg = std::make_shared<MonoObservation>();
            mono_msg->timestamp = frame.frame_id;
            mono_msg->left_img = std::move(left_img);
            preloaded_mono.push_back(std::move(mono_msg));
        }
        if (preloaded_mono.size() != frame_limit) {
            throw std::runtime_error("Failed to preload the full mono sequence");
        }
    }

    std::thread mono_loader_thread;
    if (!kHeadless) {
        mono_loader_thread = std::thread([&]() {
            for (size_t i = 0; i < frame_limit && rclcpp::ok() && !stop_reader.load(); ++i) {
                const auto& frame = frames[i];
                cv::Mat left_img = cv::imread(frame.left_path.string(), cv::IMREAD_GRAYSCALE);
                if (left_img.empty()) {
                    std::cerr << "[EuRoC] failed to read mono image at t="
                              << tassel_utils::frameIdToSeconds(frame.frame_id) << "\n";
                    break;
                }

                auto mono_msg = std::make_shared<MonoObservation>();
                mono_msg->timestamp = frame.frame_id;
                mono_msg->left_img = std::move(left_img);

                {
                    std::unique_lock<std::mutex> lock(loaded_mono_mutex);
                    loaded_mono_cv.wait(lock, [&]() {
                        return stop_reader.load() ||
                               loaded_mono_queue.size() < kMaxLoadedMonoFrames;
                    });
                    if (stop_reader.load()) {
                        break;
                    }
                    loaded_mono_queue.push_back(std::move(mono_msg));
                }
                loaded_mono_cv.notify_all();
            }

            {
                std::lock_guard<std::mutex> lock(loaded_mono_mutex);
                mono_load_done = true;
            }
            loaded_mono_cv.notify_all();
        });

        {
            std::unique_lock<std::mutex> lock(loaded_mono_mutex);
            loaded_mono_cv.wait(lock, [&]() {
                return loaded_mono_queue.size() >= preload_target || mono_load_done ||
                       !rclcpp::ok();
            });
        }
    }

    const double first_frame_ts = tassel_utils::frameIdToSeconds(frames.front().frame_id);
    const double nominal_frame_dt =
        (frame_limit > 1) ? tassel_utils::frameIdToSeconds(frames[1].frame_id - frames[0].frame_id)
                          : (1.0 / replay_hz);
    const double playback_end_ts =
        tassel_utils::frameIdToSeconds(frames[frame_limit - 1].frame_id) + nominal_frame_dt;
    const double playback_scale = (1.0 / replay_hz) / nominal_frame_dt;
    const double playback_start_ts = first_frame_ts;
    const auto playback_start_time = std::chrono::steady_clock::now();

    auto sleep_until_sensor_time = [&](double sensor_ts) {
        const double replay_elapsed =
            std::max(0.0, (sensor_ts - playback_start_ts) * playback_scale);
        const auto target_time =
            playback_start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(replay_elapsed));
        std::this_thread::sleep_until(target_time);
    };

    std::thread image_publish_thread;
    if (!kHeadless) {
        image_publish_thread = std::thread([&]() {
            const auto period = std::chrono::duration<double>(1.0 / replay_hz);
            auto next_tick = std::chrono::steady_clock::now();
            while (rclcpp::ok() && !stop_image_publisher.load()) {
                LatestDisplayImages images;
                {
                    std::lock_guard<std::mutex> lock(latest_image_mutex);
                    images.mono = latest_images.mono.clone();
                    images.tracking = latest_images.tracking.clone();
                    images.timestamp = latest_images.timestamp;
                }
                publishDisplayImages(viewer, images);

                next_tick +=
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
                std::this_thread::sleep_until(next_tick);
            }
        });
    }

    std::thread imu_reader_thread;
    std::thread mono_reader_thread;
    if (kHeadless) {
        // Headless 模式一次性灌入数据，避免真实时间调度拖慢离线评估。
        for (const auto& imu : imu_measurements) {
            if (imu.timestamp > playback_end_ts) {
                break;
            }
            if (imu.timestamp >= playback_start_ts) {
                sync.pushImu(imu);
                ++produced_imu;
            }
        }
        sync.closeImu();

        for (auto& mono_msg : preloaded_mono) {
            sync.pushMono(mono_msg);
            ++produced;
        }
        sync.closeMono();
    } else {
        imu_reader_thread = std::thread([&]() {
            for (const auto& imu : imu_measurements) {
                if (!rclcpp::ok() || stop_reader.load()) {
                    break;
                }
                if (imu.timestamp > playback_end_ts) {
                    break;
                }
                if (imu.timestamp > playback_start_ts) {
                    sleep_until_sensor_time(imu.timestamp);
                }
                if (!rclcpp::ok() || stop_reader.load()) {
                    break;
                }
                sync.pushImu(imu);
                ++produced_imu;
            }
            imu_done = true;
            sync.closeImu();
        });

        mono_reader_thread = std::thread([&]() {
            while (rclcpp::ok() && !stop_reader.load()) {
                std::shared_ptr<MonoObservation> mono_msg;
                {
                    std::unique_lock<std::mutex> lock(loaded_mono_mutex);
                    loaded_mono_cv.wait(lock, [&]() {
                        return stop_reader.load() || !loaded_mono_queue.empty() || mono_load_done;
                    });
                    if (stop_reader.load()) {
                        break;
                    }
                    if (loaded_mono_queue.empty()) {
                        if (mono_load_done) {
                            break;
                        }
                        continue;
                    }
                    mono_msg = std::move(loaded_mono_queue.front());
                    loaded_mono_queue.pop_front();
                }
                loaded_mono_cv.notify_all();

                sleep_until_sensor_time(mono_msg->get_timestamp());
                if (!rclcpp::ok() || stop_reader.load()) {
                    break;
                }
                sync.pushMono(mono_msg);
                ++produced;
            }

            sync.closeMono();
        });
    }

    size_t processed = 0;
    std::map<tassel_utils::FrameId, cv::Mat> rtabmap_images;
    while (rclcpp::ok()) {
        SyncedPacket packet;
        if (!sync.waitPop(packet, state->time_delay)) {
            break;
        }
        if (!packet.mono) {
            continue;
        }

        std::unordered_map<int, FeaturePerFrame> feature_frame;
        feature_frame = tracker.monoTracking(packet.mono->left_img);
        for (auto& [id, feature] : feature_frame) {
            (void)id;
            feature.sync_delay = packet.sync_delay;
        }

        if (!kHeadless) {
            cv::Mat left_tracking = packet.mono->left_img.clone();
            tracker.drawTrackingResult(left_tracking);
            {
                std::lock_guard<std::mutex> lock(latest_image_mutex);
                latest_images.mono = packet.mono->left_img.clone();
                latest_images.tracking = std::move(left_tracking);
                latest_images.timestamp = packet.mono->get_timestamp();
            }
        }
        latest_optimized_pose.reset();
        if (rtabmap_backend) {
            rtabmap_images[packet.mono->timestamp] = packet.mono->left_img;
        }
        estimator.processMeasurement(
            packet.mono->timestamp, feature_frame, packet.imu_slice, packet.sync_delay);
        if (rtabmap_backend && estimator.lastRetainedKeyframe()) {
            const auto& keyframe = *estimator.lastRetainedKeyframe();
            const auto image = rtabmap_images.find(keyframe.frame_id);
            if (image == rtabmap_images.end()) {
                throw std::logic_error("Retained keyframe image is missing");
            }
            rtabmap_backend->submit(
                keyframe.frame_id, image->second, keyframe.pose, keyframe.landmarks);
            rtabmap_images.erase(rtabmap_images.begin(), std::next(image));
        }
        if (rtabmap_backend && !estimator.lastMeasurementWasKeyframe()) {
            rtabmap_images.erase(packet.mono->timestamp);
        }
        ++processed;

        if (processed % 20 == 0) {
            std::cout << "[EuRoC] processed " << processed << "/" << frame_limit << " (read "
                      << produced.load() << ", imu read " << produced_imu.load() << ")"
                      << " mono frames, features=" << feature_frame.size()
                      << ", imu=" << packet.imu_slice.size() << "\n";
        }
    }

    stop_reader = true;
    loaded_mono_cv.notify_all();
    stop_image_publisher = true;
    if (mono_loader_thread.joinable()) {
        mono_loader_thread.join();
    }
    if (mono_reader_thread.joinable()) {
        mono_reader_thread.join();
    }
    if (imu_reader_thread.joinable()) {
        imu_reader_thread.join();
    }
    if (image_publish_thread.joinable()) {
        image_publish_thread.join();
    }
    stop_executor = true;
    executor.cancel();
    if (spin_thread.joinable()) {
        spin_thread.join();
    }

    rclcpp::shutdown();

    std::cout << "\n[EuRoC] done. processed=" << processed
              << ", newest frame_index=" << state->latest_active_frame_index << "\n";
    reportTrajectoryError(evaluated_poses, skipped_truth_samples);
    if (latest_optimized_pose && latest_current_speed_bias && latest_linearized_speed_bias) {
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> current_speed_bias(
            latest_current_speed_bias->data());
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> linearized_speed_bias(
            latest_linearized_speed_bias->data());
        std::cout
            << "Final pose:\n"
            << latest_optimized_pose->matrix() << "\n"
            << "Final ba current=" << current_speed_bias.segment<3>(3).transpose()
            << " FEJ=" << linearized_speed_bias.segment<3>(3).transpose() << " delta="
            << (current_speed_bias.segment<3>(3) - linearized_speed_bias.segment<3>(3)).transpose()
            << "\n"
            << "Final bg current=" << current_speed_bias.segment<3>(6).transpose()
            << " FEJ=" << linearized_speed_bias.segment<3>(6).transpose() << " delta="
            << (current_speed_bias.segment<3>(6) - linearized_speed_bias.segment<3>(6)).transpose()
            << "\n";
    }

    return 0;
}
