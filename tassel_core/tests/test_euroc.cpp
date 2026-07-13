// =============================================================================
// test_euroc.cpp
//
// Purpose:
//   Offline integration entry for EuRoC MAV simple Machine Hall stereo sequences.
//
// Usage:
//   test_euroc [config.yaml] [sequence_dir] [max_frames=0(all)] [replay_hz]
//
// Example:
//   unzip dataset/machine_hall/MH_01_easy/MH_01_easy.zip -d dataset/machine_hall/MH_01_easy
//   ./build/tassel_core/test_euroc config/euroc.yaml dataset/machine_hall/MH_01_easy 0 20
// =============================================================================

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rclcpp/executors/single_threaded_executor.hpp>

#include "cam/camera_factory.h"
#include "estimator/estimator.h"
#include "frond_end/feature_manager.h"
#include "frond_end/feature_tracker.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/timer.h"
#include "viewer/viewer.h"

namespace fs = std::filesystem;

namespace {

struct ImageEntry {
    double timestamp = 0.0;
    std::string timestamp_ns;
    std::string filename;
};

struct StereoFrame {
    double timestamp = 0.0;
    fs::path left_path;
    fs::path right_path;
};

struct LatestDisplayImage {
    cv::Mat image;
};

struct SyncedPacket {
    std::shared_ptr<tassel_utils::StereoObservation> stereo;
    std::vector<tassel_utils::IMUMeasurement> imu_slice;
    double applied_delay = 0.0;
};

class BlockingDatasetSync {
public:
    void pushStereo(std::shared_ptr<tassel_utils::StereoObservation> stereo) {
        if (!stereo) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stereo_queue_.push_back(std::move(stereo));
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

    void closeStereo() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stereo_done_ = true;
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

    bool waitPop(SyncedPacket& packet, double applied_delay) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
            if (stereo_queue_.empty()) return stereo_done_;
            const double sync_ts = stereo_queue_.front()->timestamp + applied_delay;
            return imu_done_ || (!imu_queue_.empty() && imu_queue_.back().timestamp >= sync_ts);
        });

        if (stereo_queue_.empty()) {
            return false;
        }
        const double sync_ts = stereo_queue_.front()->timestamp + applied_delay;
        if (imu_queue_.empty() || imu_queue_.back().timestamp < sync_ts) {
            std::cerr << "[EuRoC] missing IMU coverage for stereo t="
                      << stereo_queue_.front()->timestamp << " sync t=" << sync_ts
                      << " applied_delay=" << applied_delay << "\n";
            return false;
        }

        packet.stereo = std::move(stereo_queue_.front());
        stereo_queue_.pop_front();
        packet.applied_delay = applied_delay;

        packet.imu_slice.clear();
        if (has_boundary_) {
            packet.imu_slice.push_back(boundary_imu_);
        }

        const double prev_ts = has_boundary_ ? boundary_imu_.timestamp : -1.0;
        for (const auto& imu : imu_queue_) {
            if (imu.timestamp >= sync_ts) break;
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
    std::deque<std::shared_ptr<tassel_utils::StereoObservation>> stereo_queue_;
    std::deque<tassel_utils::IMUMeasurement> imu_queue_;
    bool stereo_done_ = false;
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
        entries.push_back({nsToSec(fields[0]), fields[0], fields[1]});
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

std::vector<StereoFrame> makeStereoFrames(const fs::path& sequence_dir) {
    const fs::path cam0_dir = sequence_dir / "mav0" / "cam0";
    const fs::path cam1_dir = sequence_dir / "mav0" / "cam1";
    auto left_entries = loadImageCsv(cam0_dir / "data.csv");
    auto right_entries = loadImageCsv(cam1_dir / "data.csv");

    std::unordered_map<std::string, std::string> right_by_timestamp;
    for (const auto& entry : right_entries) {
        right_by_timestamp[entry.timestamp_ns] = entry.filename;
    }

    std::vector<StereoFrame> frames;
    frames.reserve(std::min(left_entries.size(), right_entries.size()));
    for (const auto& left : left_entries) {
        auto it = right_by_timestamp.find(left.timestamp_ns);
        if (it == right_by_timestamp.end()) {
            continue;
        }
        frames.push_back(
            {left.timestamp, cam0_dir / "data" / left.filename, cam1_dir / "data" / it->second});
    }
    return frames;
}

std::vector<tassel_core::Camera> initializeCameras(const tassel_tools::Parameters& params) {
    std::vector<tassel_core::Camera> result;
    for (auto const& [id, T_ci] : params.T_cam_imu_map) {
        (void)T_ci;
        if (params.cam_intrinsic_map.find(id) == params.cam_intrinsic_map.end() ||
            params.cam_distort_map.find(id) == params.cam_distort_map.end()) {
            continue;
        }
        cv::Mat k = params.cam_intrinsic_map.at(id);
        cv::Mat dist = params.cam_distort_map.at(id);
        result.emplace_back(
            std::make_unique<tassel_core::CameraRadTan>(k, dist, params.cols, params.rows));
    }
    return result;
}

void publishStereoImage(
    const std::shared_ptr<tassel_tools::Viewer>& viewer, const LatestDisplayImage& image) {
    if (image.image.empty()) return;
    viewer->publishCompressedImage("stereo/image", "camera", image.image, "jpeg");
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
    const size_t max_frames = (argc >= 4) ? static_cast<size_t>(std::stoul(argv[3])) : 0;
    const double replay_hz = (argc >= 5) ? std::stod(argv[4]) : 20.0;

    const fs::path resolved_sequence_dir = resolveSequenceDir(sequence_dir);
    if (!fs::exists(resolved_sequence_dir / "mav0" / "cam0" / "data.csv")) {
        std::cerr << "[EuRoC] sequence is not extracted: " << sequence_dir << "\n"
                  << "        expected mav0/cam0/data.csv directly under the sequence directory "
                     "or under a same-named child directory\n";
        return 0;
    }

    tassel_tools::Parameters params(config_path.string());
    auto frames = makeStereoFrames(resolved_sequence_dir);
    auto imu_measurements = loadImuCsv(resolved_sequence_dir / "mav0" / "imu0" / "data.csv");

    if (frames.empty() || imu_measurements.empty()) {
        std::cerr << "[EuRoC] empty stereo or IMU stream under " << resolved_sequence_dir << "\n";
        return 1;
    }

    auto cameras = initializeCameras(params);
    if (cameras.size() < 2) {
        std::cerr << "[EuRoC] need two cameras in config: " << config_path << "\n";
        return 1;
    }

    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("world");
    rclcpp::QoS image_qos(rclcpp::KeepLast(1));
    image_qos.best_effort().durability_volatile();
    viewer->createCompressedImagePublisher("stereo/image", image_qos);
    viewer->createOdometryPublisher("imu", "odom/camera");
    viewer->createPathPublisher("vo/path", rclcpp::QoS(10), params.viewer_path_max_poses);
    viewer->createPointCloudPublisher("landmarks");
    rclcpp::QoS telemetry_qos(rclcpp::KeepLast(1));
    telemetry_qos.reliable().durability_volatile();
    for (const char* topic :
         {"optimization/total_reduction", "optimization/visual_reduction",
          "optimization/imu_reduction", "optimization/prior_reduction", "optimization/valid_count",
          "optimization/invalid_count", "optimization/valid"}) {
        viewer->createScalarPublisher(topic, telemetry_qos);
    }
    viewer->createIntArrayPublisher("optimization/visual_factors_per_frame", telemetry_qos);
    viewer->createImagePublisher("optimization/visual_window", telemetry_qos);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(viewer);
    std::atomic_bool stop_executor{false};
    std::thread spin_thread([&]() {
        while (rclcpp::ok() && !stop_executor.load()) {
            executor.spin_some(std::chrono::milliseconds(5));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    const CameraBase* camera_ptr = cameras[0].get();
    FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    tracker.addCamera(
        std::move(cameras[0]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius, params.min_feature_num);
    tracker.addCamera(
        std::move(cameras[1]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius, params.min_feature_num);

    auto state = std::make_shared<State>(static_cast<int>(params.max_frame_count));
    auto feature_manager = std::make_shared<FeatureManager>(
        params.reproj_err_thres, params.parallax_thres, params.tracked_times_thres,
        params.min_tracked_pts, params.min_translation, params.min_depth, params.max_depth);

    Estimator estimator(params, state, feature_manager);
    state->camera = camera_ptr;
    estimator.setCamera(camera_ptr);
    estimator.setPoseCallback([&viewer](double ts, const Sophus::SE3d& pose) {
        viewer->publishOdometry(
            "odom/camera", pose.translation(), pose.unit_quaternion(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero());
        viewer->publishPath("vo/path", pose.translation(), pose.unit_quaternion());
        std::cout << "[pose] t=" << ts << " p=" << pose.translation().transpose() << "\n";
    });
    estimator.setCloudCallback([&viewer](double /*ts*/, const std::vector<Eigen::Vector3d>& pts) {
        viewer->publishPointCloud("landmarks", pts);
    });
    estimator.setOptimizationCallback([&viewer](double /*ts*/, const OptimizationStats& stats) {
        viewer->publishScalar(
            "optimization/total_reduction", stats.total_cost_before - stats.total_cost_after);
        viewer->publishScalar(
            "optimization/visual_reduction", stats.visual_cost_before - stats.visual_cost_after);
        viewer->publishScalar(
            "optimization/imu_reduction", stats.imu_cost_before - stats.imu_cost_after);
        viewer->publishScalar(
            "optimization/prior_reduction", stats.prior_cost_before - stats.prior_cost_after);
        viewer->publishScalar("optimization/valid_count", static_cast<double>(stats.valid_count));
        viewer->publishScalar(
            "optimization/invalid_count", static_cast<double>(stats.invalid_count));
        viewer->publishScalar("optimization/valid", stats.valid ? 1.0 : 0.0);
        viewer->publishIntArray(
            "optimization/visual_factors_per_frame", stats.visual_factors_per_frame);
        viewer->publishVisualFactorWindow(
            "optimization/visual_window", stats.visual_factors_per_frame);
    });

    const size_t frame_limit =
        (max_frames == 0) ? frames.size() : std::min(max_frames, frames.size());
    BlockingDatasetSync sync;

    std::atomic_bool stereo_done{false};
    std::atomic_bool imu_done{false};
    std::atomic_bool stop_reader{false};
    std::atomic_bool stop_image_publisher{false};
    std::atomic_size_t produced{0};
    std::atomic_size_t produced_imu{0};
    std::mutex latest_image_mutex;
    LatestDisplayImage latest_image;
    std::mutex loaded_stereo_mutex;
    std::condition_variable loaded_stereo_cv;
    std::deque<std::shared_ptr<tassel_utils::StereoObservation>> loaded_stereo_queue;
    bool stereo_load_done = false;
    constexpr size_t kMaxLoadedStereoFrames = 30;
    const size_t preload_target = std::min(frame_limit, kMaxLoadedStereoFrames);

    std::thread stereo_loader_thread([&]() {
        for (size_t i = 0; i < frame_limit && rclcpp::ok() && !stop_reader.load(); ++i) {
            const auto& frame = frames[i];
            cv::Mat left_img = cv::imread(frame.left_path.string(), cv::IMREAD_GRAYSCALE);
            cv::Mat right_img = cv::imread(frame.right_path.string(), cv::IMREAD_GRAYSCALE);
            if (left_img.empty() || right_img.empty()) {
                std::cerr << "[EuRoC] failed to read stereo image at t=" << frame.timestamp << "\n";
                break;
            }

            auto stereo_msg = std::make_shared<tassel_utils::StereoObservation>();
            stereo_msg->timestamp = frame.timestamp;
            stereo_msg->left_img = std::move(left_img);
            stereo_msg->right_img = std::move(right_img);

            {
                std::unique_lock<std::mutex> lock(loaded_stereo_mutex);
                loaded_stereo_cv.wait(lock, [&]() {
                    return stop_reader.load() ||
                           loaded_stereo_queue.size() < kMaxLoadedStereoFrames;
                });
                if (stop_reader.load()) break;
                loaded_stereo_queue.push_back(std::move(stereo_msg));
            }
            loaded_stereo_cv.notify_all();
        }

        {
            std::lock_guard<std::mutex> lock(loaded_stereo_mutex);
            stereo_load_done = true;
        }
        loaded_stereo_cv.notify_all();
    });

    {
        std::unique_lock<std::mutex> lock(loaded_stereo_mutex);
        loaded_stereo_cv.wait(lock, [&]() {
            return loaded_stereo_queue.size() >= preload_target || stereo_load_done ||
                   !rclcpp::ok();
        });
    }

    const double nominal_frame_dt =
        (frame_limit > 1) ? (frames[1].timestamp - frames[0].timestamp) : (1.0 / replay_hz);
    const double playback_end_ts = frames[frame_limit - 1].timestamp + nominal_frame_dt;
    const double playback_scale = (1.0 / replay_hz) / nominal_frame_dt;
    const double playback_start_ts = frames.front().timestamp;
    const auto playback_start_time = std::chrono::steady_clock::now();

    auto sleep_until_sensor_time = [&](double sensor_ts) {
        const double replay_elapsed =
            std::max(0.0, (sensor_ts - playback_start_ts) * playback_scale);
        const auto target_time =
            playback_start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(replay_elapsed));
        std::this_thread::sleep_until(target_time);
    };

    std::thread image_publish_thread([&]() {
        const auto period = std::chrono::duration<double>(1.0 / replay_hz);
        auto next_tick = std::chrono::steady_clock::now();
        while (rclcpp::ok() && !stop_image_publisher.load()) {
            LatestDisplayImage image;
            {
                std::lock_guard<std::mutex> lock(latest_image_mutex);
                image.image = latest_image.image.clone();
            }
            publishStereoImage(viewer, image);

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next_tick);
        }
    });

    std::thread imu_reader_thread([&]() {
        for (const auto& imu : imu_measurements) {
            if (!rclcpp::ok() || stop_reader.load()) break;
            if (imu.timestamp > playback_end_ts) break;
            if (imu.timestamp > playback_start_ts) {
                sleep_until_sensor_time(imu.timestamp);
            }
            if (!rclcpp::ok() || stop_reader.load()) break;
            sync.pushImu(imu);
            ++produced_imu;
        }
        imu_done = true;
        sync.closeImu();
    });

    std::thread stereo_reader_thread([&]() {
        while (rclcpp::ok() && !stop_reader.load()) {
            std::shared_ptr<tassel_utils::StereoObservation> stereo_msg;
            {
                std::unique_lock<std::mutex> lock(loaded_stereo_mutex);
                loaded_stereo_cv.wait(lock, [&]() {
                    return stop_reader.load() || !loaded_stereo_queue.empty() || stereo_load_done;
                });
                if (stop_reader.load()) break;
                if (loaded_stereo_queue.empty()) {
                    if (stereo_load_done) break;
                    continue;
                }
                stereo_msg = std::move(loaded_stereo_queue.front());
                loaded_stereo_queue.pop_front();
            }
            loaded_stereo_cv.notify_all();

            sleep_until_sensor_time(stereo_msg->timestamp);
            if (!rclcpp::ok() || stop_reader.load()) break;
            sync.pushStereo(stereo_msg);
            ++produced;
        }

        stereo_done = true;
        sync.closeStereo();
    });

    size_t processed = 0;
    while (rclcpp::ok()) {
        SyncedPacket packet;
        if (!sync.waitPop(packet, state->delay_time)) break;
        if (!packet.stereo) continue;

        std::unordered_map<int, FeaturePerFrame> feature_frame;
        {
            tassel_utils::Timer t("euroc_stereo_tracking");
            feature_frame =
                tracker.stereoTracking(0, packet.stereo->left_img, 1, packet.stereo->right_img);
        }
        for (auto& [id, feature] : feature_frame) {
            (void)id;
            feature.applied_delay = packet.applied_delay;
        }

        cv::Mat left_tracking = packet.stereo->left_img.clone();
        cv::Mat right_tracking = packet.stereo->right_img.clone();
        tracker.drawTrackingResult(0, left_tracking);
        tracker.drawTrackingResult(1, right_tracking);
        cv::Mat tracking_disp;
        cv::hconcat(left_tracking, right_tracking, tracking_disp);
        {
            std::lock_guard<std::mutex> lock(latest_image_mutex);
            latest_image.image = std::move(tracking_disp);
        }

        estimator.processMeasurement(
            packet.stereo->timestamp, feature_frame, packet.imu_slice, packet.applied_delay);

        ++processed;

        if (processed % 20 == 0) {
            std::cout << "[EuRoC] processed " << processed << "/" << frame_limit << " (read "
                      << produced.load() << ", imu read " << produced_imu.load() << ")"
                      << " stereo frames, features=" << feature_frame.size()
                      << ", imu=" << packet.imu_slice.size() << "\n";
        }
    }

    stop_reader = true;
    loaded_stereo_cv.notify_all();
    stop_image_publisher = true;
    if (stereo_loader_thread.joinable()) stereo_loader_thread.join();
    if (stereo_reader_thread.joinable()) stereo_reader_thread.join();
    if (imu_reader_thread.joinable()) imu_reader_thread.join();
    if (image_publish_thread.joinable()) image_publish_thread.join();

    stop_executor = true;
    executor.cancel();
    if (spin_thread.joinable()) spin_thread.join();

    rclcpp::shutdown();

    std::cout << "\n[EuRoC] done. processed=" << processed
              << ", keyframes in window=" << state->cur_frame_count << "\n";
    if (state->cur_frame_count > 0) {
        int idx = state->cur_frame_count - 1;
        std::cout << "Final pose:\n"
                  << Sophus::SE3d(state->Rs[idx], state->Ps[idx]).matrix() << "\n";
    }

    return 0;
}
