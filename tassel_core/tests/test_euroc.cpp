// =============================================================================
// test_euroc.cpp — EuRoC MAV offline stereo-inertial estimator
// =============================================================================
//
// Reads raw EuRoC data (PNG images + CSV) and runs the estimator offline.
// Outputs trajectory in TUM format for evaluation with evo/rpg_trajectory_evaluation.
//
// Usage:
//   ./test_euroc <euroc_path> [config_yaml] [output_traj.txt]
//
// Example:
//   ./test_euroc /data/EuRoC/MH_01_easy config/euroc_stereo_vins.yaml traj.txt
//
// Output:
//   - Estimated trajectory in TUM format (timestamp tx ty tz qx qy qz qw)
//   - If ground truth available, also saved as traj_gt.txt
// =============================================================================

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cam/camera_base.h"
#include "cam/camera_factory.h"
#include "estimator/estimator.h"
#include "estimator/estimator_option.h"
#include "frond_end/feature_manager.h"
#include "frond_end/feature_tracker.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/timer.h"
#include "viewer/viewer.h"

#include <buffer.h>
#include <synchronizer.h>
#include <rclcpp/rclcpp.hpp>

namespace {

struct EuRoCImu {
    double ts;
    Eigen::Vector3d gyro;
    Eigen::Vector3d acc;
};

struct EuRoStereo {
    double ts;
    std::string left_path;
    std::string right_path;
};

struct EuRoCGt {
    double ts;
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
};

// Parse EuRoC CSV. Lines start with # or a text header. Data: comma-separated.
std::vector<std::vector<double>> parseEuRoCCsv(const std::string& path, size_t expected_cols) {
    std::vector<std::vector<double>> rows;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << "\n";
        return rows;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        std::vector<double> vals;
        double v;
        while (ss >> v) vals.push_back(v);
        if (vals.size() >= expected_cols) rows.push_back(vals);
    }
    return rows;
}

// Parse image CSV: first column is timestamp_ns, second is filename string.
// Uses string parsing for filename to avoid double precision loss (19-digit timestamps).
std::vector<std::pair<double, std::string>> parseImageCsv(const std::string& path) {
    std::vector<std::pair<double, std::string>> rows;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << "\n";
        return rows;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        std::string ts_str = line.substr(0, comma);
        std::string fname = line.substr(comma + 1);
        // trim whitespace
        fname.erase(0, fname.find_first_not_of(" \t\r\n"));
        fname.erase(fname.find_last_not_of(" \t\r\n") + 1);
        double ts = std::stod(ts_str);
        if (ts < 1e9) continue;  // skip header row where timestamp_ns fails to parse
        rows.emplace_back(ts, fname);
    }
    return rows;
}

tassel_utils::IMUMeasurement imuInterpolate(
    const tassel_utils::IMUMeasurement& p0, const tassel_utils::IMUMeasurement& p1,
    double t_target) {
    tassel_utils::IMUMeasurement out;
    out.timestamp = t_target;
    double alpha = (t_target - p0.timestamp) / (p1.timestamp - p0.timestamp);
    out.acc = p0.acc + alpha * (p1.acc - p0.acc);
    out.gyro = p0.gyro + alpha * (p1.gyro - p0.gyro);
    return out;
}

using StereoBuf =
    ts::Buffer<tassel_utils::StereoObservation, ts::SensorType::Nearest, ts::SharedPtrStorage>;
using IMUBuf = ts::Buffer<tassel_utils::IMUMeasurement, ts::SensorType::Slice, ts::ValueStorage>;
using SyncType = ts::Synchronizer<StereoBuf, IMUBuf>;

std::vector<tassel_core::Camera> initializeCameras(const tassel_tools::Parameters& params) {
    std::vector<tassel_core::Camera> result;
    for (auto const& [id, T_imu_cam] : params.T_cam_imu_map) {
        auto it_k = params.cam_intrinsic_map.find(id);
        auto it_d = params.cam_distort_map.find(id);
        if (it_k == params.cam_intrinsic_map.end() || it_d == params.cam_distort_map.end())
            continue;
        std::string model = (id == 0) ? "radtan" : "radtan";
        result.push_back(tassel_core::CameraFactory::create(
            model, it_k->second, it_d->second, params.cols, params.rows));
    }
    return result;
}

void saveTrajectory(
    const std::string& path, const std::vector<std::pair<double, Sophus::SE3d>>& poses) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot write " << path << "\n";
        return;
    }
    f << std::fixed << std::setprecision(6);
    for (const auto& [ts, pose] : poses) {
        Eigen::Vector3d p = pose.translation();
        Eigen::Quaterniond q = pose.unit_quaternion();
        f << ts << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y()
          << " " << q.z() << " " << q.w() << "\n";
    }
    std::cout << "Saved trajectory (" << poses.size() << " poses) to " << path << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tassel_core;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <euroc_path> [config_yaml] [output_traj.txt]\n"
                  << "Example: " << argv[0]
                  << " /data/EuRoC/MH_01_easy config/euroc_stereo_vins.yaml traj.txt\n";
        return 1;
    }

    std::string euroc_path = argv[1];
    while (!euroc_path.empty() && euroc_path.back() == '/') euroc_path.pop_back();
    std::string mav_path = euroc_path + "/mav0";

    std::string config_path =
        (argc >= 3) ? argv[2] : "/home/adrewn/Tassel/config/euroc_stereo_vins.yaml";
    std::string output_path = (argc >= 4) ? argv[3] : "euroc_traj.txt";

    tassel_tools::Parameters params(config_path);

    // ── Load EuRoC data ────────────────────────────────────────────────
    std::cout << "[EuRoC] Loading data from " << euroc_path << "...\n";

    // IMU data: ts, wx, wy, wz, ax, ay, az
    auto imu_csv = parseEuRoCCsv(mav_path + "/imu0/data.csv", 7);
    std::vector<EuRoCImu> imu_data;
    imu_data.reserve(imu_csv.size());
    for (const auto& row : imu_csv) {
        EuRoCImu m;
        m.ts = row[0] / 1e9;
        m.gyro = Eigen::Vector3d(row[1], row[2], row[3]);
        m.acc = Eigen::Vector3d(row[4], row[5], row[6]);
        imu_data.push_back(m);
    }
    std::cout << "  IMU: " << imu_data.size() << " measurements\n";

    // Stereo images: ts, filename (use string-based parsing to preserve precision)
    auto cam0_data = parseImageCsv(mav_path + "/cam0/data.csv");
    auto cam1_data = parseImageCsv(mav_path + "/cam1/data.csv");
    size_t n_stereo = std::min(cam0_data.size(), cam1_data.size());
    std::vector<EuRoStereo> stereo_data;
    stereo_data.reserve(n_stereo);
    for (size_t i = 0; i < n_stereo; ++i) {
        EuRoStereo s;
        s.ts = cam0_data[i].first / 1e9;
        s.left_path = mav_path + "/cam0/data/" + cam0_data[i].second;
        s.right_path = mav_path + "/cam1/data/" + cam1_data[i].second;
        stereo_data.push_back(s);
    }
    std::cout << "  Stereo: " << stereo_data.size() << " frames\n";

    // Ground truth (optional): ts, px, py, pz [, qw, qx, qy, qz]
    // Standard EuRoC: state_groundtruth_estimate0/data.csv (8 cols)
    // Some mirrors:   gt/data.csv (4 cols, position only)
    std::string gt_csv_path = mav_path + "/state_groundtruth_estimate0/data.csv";
    auto gt_csv = parseEuRoCCsv(gt_csv_path, 4);
    if (gt_csv.empty()) {
        gt_csv_path = mav_path + "/gt/data.csv";
        gt_csv = parseEuRoCCsv(gt_csv_path, 4);
    }
    std::vector<EuRoCGt> gt_data;
    if (!gt_csv.empty()) {
        bool has_orientation = (gt_csv[0].size() >= 8);
        gt_data.reserve(gt_csv.size());
        for (const auto& row : gt_csv) {
            EuRoCGt g;
            g.ts = row[0] / 1e9;
            g.p = Eigen::Vector3d(row[1], row[2], row[3]);
            if (has_orientation) {
                g.q = Eigen::Quaterniond(row[4], row[5], row[6], row[7]);
            } else {
                g.q = Eigen::Quaterniond::Identity();
            }
            gt_data.push_back(g);
        }
        std::cout << "  Ground truth: " << gt_data.size()
                  << (has_orientation ? " poses\n" : " positions (no orientation)\n");
    } else {
        std::cout << "  Ground truth: not found\n";
    }

    if (stereo_data.empty()) {
        std::cerr << "ERROR: No stereo data found.\n";
        return 1;
    }

    // ── Buffers & Synchronizer ─────────────────────────────────────────
    auto stereo_buffer = StereoBuf::createShared(15);
    auto imu_buffer = IMUBuf::createShared(600);
    imu_buffer->set_interpolator(imuInterpolate);
    SyncType sync(0, stereo_buffer.get(), imu_buffer.get());

    // ── Cameras ────────────────────────────────────────────────────────
    auto cameras = initializeCameras(params);
    if (cameras.size() < 2) {
        std::cerr << "ERROR: Need 2 cameras, got " << cameras.size() << "\n";
        return 1;
    }
    const CameraBase* camera_ptr = cameras[0].get();

    Eigen::Matrix4d T_imu_cam0 = params.T_cam_imu_map.at(0);
    Eigen::Matrix4d T_imu_cam1 = params.T_cam_imu_map.at(1);
    Eigen::Matrix3d ric = T_imu_cam0.block<3, 3>(0, 0);
    Eigen::Vector3d tic = T_imu_cam0.block<3, 1>(0, 3);
    Eigen::Matrix3d ric1 = T_imu_cam1.block<3, 3>(0, 0);
    Eigen::Vector3d tic1 = T_imu_cam1.block<3, 1>(0, 3);

    // ── Frontend ───────────────────────────────────────────────────────
    FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    tracker.addCamera(
        std::move(cameras[0]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius);
    tracker.addCamera(
        std::move(cameras[1]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius);

    // ── Estimator ──────────────────────────────────────────────────────
    EstimatorOption option;
    option.num_iterations = params.num_iterations;
    option.min_depth = params.min_depth;
    option.max_depth = params.max_depth;
    option.acc_n = params.acc_n;
    option.acc_w = params.acc_w;
    option.gyr_n = params.gyr_n;
    option.gyr_w = params.gyr_w;
    option.g_norm = params.g_norm;
    option.estimate_ba_init = params.estimate_ba_init;
    option.init_ba = params.init_ba;
    option.min_rot_excitation = params.min_rot_excitation;
    option.min_excited_frames = params.min_excited_frames;
    option.num_init_iterations = params.num_init_iterations;
    option.acc_correction_matrix = params.acc_correction_matrix;
    option.acc_bias = params.acc_bias;

    auto state = std::make_shared<State>(static_cast<int>(params.max_frame_count));
    state->visual_sqrt_info = Eigen::Matrix2d::Identity() * params.visual_factor_weight;

    auto feature_manager = std::make_shared<FeatureManager>(
        params.reprojection_error_thres, params.pnp_reprojection_error_thres, params.parallax_thres,
        params.tracked_times_thres, params.min_tracked_pts_num, params.min_pnp_num,
        params.min_pnp_inliers_ratio, params.min_translation, params.min_depth, params.max_depth);

    Estimator estimator(option, state, feature_manager, ric, tic, ric1, tic1);
    estimator.setCamera(camera_ptr);

    // ── Viewer ──────────────────────────────────────────────────────────
    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("world");
    viewer->createImagePublisher("stereo/image");
    viewer->createOdometryPublisher("camera", "odom/camera");
    viewer->createPathPublisher("vo/path");
    viewer->createPointCloudPublisher("landmarks");

    std::vector<std::pair<double, Sophus::SE3d>> trajectory;
    estimator.setPoseCallback([&trajectory, &viewer](double ts, const Sophus::SE3d& pose) {
        trajectory.emplace_back(ts, pose);
        viewer->publishOdometry("odom/camera", pose.translation(), pose.unit_quaternion());
        viewer->publishPath("vo/path", pose.translation(), pose.unit_quaternion());
    });
    estimator.setCloudCallback([&viewer](double /*ts*/, const std::vector<Eigen::Vector3d>& pts) {
        viewer->publishPointCloud("landmarks", pts);
    });

    // ── Main processing loop ───────────────────────────────────────────
    std::cout << "[EuRoC] Processing " << stereo_data.size() << " frames...\n";

    size_t imu_idx = 0;

    for (size_t i = 0; i < stereo_data.size(); ++i) {
        const auto& s = stereo_data[i];

        // Push IMU data up to this stereo frame (plus margin for synchronizer window)
        while (imu_idx < imu_data.size() && imu_data[imu_idx].ts <= s.ts + 0.1) {
            tassel_utils::IMUMeasurement m;
            m.timestamp = imu_data[imu_idx].ts;
            m.acc = imu_data[imu_idx].acc;
            m.gyro = imu_data[imu_idx].gyro;
            imu_buffer->push_back(m);
            ++imu_idx;
        }

        // Load images
        cv::Mat left_img = cv::imread(s.left_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(s.right_path, cv::IMREAD_GRAYSCALE);
        if (left_img.empty() || right_img.empty()) {
            std::cerr << "WARN: Skipping frame " << i << " — cannot read images\n";
            continue;
        }

        auto stereo_msg = std::make_shared<tassel_utils::StereoObservation>();
        stereo_msg->timestamp = s.ts;
        stereo_msg->left_img = left_img;
        stereo_msg->right_img = right_img;
        stereo_buffer->push_back(stereo_msg);

        SyncType::DataPackage package;
        if (!sync.pop_package(package)) continue;

        auto& stereo_ptr = package.get<0>();
        auto& imu_vec = package.get<1>();
        double ts = stereo_ptr->timestamp;

        std::unordered_map<int, FeaturePerFrame> feature_frame;
        {
            tassel_utils::Timer t("tracking");
            feature_frame =
                tracker.stereoTracking(0, stereo_ptr->left_img, 1, stereo_ptr->right_img);
        }

        estimator.processMeasurement(ts, feature_frame, imu_vec);

        // Visualization: feature overlay + publish
        cv::Mat disp_left, disp_right;
        cv::cvtColor(stereo_ptr->left_img, disp_left, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(0, disp_left);
        cv::cvtColor(stereo_ptr->right_img, disp_right, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(1, disp_right);

        // Pose text overlay
        if (state->cur_frame_count > 0) {
            int k = state->cur_frame_count - 1;
            Eigen::Vector3d P = state->Ps[k];
            Eigen::Vector3d V = state->Vs[k];
            char buf[256];
            std::snprintf(
                buf, sizeof(buf), "P: %.3f %.3f %.3f | V: %.3f %.3f %.3f | kf: %d", P.x(), P.y(),
                P.z(), V.x(), V.y(), V.z(), state->cur_frame_count);
            cv::putText(
                disp_left, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
        }

        cv::Mat stereo_disp;
        cv::hconcat(disp_left, disp_right, stereo_disp);
        viewer->publishImage("stereo/image", "camera", stereo_disp);

        rclcpp::spin_some(viewer);

        if (i % 100 == 0 || i == stereo_data.size() - 1) {
            double pct = (i + 1) * 100.0 / stereo_data.size();
            std::printf(
                "\r  Progress: %.0f%%  (%zu frames, %d keyframes)", pct, i + 1,
                state->cur_frame_count);
            std::fflush(stdout);
        }
    }
    std::cout << "\n";

    rclcpp::shutdown();

    // ── Final pose ─────────────────────────────────────────────────────
    std::cout << "\n[EuRoC] Done. " << state->cur_frame_count << " keyframes in window.\n";
    if (state->cur_frame_count > 0) {
        int idx = state->cur_frame_count - 1;
        std::cout << "Final pose:\n"
                  << Sophus::SE3d(state->Rs[idx], state->Ps[idx]).matrix() << "\n";
    }

    // ── Save trajectory ────────────────────────────────────────────────
    saveTrajectory(output_path, trajectory);

    if (!gt_data.empty()) {
        std::string gt_path = output_path;
        size_t dot = gt_path.rfind('.');
        if (dot != std::string::npos)
            gt_path.insert(dot, "_gt");
        else
            gt_path += "_gt";
        std::ofstream f(gt_path);
        f << std::fixed << std::setprecision(6);
        for (const auto& g : gt_data) {
            f << g.ts << " " << g.p.x() << " " << g.p.y() << " " << g.p.z() << " " << g.q.x() << " "
              << g.q.y() << " " << g.q.z() << " " << g.q.w() << "\n";
        }
        std::cout << "Saved ground truth (" << gt_data.size() << " poses) to " << gt_path << "\n";
    }

    return 0;
}
