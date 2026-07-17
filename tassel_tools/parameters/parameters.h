#ifndef TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
#define TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_

#include <Eigen/Core>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <map>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "parameters/params_parser.h"
#include "tassel_utils/types.h"

namespace tassel_tools {

struct Parameters {
    explicit Parameters(const std::string& config_file) {
        ParamsParser parser(config_file);
        loadCameras(parser);
        loadTracker(parser);
        loadFeatureManager(parser);
        loadEstimator(parser);
        loadImu(parser);
        loadInitialization(parser);
        loadHardware(parser);
        loadViewer(parser);
        validate();
    }

    // Camera calibration: consumed by test_estimator camera construction, FeatureManager
    // triangulation, visual factors, initialization, and world/IMU alignment.
    std::map<size_t, Eigen::Matrix4d> T_cam_imu_map;
    std::map<size_t, cv::Mat> cam_distort_map;
    std::map<size_t, cv::Mat> cam_intrinsic_map;
    Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic = Eigen::Vector3d::Zero();

    // Image and feature-tracker settings: consumed by FeatureTracker and camera creation.
    int rows, cols;
    int per_grid_rows, per_grid_cols;
    int edge_x, edge_y;
    double mask_radius;
    int min_feature_num;
    bool flow_back;
    double max_square_move_dist;
    double min_gradient;
    bool enable_statistics;

    // Landmark and keyframe management: consumed by FeatureManager.
    double reproj_err_thres;
    double reproj_huber_thres;
    double parallax_thres;
    int min_tracked_pts;
    int tracked_times_thres;
    double min_translation;
    double min_depth;
    double max_depth;
    double keyframe_new_feature_ratio;

    // Sliding-window optimization: consumed by Estimator::optimize/buildPrior/reset.
    int num_iterations;
    size_t max_frame_count;
    double visual_factor_weight;
    int num_threads = 1;
    double dt_gyro_threshold = 0.7;
    double imu_repropagate_ba_threshold = 0.02;
    double imu_repropagate_bg_threshold = 0.002;
    tassel_utils::IntegratorType integrator_type = tassel_utils::IntegratorType::kMidPoint;

    // IMU model and calibration: consumed by Estimator propagation, preintegration, and init.
    double acc_n, acc_w;
    double gyr_n, gyr_w;
    double g_norm;
    Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();

    // Visual-inertial initialization and SFM: consumed by Estimator::tryInitialize.
    double gravity_diff_threshold = 0.17;
    int sfm_min_seed_pts = 10;
    int sfm_min_e_inliers = 8;
    double sfm_e_ransac_threshold = 0.004;
    int sfm_min_pnp_pts = 10;
    double sfm_pnp_reproj_threshold = 0.03;
    double sfm_max_bad_pnp_ratio = 0.3;
    int sfm_ba_max_iterations = 30;
    int sfm_ba_num_threads = 5;

    // Hardware capture: consumed by OAK/DepthAI integration tests.
    int initial_exposure_time_us;

    // Visualization: consumed by Viewer publishers.
    size_t viewer_path_max_poses = 300;

private:
    void validate() const {
        if (!(min_depth > 0.0 && max_depth > min_depth)) {
            throw std::invalid_argument("Expected 0 < min_depth < max_depth");
        }
        if (max_frame_count < 3) {
            throw std::invalid_argument("max_frame_count must be at least 3");
        }
        if (num_iterations <= 0 || num_threads <= 0 || visual_factor_weight <= 0.0) {
            throw std::invalid_argument("Invalid optimization parameters");
        }
        if (acc_n <= 0.0 || acc_w <= 0.0 || gyr_n <= 0.0 || gyr_w <= 0.0 || g_norm <= 0.0) {
            throw std::invalid_argument("IMU noise and gravity parameters must be positive");
        }
        if (keyframe_new_feature_ratio < 0.0 || keyframe_new_feature_ratio > 1.0) {
            throw std::invalid_argument("keyframe_new_feature_ratio must be in [0, 1]");
        }
    }

    static void loadCamera(ParamsParser& parser, size_t id, Parameters& params) {
        const std::string cam_key = "cam" + std::to_string(id);
        params.cam_intrinsic_map[id] = parser.as<cv::Mat>(cam_key, "intrinsics");
        params.cam_distort_map[id] = parser.as<cv::Mat>(cam_key, "distortion_coeffs");
        params.T_cam_imu_map[id] = parser.as<Eigen::Matrix4d>(cam_key, "T_cam_imu").inverse();
    }

    void loadCameras(ParamsParser& parser) {
        loadCamera(parser, 0, *this);
        loadCamera(parser, 1, *this);
        ric = T_cam_imu_map[0].block<3, 3>(0, 0);
        tic = T_cam_imu_map[0].block<3, 1>(0, 3);
    }

    void loadTracker(ParamsParser& parser) {
        rows = parser.as<int>("rows");
        cols = parser.as<int>("cols");
        per_grid_rows = parser.as<int>("per_grid_rows");
        per_grid_cols = parser.as<int>("per_grid_cols");
        edge_x = parser.as<int>("edge_x");
        edge_y = parser.as<int>("edge_y");
        mask_radius = parser.as<double>("mask_radius");
        min_feature_num = parser.as<int>("min_feature_num");
        flow_back = parser.as<bool>("flow_back");
        max_square_move_dist = parser.as<double>("max_square_move_dist");
        min_gradient = parser.as<double>("min_gradient");
        enable_statistics = parser.as<bool>("enable_statistics");
    }

    void loadFeatureManager(ParamsParser& parser) {
        reproj_err_thres = parser.as<double>("reproj_err_thres");
        reproj_huber_thres = parser.as<double>("reproj_huber_thres");
        parallax_thres = parser.as<double>("parallax_thres");
        min_tracked_pts = parser.as<int>("min_tracked_pts");
        tracked_times_thres = parser.as<int>("tracked_times_thres");
        min_translation = parser.as<double>("min_translation");
        min_depth = parser.as<double>("min_depth");
        max_depth = parser.as<double>("max_depth");
        keyframe_new_feature_ratio = parser.as<double>("keyframe_new_feature_ratio");
    }

    void loadEstimator(ParamsParser& parser) {
        num_iterations = parser.as<int>("num_iterations");
        max_frame_count = parser.as<size_t>("max_frame_count");
        visual_factor_weight = parser.as<double>("visual_factor_weight");
        num_threads = parser.as<int>("num_threads");
        dt_gyro_threshold = parser.as<double>("dt_gyro_threshold");
        imu_repropagate_ba_threshold = parser.as<double>("imu_repropagate_ba_threshold");
        imu_repropagate_bg_threshold = parser.as<double>("imu_repropagate_bg_threshold");
        integrator_type = parseIntegratorType(parser.as<std::string>("integrator_type"));
    }

    void loadImu(ParamsParser& parser) {
        acc_n = parser.as<double>("acc_n");
        acc_w = parser.as<double>("acc_w");
        gyr_n = parser.as<double>("gyr_n");
        gyr_w = parser.as<double>("gyr_w");
        g_norm = parser.as<double>("g_norm");
        acc_bias = parser.as<Eigen::Vector3d>("acc_bias");
    }

    void loadInitialization(ParamsParser& parser) {
        gravity_diff_threshold = parser.as<double>("gravity_diff_threshold");
        sfm_min_seed_pts = parser.as<int>("sfm_min_seed_pts");
        sfm_min_e_inliers = parser.as<int>("sfm_min_e_inliers");
        sfm_e_ransac_threshold = parser.as<double>("sfm_e_ransac_threshold");
        sfm_min_pnp_pts = parser.as<int>("sfm_min_pnp_pts");
        sfm_pnp_reproj_threshold = parser.as<double>("sfm_pnp_reproj_threshold");
        sfm_max_bad_pnp_ratio = parser.as<double>("sfm_max_bad_pnp_ratio");
        sfm_ba_max_iterations = parser.as<int>("sfm_ba_max_iterations");
        sfm_ba_num_threads = parser.as<int>("sfm_ba_num_threads");
    }

    void loadHardware(ParamsParser& parser) {
        initial_exposure_time_us = parser.as<int>("initial_exposure_time_us");
    }

    void loadViewer(ParamsParser& parser) {
        viewer_path_max_poses = parser.as<size_t>("viewer", "path_max_poses");
    }

    static std::string normalizeToken(std::string value) {
        const auto first = std::find_if_not(
            value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch); });
        const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
                              return std::isspace(ch);
                          }).base();
        if (first >= last) {
            return "";
        }
        value = std::string(first, last);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    static tassel_utils::IntegratorType parseIntegratorType(
        const std::string& integrator_name_raw) {
        const std::string integrator_name = normalizeToken(integrator_name_raw);
        if (integrator_name == "midpoint") {
            return tassel_utils::IntegratorType::kMidPoint;
        }
        if (integrator_name == "euler") {
            return tassel_utils::IntegratorType::kEuler;
        }
        throw std::runtime_error(
            "Invalid integrator_type: \"" + integrator_name_raw +
            "\". Supported values: midpoint, euler");
    }
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
