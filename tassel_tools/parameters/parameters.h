#ifndef TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
#define TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_

#include <Eigen/Core>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "parameters/params_parser.h"
#include "tassel_utils/types.h"

namespace tassel_tools {

enum class TrustRegionStrategy {
    LevenbergMarquardt,
    Dogleg,
};

inline TrustRegionStrategy parseTrustRegionStrategy(const std::string& strategy_raw) {
    std::string strategy = strategy_raw;
    const auto first = std::find_if_not(
        strategy.begin(), strategy.end(), [](unsigned char ch) { return std::isspace(ch); });
    const auto last = std::find_if_not(strategy.rbegin(), strategy.rend(), [](unsigned char ch) {
                          return std::isspace(ch);
                      }).base();
    if (first >= last) {
        throw std::runtime_error("trust_region_strategy must not be empty");
    }
    strategy = std::string(first, last);
    std::transform(strategy.begin(), strategy.end(), strategy.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (strategy == "levenberg_marquardt") {
        return TrustRegionStrategy::LevenbergMarquardt;
    }
    if (strategy == "dogleg") {
        return TrustRegionStrategy::Dogleg;
    }
    throw std::runtime_error(
        "Invalid trust_region_strategy: \"" + strategy_raw +
        "\". Supported values: levenberg_marquardt, dogleg");
}

struct Parameters {
    explicit Parameters(const std::string& config_file) {
        ParamsParser parser(config_file);
        loadCameras(parser);
        loadTracker(parser);
        loadFeatureManager(parser);
        loadEstimator(parser);
        loadImu(parser);
        loadInitialization(parser);
        loadViewer(parser);
        validate();
    }

    // 相机标定：用于 test_estimator 相机构造、特征管理器三角化、视觉因子、初始化和世界/IMU 对齐。
    std::string camera_model;
    cv::Mat cam_distort;
    cv::Mat cam_intrinsic;
    Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic = Eigen::Vector3d::Zero();

    // 图像和特征跟踪器配置：用于 FeatureTracker 和相机创建。
    int rows, cols;
    int per_grid_rows, per_grid_cols;
    std::string valid_mask_path;
    double mask_radius;
    int min_feature_num;
    bool flow_back;
    double max_square_move_dist;
    double min_gradient;
    bool enable_statistics;

    // 路标和关键帧管理：用于 FeatureManager。
    double reproj_err_thres;
    double reproj_huber_thres;
    double parallax_threshold;
    int min_landmark_observations;
    double min_depth;
    double max_depth;
    double keyframe_min_connection_ratio;

    // 滑窗优化：用于 Estimator::optimize、先验更新和 reset。
    int num_iterations;
    double max_solver_time = 0.0;
    // 活动图像状态数量；估计器会额外分配一个保留宿主槽。
    size_t max_frame_count;
    double visual_factor_weight;
    int num_threads = 1;
    TrustRegionStrategy trust_region_strategy = TrustRegionStrategy::LevenbergMarquardt;
    double delay_obs_gyro_threshold = 0.7;
    double delay_obs_speed_threshold = 0.2;
    int delay_obs_min_frames = 3;
    double imu_repropagate_ba_threshold = 0.02;
    double imu_repropagate_bg_threshold = 0.002;
    tassel_utils::IntegratorType integrator_type = tassel_utils::IntegratorType::MidPoint;

    // IMU 模型和标定：用于 Estimator 预测、预积分和初始化。
    double acc_n, acc_w;
    double gyr_n, gyr_w;
    double g_norm;
    Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();

    // 视觉惯性初始化和 SFM：用于 Estimator::tryInitialize。
    double gravity_diff_threshold = 0.17;
    double init_min_scale = 0.01;
    int sfm_min_correspondences = 10;
    int sfm_min_e_inliers = 8;
    double sfm_e_ransac_threshold = 0.004;
    double sfm_pnp_reproj_threshold = 0.03;
    double sfm_max_bad_pnp_ratio = 0.3;
    int sfm_ba_max_iterations = 30;
    int sfm_ba_num_threads = 5;

    // 可视化：用于 Viewer 发布器。
    size_t viewer_path_max_poses = 300;

private:
    void validate() const {
        if (!(min_depth > 0.0 && max_depth > min_depth)) {
            throw std::invalid_argument("Expected 0 < min_depth < max_depth");
        }
        if (max_frame_count < 3) {
            throw std::invalid_argument("max_frame_count must be at least 3");
        }
        if (num_iterations <= 0 || max_solver_time < 0.0 || num_threads <= 0 ||
            visual_factor_weight <= 0.0) {
            throw std::invalid_argument("Invalid optimization parameters");
        }
        if (delay_obs_gyro_threshold < 0.0 || delay_obs_speed_threshold < 0.0 ||
            delay_obs_min_frames <= 0) {
            throw std::invalid_argument("Invalid time-delay excitation parameters");
        }
        if (acc_n <= 0.0 || acc_w <= 0.0 || gyr_n <= 0.0 || gyr_w <= 0.0 || g_norm <= 0.0) {
            throw std::invalid_argument("IMU noise and gravity parameters must be positive");
        }
        if (min_landmark_observations < 2) {
            throw std::invalid_argument("min_landmark_observations must be at least 2");
        }
        if (keyframe_min_connection_ratio < 0.0 || keyframe_min_connection_ratio > 1.0) {
            throw std::invalid_argument("keyframe_min_connection_ratio must be in [0, 1]");
        }
        if (!std::isfinite(parallax_threshold) || parallax_threshold < 0.0) {
            throw std::invalid_argument("parallax_threshold must be finite and non-negative");
        }
        if (min_feature_num < 0) {
            throw std::invalid_argument("min_feature_num must be non-negative");
        }
        if (!std::isfinite(init_min_scale) || init_min_scale <= 0.0) {
            throw std::invalid_argument("init_min_scale must be finite and positive");
        }
        if (camera_model != "radtan" && camera_model != "equi") {
            throw std::invalid_argument("Unsupported camera_model: " + camera_model);
        }
    }

    void loadCameras(ParamsParser& parser) {
        camera_model = normalizeToken(parser.as<std::string>("cam0", "camera_model"));
        cam_intrinsic = parser.as<cv::Mat>("cam0", "intrinsics");
        cam_distort = parser.as<cv::Mat>("cam0", "distortion_coeffs");
        const Eigen::Matrix4d T_imu_cam = parser.as<Eigen::Matrix4d>("cam0", "T_cam_imu").inverse();
        ric = T_imu_cam.block<3, 3>(0, 0);
        tic = T_imu_cam.block<3, 1>(0, 3);
    }

    void loadTracker(ParamsParser& parser) {
        rows = parser.as<int>("rows");
        cols = parser.as<int>("cols");
        per_grid_rows = parser.as<int>("per_grid_rows");
        per_grid_cols = parser.as<int>("per_grid_cols");
        valid_mask_path = parser.as<std::string>("valid_mask_path");
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
        min_landmark_observations = parser.as<int>("min_landmark_observations");
        min_depth = parser.as<double>("min_depth");
        max_depth = parser.as<double>("max_depth");
        keyframe_min_connection_ratio = parser.as<double>("keyframe_min_connection_ratio");
        parallax_threshold = parser.as<double>("parallax_threshold");
    }

    void loadEstimator(ParamsParser& parser) {
        num_iterations = parser.as<int>("num_iterations");
        max_solver_time = parser.as<double>("max_solver_time");
        max_frame_count = parser.as<size_t>("max_frame_count");
        visual_factor_weight = parser.as<double>("visual_factor_weight");
        num_threads = parser.as<int>("num_threads");
        trust_region_strategy =
            parseTrustRegionStrategy(parser.as<std::string>("trust_region_strategy"));
        delay_obs_gyro_threshold = parser.as<double>("delay_obs_gyro_threshold");
        delay_obs_speed_threshold = parser.as<double>("delay_obs_speed_threshold");
        delay_obs_min_frames = parser.as<int>("delay_obs_min_frames");
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
        init_min_scale = parser.as<double>("init_min_scale");
        sfm_min_correspondences = parser.as<int>("sfm_min_correspondences");
        sfm_min_e_inliers = parser.as<int>("sfm_min_e_inliers");
        sfm_e_ransac_threshold = parser.as<double>("sfm_e_ransac_threshold");
        sfm_pnp_reproj_threshold = parser.as<double>("sfm_pnp_reproj_threshold");
        sfm_max_bad_pnp_ratio = parser.as<double>("sfm_max_bad_pnp_ratio");
        sfm_ba_max_iterations = parser.as<int>("sfm_ba_max_iterations");
        sfm_ba_num_threads = parser.as<int>("sfm_ba_num_threads");
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
            return tassel_utils::IntegratorType::MidPoint;
        }
        if (integrator_name == "euler") {
            return tassel_utils::IntegratorType::Euler;
        }
        throw std::runtime_error(
            "Invalid integrator_type: \"" + integrator_name_raw +
            "\". Supported values: midpoint, euler");
    }
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
