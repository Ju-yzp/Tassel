#ifndef TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
#define TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_

#include <Eigen/Core>
#include <cstddef>
#include <map>
#include <opencv2/core.hpp>

#include "parameters/params_parser.h"

namespace tassel_tools {

struct Parameters {
    explicit Parameters(const std::string& config_file) {
        ParamsParser parser(config_file);
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

        reproj_err_thres = parser.as<double>("reproj_err_thres");
        pnp_reproj_err_thres = parser.as<double>("pnp_reproj_err_thres");
        parallax_thres = parser.as<double>("parallax_thres");
        min_tracked_pts = parser.as<int>("min_tracked_pts");
        min_pnp_pts = parser.as<int>("min_pnp_pts");
        tracked_times_thres = parser.as<int>("tracked_times_thres");
        min_pnp_inliers_ratio = parser.as<double>("min_pnp_inliers_ratio");
        min_translation = parser.as<double>("min_translation");
        min_depth = parser.as<double>("min_depth");
        max_depth = parser.as<double>("max_depth");

        cam_intrinsic_map[0] = parser.as<cv::Mat>("cam0", "intrinsics");
        cam_distort_map[0] = parser.as<cv::Mat>("cam0", "distortion_coeffs");
        T_cam_imu_map[0] = parser.as<Eigen::Matrix4d>("cam0", "T_cam_imu").inverse();
        cam_intrinsic_map[1] = parser.as<cv::Mat>("cam1", "intrinsics");
        cam_distort_map[1] = parser.as<cv::Mat>("cam1", "distortion_coeffs");
        T_cam_imu_map[1] = parser.as<Eigen::Matrix4d>("cam1", "T_cam_imu").inverse();

        ric = T_cam_imu_map[0].block<3, 3>(0, 0);
        tic = T_cam_imu_map[0].block<3, 1>(0, 3);
        ric1 = T_cam_imu_map[1].block<3, 3>(0, 0);
        tic1 = T_cam_imu_map[1].block<3, 1>(0, 3);

        use_imu = parser.as<bool>("use_imu");
        num_iterations = parser.as<int>("num_iterations");
        max_frame_count = parser.as<size_t>("max_frame_count");

        initial_exposure_time_us = parser.as<int>("initial_exposure_time_us");

        visual_factor_weight = parser.as<double>("visual_factor_weight");

        acc_n = parser.as<double>("acc_n");
        acc_w = parser.as<double>("acc_w");
        gyr_n = parser.as<double>("gyr_n");
        gyr_w = parser.as<double>("gyr_w");
        g_norm = parser.as<double>("g_norm");

        num_init_iterations = parser.as<int>("num_init_iterations");

        acc_bias = parser.as<Eigen::Vector3d>("acc_bias");
        num_threads = parser.as<int>("num_threads");
        init_time_span = parser.as<double>("init_time_span");
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

    std::map<size_t, Eigen::Matrix4d> T_cam_imu_map;
    std::map<size_t, cv::Mat> cam_distort_map;
    std::map<size_t, cv::Mat> cam_intrinsic_map;

    int rows, cols;
    int per_grid_rows, per_grid_cols;
    int edge_x, edge_y;
    double mask_radius;
    int min_feature_num;

    bool flow_back;
    double max_square_move_dist;
    double min_gradient;
    bool enable_statistics;

    double reproj_err_thres;
    double pnp_reproj_err_thres;
    double parallax_thres;
    int min_tracked_pts;
    int min_pnp_pts;
    double min_pnp_inliers_ratio;
    double min_translation;
    double min_depth;
    double max_depth;

    bool use_imu;
    int num_iterations;
    int tracked_times_thres;
    size_t max_frame_count;

    int initial_exposure_time_us;

    double visual_factor_weight;

    double acc_n, acc_w;
    double gyr_n, gyr_w;
    double g_norm;

    Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();

    int num_init_iterations;

    // ── 外参 ──────────────────────────────────────────────────────────
    Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic = Eigen::Vector3d::Zero();
    Eigen::Matrix3d ric1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic1 = Eigen::Vector3d::Zero();

    // ── 优化参数 ──────────────────────────────────────────────────────
    int num_threads = 1;
    double init_time_span = 5.0;
    double gravity_diff_threshold = 0.17;

    // ── SFM 阈值 ──────────────────────────────────────────────────────
    int sfm_min_seed_pts = 10;
    int sfm_min_e_inliers = 8;
    double sfm_e_ransac_threshold = 0.004;
    int sfm_min_pnp_pts = 10;
    double sfm_pnp_reproj_threshold = 0.03;
    double sfm_max_bad_pnp_ratio = 0.3;
    int sfm_ba_max_iterations = 30;
    int sfm_ba_num_threads = 5;
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
