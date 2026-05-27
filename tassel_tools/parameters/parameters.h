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

        reprojection_error_thres = parser.as<double>("reprojection_error_thres");
        parallax_thres = parser.as<double>("parallax_thres");
        min_tracked_pts_num = parser.as<int>("min_tracked_pts_num");
        min_pnp_num = parser.as<int>("min_pnp_num");
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

        use_imu = parser.as<bool>("use_imu");
        visual_sqrt_info = parser.as<double>("visual_sqrt_info");
        num_iterations = parser.as<int>("num_iterations");
        solve_time = parser.as<double>("solve_time");
        max_frame_count = parser.as<size_t>("max_frame_count");
        spand_time = parser.as<double>("spand_time");
        repropagate_ba_thres = parser.as<double>("repropagate_ba_thres");
        repropagate_bg_thres = parser.as<double>("repropagate_bg_thres");
        num_threads = parser.as<int>("num_threads");

        optimize_enabled = parser.as<bool>("optimize_enabled");
        marginalization_enabled = parser.as<bool>("marginalization_enabled");
        lambda_initial = parser.as<double>("lambda_initial");

        acc_n = parser.as<double>("acc_n");
        acc_w = parser.as<double>("acc_w");
        gyr_n = parser.as<double>("gyr_n");
        gyr_w = parser.as<double>("gyr_w");
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

    double reprojection_error_thres;
    double parallax_thres;
    int min_tracked_pts_num;
    int min_pnp_num;
    double min_pnp_inliers_ratio;
    double min_translation;
    double min_depth;
    double max_depth;

    bool use_imu;
    double visual_sqrt_info;
    int num_iterations;
    double solve_time;
    int tracked_times_thres;
    size_t max_frame_count;
    double spand_time;
    double repropagate_ba_thres;
    double repropagate_bg_thres;
    int num_threads;

    bool optimize_enabled;
    bool marginalization_enabled;
    double lambda_initial;

    double acc_n, acc_w;
    double gyr_n, gyr_w;
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
