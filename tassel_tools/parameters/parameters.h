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
        pnp_reprojection_error_thres = parser.as<double>("pnp_reprojection_error_thres");
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
        num_iterations = parser.as<int>("num_iterations");
        max_frame_count = parser.as<size_t>("max_frame_count");

        initial_exposure_time_us = parser.as<int>("initial_exposure_time_us");

        visual_factor_weight = parser.as<double>("visual_factor_weight");

        acc_n = parser.as<double>("acc_n");
        acc_w = parser.as<double>("acc_w");
        gyr_n = parser.as<double>("gyr_n");
        gyr_w = parser.as<double>("gyr_w");
        g_norm = parser.as<double>("g_norm");

        estimate_ba_init = parser.as<bool>("estimate_ba_init");
        init_ba = parser.as<Eigen::Vector3d>("init_ba");
        min_rot_excitation = parser.as<double>("min_rot_excitation");
        min_excited_frames = parser.as<int>("min_excited_frames");
        num_init_iterations = parser.as<int>("num_init_iterations");

        try {
            acc_correction_matrix = parser.as<Eigen::Matrix3d>("acc_correction_matrix");
        } catch (const std::runtime_error&) {
            acc_correction_matrix = Eigen::Matrix3d::Identity();
        }
        try {
            acc_bias = parser.as<Eigen::Vector3d>("acc_bias");
        } catch (const std::runtime_error&) {
            acc_bias = Eigen::Vector3d::Zero();
        }
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
    double pnp_reprojection_error_thres;  // normalized-coordinate threshold for PnP
    double parallax_thres;
    int min_tracked_pts_num;
    int min_pnp_num;
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

    Eigen::Matrix3d acc_correction_matrix = Eigen::Matrix3d::Identity();
    Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();

    bool estimate_ba_init;
    Eigen::Vector3d init_ba;
    double min_rot_excitation;
    int min_excited_frames;
    int num_init_iterations;
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMETERS_H_
