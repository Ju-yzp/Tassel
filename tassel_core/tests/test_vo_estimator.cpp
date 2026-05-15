#include <depthai/depthai.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>

#include "cam/camera_factory.h"
#include "estimator/estimator_option.h"
#include "estimator/vo_estimator.h"
#include "frond_end/feature_manager.h"
#include "frond_end/feature_tracker.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/timer.h"
#include "viewer/viewer.h"

Eigen::Matrix3d ric, ric1;
Eigen::Vector3d tic, tic1;

namespace {
using CameraInitResult = std::vector<tassel_core::Camera>;

CameraInitResult initializeCameras(const tassel_tools::Parameters& params) {
    CameraInitResult result;

    for (auto const& [id, T_ci] : params.T_cam_imu_map) {
        if (params.cam_intrinsic_map.find(id) == params.cam_intrinsic_map.end() ||
            params.cam_distort_map.find(id) == params.cam_distort_map.end()) {
            continue;
        }

        Eigen::Matrix3d ric_temp = T_ci.block<3, 3>(0, 0);
        Eigen::Vector3d tic_temp = T_ci.block<3, 1>(0, 3);

        cv::Mat k = params.cam_intrinsic_map.at(id);
        cv::Mat dist = params.cam_distort_map.at(id);

        result.emplace_back(
            std::make_unique<tassel_core::CameraRadTan>(k, dist, params.cols, params.rows));

        if (id == 0) {
            ric = ric_temp;
            tic = tic_temp;
        } else if (id == 1) {
            ric1 = ric_temp;
            tic1 = tic_temp;
        }
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tassel_core;

    // ── Load parameters from YAML config ──────────────────────────────────
    tassel_tools::Parameters params(
        (argc >= 2) ? argv[1] : "/home/adrewn/Tassel/config/stereo_vins.yaml");

    // ── ROS2 + Viewer ──────────────────────────────────────────────────────
    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("world");

    viewer->createImagePublisher("stereo/image");
    viewer->createOdometryPublisher("camera", "odom/camera");
    viewer->createPointCloudPublisher("landmarks");

    // ── OAK pipeline — left + right mono cameras, hardware sync ────────────
    dai::Pipeline pipeline;

    auto mono_left = pipeline.create<dai::node::MonoCamera>();
    mono_left->setCamera("left");
    mono_left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    auto xout_left = pipeline.create<dai::node::XLinkOut>();
    xout_left->setStreamName("left");
    mono_left->out.link(xout_left->input);

    auto mono_right = pipeline.create<dai::node::MonoCamera>();
    mono_right->setCamera("right");
    mono_right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    auto xout_right = pipeline.create<dai::node::XLinkOut>();
    xout_right->setStreamName("right");
    mono_right->out.link(xout_right->input);

    dai::Device device(pipeline);
    auto queue_left = device.getOutputQueue("left", 8, false);
    auto queue_right = device.getOutputQueue("right", 8, false);

    // ── Initialize cameras from Parameters ────────────────────────────────
    auto cameras = initializeCameras(params);

    // ── Feature tracker — left (id=0) and right (id=1) cameras ─────────────
    FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    tracker.addCamera(
        std::move(cameras[0]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius);
    tracker.addCamera(
        std::move(cameras[1]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius);

    // ── VO estimator — initialized from Parameters ──────────────────────────
    EstimatorOption option;
    option.optimize_enabled = params.optimize_enabled;
    option.marginalization_enabled = params.marginalization_enabled;
    option.num_iterations = params.num_iterations;
    option.lambda_initial = params.lambda_initial;
    option.reprojection_loss = HuberLoss{0.5};
    option.min_depth = params.min_depth;
    option.max_depth = params.max_depth;

    auto state = std::make_shared<State>(static_cast<int>(params.max_frame_count));
    auto feature_manager = std::make_shared<FeatureManager>(
        params.reprojection_error_thres, params.parallax_thres, params.tracked_times_thres,
        params.min_tracked_pts_num, params.min_pnp_num, params.min_pnp_inliers_ratio,
        params.min_translation, params.min_depth, params.max_depth);

    VoEstimator vo_estimator(option, state, feature_manager, ric, tic, ric1, tic1);

    int frame_count = 0;
    rclcpp::Rate rate(30);

    std::cout << "[VO] running — Ctrl-C to stop\n";

    while (rclcpp::ok()) {
        auto left_frame = queue_left->get<dai::ImgFrame>();
        auto right_frame = queue_right->get<dai::ImgFrame>();
        if (!left_frame || !right_frame) continue;

        double ts = left_frame->getTimestamp().time_since_epoch().count() / 1e9;
        auto left_data = left_frame->getData();
        cv::Mat left_img(
            left_frame->getHeight(), left_frame->getWidth(), CV_8UC1, left_data.data());
        left_img = left_img.clone();
        auto right_data = right_frame->getData();
        cv::Mat right_img(
            right_frame->getHeight(), right_frame->getWidth(), CV_8UC1, right_data.data());
        right_img = right_img.clone();

        // ── stereo tracking ────────────────────────────────────────────────
        std::unordered_map<int, FeaturePerFrame> feature_frame;
        {
            tassel_utils::Timer t("tracking");
            feature_frame = tracker.stereoTracking(0, left_img, 1, right_img);
        }

        vo_estimator.processMeasurement(ts, feature_frame);

        // ── Build stereo display image ──────────────────────────────────────
        cv::Mat disp_left, disp_right;
        cv::cvtColor(left_img, disp_left, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(0, disp_left);
        cv::cvtColor(right_img, disp_right, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(1, disp_right);

        cv::Mat stereo_disp;
        cv::hconcat(disp_left, disp_right, stereo_disp);

        viewer->publishImage("stereo/image", "camera", stereo_disp);

        // Publish odometry (latest optimized pose)
        if (state->cur_frame_count > 0) {
            Pose pose = state->poses[state->cur_frame_count - 1].get_optimized_pose();
            viewer->publishOdometry("odom/camera", pose.translation(), pose.unit_quaternion());
        }

        // Publish point cloud
        auto points = feature_manager->getPointCloud(
            *state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        viewer->publishPointCloud("landmarks", points);
        rclcpp::spin_some(viewer);
        rate.sleep();

        ++frame_count;
    }

    rclcpp::shutdown();

    // ── final report ───────────────────────────────────────────────────────
    std::cout << "\n[VO] done. " << frame_count << " frames, " << state->cur_frame_count
              << " keyframes in window.\n";
    if (state->cur_frame_count > 0) {
        std::cout << "Final pose:\n"
                  << state->poses[state->cur_frame_count - 1].get_pose().matrix() << "\n";
    }

    return 0;
}
