#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>

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

#include <depthai/depthai.hpp>

Eigen::Matrix3d ric, ric1;
Eigen::Vector3d tic, tic1;

namespace {

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

    tassel_tools::Parameters params(
        (argc >= 2) ? argv[1] : "/home/adrewn/Tassel/config/stereo_vins.yaml");

    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("world");

    viewer->createImagePublisher("stereo/image");
    viewer->createOdometryPublisher("camera", "odom/camera");
    viewer->createPathPublisher("vo/path");
    viewer->createPointCloudPublisher("landmarks");

    auto stereo_buffer = StereoBuf::createShared(15);
    auto imu_buffer = IMUBuf::createShared(600);
    imu_buffer->set_interpolator(imuInterpolate);
    SyncType sync(0, stereo_buffer.get(), imu_buffer.get());

    dai::Pipeline pipeline;

    auto mono_left = pipeline.create<dai::node::MonoCamera>();
    mono_left->setCamera("left");
    mono_left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    mono_left->setFps(15);
    mono_left->initialControl.setManualExposure(params.initial_exposure_time_us, 1200);

    auto mono_right = pipeline.create<dai::node::MonoCamera>();
    mono_right->setCamera("right");
    mono_right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    mono_right->setFps(15);
    mono_right->initialControl.setManualExposure(params.initial_exposure_time_us, 1200);

    auto imu_node = pipeline.create<dai::node::IMU>();
    imu_node->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 400);
    imu_node->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 400);
    imu_node->setBatchReportThreshold(1);

    auto sync_node = pipeline.create<dai::node::Sync>();
    mono_left->out.link(sync_node->inputs["left"]);
    mono_right->out.link(sync_node->inputs["right"]);
    sync_node->setSyncThreshold(std::chrono::milliseconds(10));

    auto xout_stereo = pipeline.create<dai::node::XLinkOut>();
    xout_stereo->setStreamName("stereo");
    sync_node->out.link(xout_stereo->input);

    auto xout_imu = pipeline.create<dai::node::XLinkOut>();
    xout_imu->setStreamName("imu");
    imu_node->out.link(xout_imu->input);

    dai::Device device(pipeline);

    device.getOutputQueue("stereo", 8, false)
        ->addCallback([&](std::shared_ptr<dai::ADatatype> data) {
            auto msg_group = std::static_pointer_cast<dai::MessageGroup>(data);
            auto left_frame = msg_group->get<dai::ImgFrame>("left");
            auto right_frame = msg_group->get<dai::ImgFrame>("right");
            if (!left_frame || !right_frame) return;

            auto stereo_msg = std::make_shared<tassel_utils::StereoObservation>();
            stereo_msg->timestamp = left_frame->getTimestamp().time_since_epoch().count() / 1e9;
            stereo_msg->left_img = cv::Mat(
                                       left_frame->getHeight(), left_frame->getWidth(), CV_8UC1,
                                       left_frame->getData().data())
                                       .clone();
            stereo_msg->right_img = cv::Mat(
                                        right_frame->getHeight(), right_frame->getWidth(), CV_8UC1,
                                        right_frame->getData().data())
                                        .clone();
            stereo_buffer->push_back(stereo_msg);
        });

    device.getOutputQueue("imu", 100, false)
        ->addCallback([&](std::shared_ptr<dai::ADatatype> data) {
            auto imu_data = std::static_pointer_cast<dai::IMUData>(data);
            for (auto& packet : imu_data->packets) {
                tassel_utils::IMUMeasurement m;
                m.timestamp = packet.acceleroMeter.getTimestamp().time_since_epoch().count() / 1e9;
                m.acc << packet.acceleroMeter.x, packet.acceleroMeter.y, packet.acceleroMeter.z;
                m.gyro << packet.gyroscope.x, packet.gyroscope.y, packet.gyroscope.z;
                imu_buffer->push_back(m);
            }
        });

    std::cout << "[VO] Stereo-IMU Synchronizer running. Waiting for data..." << std::endl;

    auto cameras = initializeCameras(params);
    const tassel_core::CameraBase* camera_ptr = cameras[0].get();

    FeatureTracker tracker(
        params.flow_back, params.max_square_move_dist, false, 5, params.min_gradient);
    tracker.addCamera(
        std::move(cameras[0]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius);
    tracker.addCamera(
        std::move(cameras[1]), params.per_grid_rows, params.per_grid_cols, params.edge_y,
        params.edge_x, params.mask_radius);

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
    auto state = std::make_shared<State>(static_cast<int>(params.max_frame_count));
    state->visual_sqrt_info = Eigen::Matrix2d::Identity() * params.visual_factor_weight;
    auto feature_manager = std::make_shared<FeatureManager>(
        params.reprojection_error_thres, params.parallax_thres, params.tracked_times_thres,
        params.min_tracked_pts_num, params.min_pnp_num, params.min_pnp_inliers_ratio,
        params.min_translation, params.min_depth, params.max_depth);

    Estimator estimator(option, state, feature_manager, ric, tic, ric1, tic1);
    estimator.setCamera(camera_ptr);
    estimator.setPoseCallback([&viewer](double /*ts*/, const Sophus::SE3d& pose) {
        viewer->publishOdometry("odom/camera", pose.translation(), pose.unit_quaternion());
        viewer->publishPath("vo/path", pose.translation(), pose.unit_quaternion());
    });
    estimator.setCloudCallback([&viewer](double /*ts*/, const std::vector<Eigen::Vector3d>& pts) {
        viewer->publishPointCloud("landmarks", pts);
    });

    rclcpp::Rate rate(30);

    while (rclcpp::ok()) {
        SyncType::DataPackage package;
        if (sync.pop_package(package)) {
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

            cv::Mat disp_left, disp_right;
            cv::cvtColor(stereo_ptr->left_img, disp_left, cv::COLOR_GRAY2BGR);
            tracker.drawTrackingResult(0, disp_left);
            cv::cvtColor(stereo_ptr->right_img, disp_right, cv::COLOR_GRAY2BGR);
            tracker.drawTrackingResult(1, disp_right);

            cv::Mat stereo_disp;
            cv::hconcat(disp_left, disp_right, stereo_disp);

            viewer->publishImage("stereo/image", "camera", stereo_disp);
        }
        rclcpp::spin_some(viewer);
        rate.sleep();
    }

    rclcpp::shutdown();

    std::cout << "\n[VO] done. " << state->cur_frame_count << " keyframes in window.\n";
    if (state->cur_frame_count > 0) {
        int idx = state->cur_frame_count - 1;
        std::cout << "Final pose:\n"
                  << Sophus::SE3d(state->Rs[idx], state->Ps[idx]).matrix() << "\n";
    }

    return 0;
}
