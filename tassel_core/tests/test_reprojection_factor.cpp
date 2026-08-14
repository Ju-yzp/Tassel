#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <sophus/so3.hpp>

#include "cam/camera_rad_tan.h"
#include "factor/reprojection_factor.h"
#include "factor/visual_frame_cache.h"
#include "imu_test_utils.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

constexpr double kDifferenceStep = 1e-6;
constexpr double kJacobianTolerance = 2e-5;

void expectJacobianNear(
    const Eigen::MatrixXd& analytic, const Eigen::MatrixXd& numeric, const char* parameter,
    size_t landmark_index, double delay) {
    ASSERT_EQ(analytic.rows(), numeric.rows());
    ASSERT_EQ(analytic.cols(), numeric.cols());
    const double scale =
        std::max({1.0, analytic.cwiseAbs().maxCoeff(), numeric.cwiseAbs().maxCoeff()});
    EXPECT_LE((analytic - numeric).cwiseAbs().maxCoeff(), kJacobianTolerance * scale)
        << "parameter=" << parameter << " landmark=" << landmark_index << " delay=" << delay;
}

class ReprojectionFactorTest : public ::testing::Test {
protected:
    struct Landmark {
        Eigen::Vector3d host_ray;
        Eigen::Vector2d target_pixel;
        double inverse_depth = 0.0;
    };

    void SetUp() override {
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d(0.05, 0.0, 0.0);
        accel_bias_ = Eigen::Vector3d(0.08, -0.04, 0.03);
        gyro_bias_ = Eigen::Vector3d(0.012, -0.006, 0.004);
        constexpr double true_delay = 0.005;
        sqrt_info_ = Eigen::Matrix2d::Identity() * 320.0;

        const Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();
        const Eigen::VectorXd distortion = Eigen::VectorXd::Zero(4);
        camera_ = std::make_unique<CameraRadTan>(intrinsics, distortion, 640, 480);

        constexpr double imu_dt = 0.0025;
        const Eigen::Vector3d acceleration(0.3, -0.1, 0.05);
        const Eigen::Vector3d angular_velocity(0.1, 0.3, 0.4);
        const test::ImuTimeline timeline = test::generateConstantMotionTimeline(
            2.0, imu_dt, acceleration, angular_velocity, accel_bias_, gyro_bias_);

        constexpr int host_sample = 200;
        constexpr int target_sample = 227;
        const int delay_steps = static_cast<int>(std::round(true_delay / imu_dt));
        const auto& host_query = timeline.at_index(host_sample + delay_steps);
        const auto& target_query = timeline.at_index(target_sample + delay_steps);
        const auto& host = timeline.at_index(host_sample);
        const auto& target = timeline.at_index(target_sample);
        initializeFrame(host, state_.frames[0]);
        initializeFrame(target, state_.frames[1]);
        state_.latest_active_frame_index = 1;

        const std::vector<Eigen::Vector3d> host_points = {{0.3, -0.2, 1.5}, {-0.5, 0.3, 3.0},
                                                          {0.2, -0.4, 2.0}, {1.2, -0.1, 8.0},
                                                          {-1.0, 0.5, 6.0}, {0.01, 0.01, 12.0}};
        for (const Eigen::Vector3d& host_point : host_points) {
            Landmark landmark;
            landmark.host_ray = host_point / host_point.z();
            landmark.inverse_depth = 1.0 / host_point.z();
            const Eigen::Vector3d host_imu_point = ric_ * host_point + tic_;
            const Eigen::Vector3d world_point =
                host_query.rot_w_i * host_imu_point + host_query.pos_w_i;
            const Eigen::Vector3d target_imu_point =
                target_query.rot_w_i.transpose() * (world_point - target_query.pos_w_i);
            const Eigen::Vector3d target_camera_point =
                ric_.transpose() * (target_imu_point - tic_);
            landmark.target_pixel = target_camera_point.head<2>() / target_camera_point.z();
            landmarks_.push_back(landmark);
        }
    }

    void initializeFrame(const test::ImuSample& sample, FrameState& frame) {
        const Eigen::Vector3d rotation_vector = Sophus::SO3d(sample.rot_w_i).log();
        for (int d = 0; d < 3; ++d) {
            frame.param_pose[d] = sample.pos_w_i[d];
            frame.param_pose[d + 3] = rotation_vector[d];
            frame.param_speed_bias[d] = sample.vel_w[d];
            frame.param_speed_bias[d + 3] = accel_bias_[d];
            frame.param_speed_bias[d + 6] = gyro_bias_[d];
        }
        frame.imu_gyro = sample.gyro;
        frame.imu_acc = sample.acc;
    }

    std::unique_ptr<ReprojectionFactor> makeFactor(size_t landmark_index) {
        const Landmark& landmark = landmarks_[landmark_index];
        return std::make_unique<ReprojectionFactor>(
            landmark.host_ray, landmark.target_pixel, ric_, tic_, state_.frames[0].imu_gyro,
            state_.frames[1].imu_gyro, state_.frames[0].imu_acc, state_.frames[1].imu_acc,
            state_.frames[0].param_speed_bias.data(), state_.frames[1].param_speed_bias.data(),
            state_.frames[0].param_speed_bias.data() + 6,
            state_.frames[1].param_speed_bias.data() + 6,
            state_.frames[0].param_speed_bias.data() + 3,
            state_.frames[1].param_speed_bias.data() + 3, sqrt_info_, camera_.get(), 0.0, 0.0,
            &state_, 0, 1);
    }

    Eigen::Vector2d evaluate(
        ReprojectionFactor& factor, VisualFrameCache& cache, double inverse_depth) {
        cache.PrepareForEvaluation(false, true);
        const double* parameters[] = {
            state_.frames[0].param_pose.data(), state_.frames[1].param_pose.data(),
            &state_.param_time_delay, &inverse_depth};
        Eigen::Vector2d residual;
        EXPECT_TRUE(factor.Evaluate(parameters, residual.data(), nullptr));
        return residual;
    }

    Eigen::Matrix<double, 2, 6> numericPoseJacobian(
        ReprojectionFactor& factor, VisualFrameCache& cache, int frame_index,
        double inverse_depth) {
        SE3RightManifold manifold;
        const std::array<double, 6> original = state_.frames[frame_index].param_pose;
        Eigen::Matrix<double, 2, 6> jacobian;
        for (int column = 0; column < 6; ++column) {
            Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
            delta[column] = kDifferenceStep;
            manifold.Plus(
                original.data(), delta.data(), state_.frames[frame_index].param_pose.data());
            const Eigen::Vector2d positive = evaluate(factor, cache, inverse_depth);
            manifold.Plus(
                original.data(), (-delta).eval().data(),
                state_.frames[frame_index].param_pose.data());
            const Eigen::Vector2d negative = evaluate(factor, cache, inverse_depth);
            jacobian.col(column) = (positive - negative) / (2.0 * kDifferenceStep);
        }
        state_.frames[frame_index].param_pose = original;
        return jacobian;
    }

    State state_{2};
    Eigen::Matrix3d ric_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
    Eigen::Matrix2d sqrt_info_ = Eigen::Matrix2d::Identity();
    std::unique_ptr<CameraRadTan> camera_;
    std::vector<Landmark> landmarks_;
};

TEST_F(ReprojectionFactorTest, AnalyticJacobiansMatchNumericDifferentiation) {
    SE3RightManifold manifold;
    VisualFrameCache cache(state_, ric_);
    for (const double delay : {0.005, 0.2}) {
        state_.param_time_delay = delay;
        for (size_t landmark_index = 0; landmark_index < landmarks_.size(); ++landmark_index) {
            std::unique_ptr<ReprojectionFactor> factor = makeFactor(landmark_index);
            const double inverse_depth = landmarks_[landmark_index].inverse_depth;
            const double* parameters[] = {
                state_.frames[0].param_pose.data(), state_.frames[1].param_pose.data(),
                &state_.param_time_delay, &inverse_depth};
            double residual[2];
            double host_pose_data[12], target_pose_data[12], delay_data[2], depth_data[2];
            double* jacobians[] = {host_pose_data, target_pose_data, delay_data, depth_data};
            cache.PrepareForEvaluation(true, true);
            ASSERT_TRUE(factor->Evaluate(parameters, residual, jacobians));

            double host_plus_data[36], target_plus_data[36];
            ASSERT_TRUE(manifold.PlusJacobian(parameters[0], host_plus_data));
            ASSERT_TRUE(manifold.PlusJacobian(parameters[1], target_plus_data));
            const Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> host_ambient(
                host_pose_data);
            const Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> target_ambient(
                target_pose_data);
            const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> host_plus(
                host_plus_data);
            const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> target_plus(
                target_plus_data);

            expectJacobianNear(
                host_ambient * host_plus, numericPoseJacobian(*factor, cache, 0, inverse_depth),
                "host_pose", landmark_index, delay);
            expectJacobianNear(
                target_ambient * target_plus, numericPoseJacobian(*factor, cache, 1, inverse_depth),
                "target_pose", landmark_index, delay);

            const double original_delay = state_.param_time_delay;
            state_.param_time_delay = original_delay + kDifferenceStep;
            const Eigen::Vector2d delay_positive = evaluate(*factor, cache, inverse_depth);
            state_.param_time_delay = original_delay - kDifferenceStep;
            const Eigen::Vector2d delay_negative = evaluate(*factor, cache, inverse_depth);
            state_.param_time_delay = original_delay;
            expectJacobianNear(
                Eigen::Map<const Eigen::Vector2d>(delay_data),
                (delay_positive - delay_negative) / (2.0 * kDifferenceStep), "time_delay",
                landmark_index, delay);

            const Eigen::Vector2d depth_positive =
                evaluate(*factor, cache, inverse_depth + kDifferenceStep);
            const Eigen::Vector2d depth_negative =
                evaluate(*factor, cache, inverse_depth - kDifferenceStep);
            expectJacobianNear(
                Eigen::Map<const Eigen::Vector2d>(depth_data),
                (depth_positive - depth_negative) / (2.0 * kDifferenceStep), "inverse_depth",
                landmark_index, delay);
        }
    }
}

}  // namespace
}  // namespace tassel_core
