#include <gtest/gtest.h>

#include <array>

#include "state/state.h"

namespace tassel_core {
namespace {

TEST(StateTest, WindowMovePreservesFejAndSeedsLatestFromPosterior) {
    State state(3);
    FrameState& first = state.frames[0];
    first.frame_id = 10;
    first.frame_type = FrameType::KeyFrame;
    first.pos_w_i = Eigen::Vector3d(1.0, 2.0, 3.0);
    first.vel_w = Eigen::Vector3d(4.0, 5.0, 6.0);
    state.time_delay = 0.01;
    state.prepareOptimization();

    const std::array<double, 6> first_linearized_pose = first.linearized_pose;
    const std::array<double, 9> first_linearized_speed_bias = first.linearized_speed_bias;

    first.param_pose[0] = 7.0;
    first.param_pose[1] = 8.0;
    first.param_pose[2] = 9.0;
    first.param_speed_bias[0] = 10.0;
    first.param_speed_bias[1] = 11.0;
    first.param_speed_bias[2] = 12.0;
    state.param_time_delay = 0.02;
    state.acceptOptimization();

    const Eigen::Matrix3d gauge_rotation =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d source_origin(0.5, -0.4, 0.2);
    const Eigen::Vector3d target_origin(-1.0, 0.7, 0.3);
    const Eigen::Vector3d expected_current_position =
        gauge_rotation * (first.pos_w_i - source_origin) + target_origin;
    const Eigen::Vector3d first_linearized_position(
        first_linearized_pose[0], first_linearized_pose[1], first_linearized_pose[2]);
    const Eigen::Vector3d expected_linearized_position =
        gauge_rotation * (first_linearized_position - source_origin) + target_origin;
    const Eigen::Vector3d first_linearized_velocity(
        first_linearized_speed_bias[0], first_linearized_speed_bias[1],
        first_linearized_speed_bias[2]);
    first.transformGauge(gauge_rotation, source_origin, target_origin);
    EXPECT_TRUE(first.pos_w_i.isApprox(expected_current_position, 1e-12));
    EXPECT_TRUE(Eigen::Vector3d(
                    first.linearized_pose[0], first.linearized_pose[1], first.linearized_pose[2])
                    .isApprox(expected_linearized_position, 1e-12));
    EXPECT_TRUE(Eigen::Vector3d(
                    first.linearized_speed_bias[0], first.linearized_speed_bias[1],
                    first.linearized_speed_bias[2])
                    .isApprox(gauge_rotation * first_linearized_velocity, 1e-12));

    state.copyFrame(0, 1);
    EXPECT_EQ(state.frames[1].linearized_pose, first.linearized_pose);
    EXPECT_EQ(state.frames[1].linearized_speed_bias, first.linearized_speed_bias);

    state.seedFrameFromPosterior(1, 2);
    const FrameState& latest = state.frames[2];
    EXPECT_EQ(latest.frame_id, tassel_utils::kInvalidFrameId);
    EXPECT_EQ(latest.frame_type, FrameType::Unknown);
    EXPECT_FALSE(latest.has_linearized);
    EXPECT_EQ(latest.pos_w_i, first.pos_w_i);
    EXPECT_EQ(latest.vel_w, first.vel_w);
    EXPECT_THROW(latest.linearizedPose(), std::logic_error);

    state.frames[2].frame_id = 20;
    state.prepareOptimization();
    const std::array<double, 6> latest_linearized_pose = state.frames[2].linearized_pose;
    EXPECT_EQ(latest_linearized_pose, state.frames[2].param_pose);
    EXPECT_DOUBLE_EQ(*state.linearizedTimeDelay(), 0.01);
    EXPECT_DOUBLE_EQ(*state.currentTimeDelay(), 0.02);

    state.frames[2].pos_w_i.x() = 13.0;
    state.prepareOptimization();
    EXPECT_EQ(state.frames[2].linearized_pose, latest_linearized_pose);
    EXPECT_DOUBLE_EQ(state.frames[2].param_pose[0], 13.0);
}

}  // namespace
}  // namespace tassel_core
