#include <gtest/gtest.h>

#include <array>
#include <sophus/so3.hpp>

#include "state/state.h"

namespace tassel_core {
namespace {

TEST(StateTest, RotationParameterRoundTripIncludingNearPi) {
    const std::array<Eigen::Vector3d, 5> rotations = {
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.2, -0.3, 0.4),
        Eigen::Vector3d::UnitX() * (M_PI - 1e-8),
        Eigen::Vector3d(1.0, 2.0, -1.0).normalized() * (M_PI - 1e-7),
        Eigen::Vector3d(-2.0, 1.0, 3.0).normalized() * (M_PI - 1e-5),
    };

    State state(1);
    for (const Eigen::Vector3d& phi : rotations) {
        state.frames[0].rot_w_i = Sophus::SO3d::exp(phi).matrix();
        state.stateToParam(0);
        const Eigen::Vector3d converted(
            state.frames[0].param_pose[3], state.frames[0].param_pose[4],
            state.frames[0].param_pose[5]);
        const Eigen::Matrix3d reconstructed = Sophus::SO3d::exp(converted).matrix();
        EXPECT_TRUE(reconstructed.isApprox(state.frames[0].rot_w_i, 1e-10))
            << "phi=" << phi.transpose() << " converted=" << converted.transpose();
    }
}

TEST(StateTest, ResetClearsFrameIds) {
    State state(3);
    state.frames[0].frame_id = 1;
    state.captureGauge(0);
    for (int i = 0; i < 3; ++i) {
        state.frames[i].frame_id = 10 * (i + 1);
    }

    state.reset();

    for (const auto& frame : state.frames) {
        EXPECT_EQ(frame.frame_id, tassel_utils::kInvalidFrameId);
    }
    EXPECT_FALSE(state.gauge_reference.has_value());
}

TEST(StateTest, CapturesGaugeFromOptimizedFrame) {
    State state(2);
    state.latest_active_frame_index = 1;
    state.frames[1].frame_id = 42;
    state.frames[1].rot_w_i = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)).matrix();
    state.frames[1].pos_w_i = Eigen::Vector3d(1.0, 2.0, 3.0);

    state.captureGauge(1);

    ASSERT_TRUE(state.gauge_reference.has_value());
    EXPECT_EQ(state.gauge_reference->reference_frame_id, 42);
    EXPECT_TRUE(state.gauge_reference->reference_rotation.isApprox(state.frames[1].rot_w_i));
    EXPECT_EQ(state.gauge_reference->reference_position, state.frames[1].pos_w_i);
}

TEST(StateTest, RejectsInvalidWindowSizeBeforeAllocation) {
    EXPECT_THROW(State(-1), std::runtime_error);
    EXPECT_THROW(State(0), std::runtime_error);
}

TEST(StateTest, CopyFrameStateCopiesCompletePhysicalState) {
    State state(2);
    state.frames[1].rot_w_i = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)).matrix();
    state.frames[1].pos_w_i = Eigen::Vector3d(1.0, 2.0, 3.0);
    state.frames[1].vel_w = Eigen::Vector3d(4.0, 5.0, 6.0);
    state.frames[1].accel_bias = Eigen::Vector3d(0.1, 0.2, 0.3);
    state.frames[1].gyro_bias = Eigen::Vector3d(0.01, 0.02, 0.03);
    state.frames[1].image_sync_delay = 0.004;
    state.frames[1].frame_id = 123456789;
    state.frames[1].frame_type = FrameType::KeyFrame;

    state.copyFrameState(1, 0);

    EXPECT_TRUE(state.frames[0].rot_w_i.isApprox(state.frames[1].rot_w_i));
    EXPECT_EQ(state.frames[0].pos_w_i, state.frames[1].pos_w_i);
    EXPECT_EQ(state.frames[0].vel_w, state.frames[1].vel_w);
    EXPECT_EQ(state.frames[0].accel_bias, state.frames[1].accel_bias);
    EXPECT_EQ(state.frames[0].gyro_bias, state.frames[1].gyro_bias);
    EXPECT_EQ(state.frames[0].image_sync_delay, state.frames[1].image_sync_delay);
    EXPECT_EQ(state.frames[0].frame_id, state.frames[1].frame_id);
    EXPECT_EQ(state.frames[0].frame_type, FrameType::KeyFrame);
}

}  // namespace
}  // namespace tassel_core
