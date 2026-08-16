#include <gtest/gtest.h>

#include <array>
#include <limits>
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
    for (int i = 0; i < 3; ++i) {
        state.frames[i].frame_id = 10 * (i + 1);
    }

    state.reset();

    for (const auto& frame : state.frames) {
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(frame->frame_id, tassel_utils::kInvalidFrameId);
    }
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

TEST(StateTest, CapturesFrameAndDelayLinearizationOnlyOnce) {
    State state(1);
    FrameState& frame = state.frames[0];
    frame.frame_id = 10;
    frame.pos_w_i = Eigen::Vector3d(1.0, 2.0, 3.0);
    frame.vel_w = Eigen::Vector3d(4.0, 5.0, 6.0);
    state.time_delay = 0.01;

    state.stateToParams();
    const std::array<double, 6> first_pose = frame.linearized_pose;
    const std::array<double, 9> first_speed_bias = frame.linearized_speed_bias;

    frame.pos_w_i = Eigen::Vector3d(7.0, 8.0, 9.0);
    frame.vel_w = Eigen::Vector3d(10.0, 11.0, 12.0);
    state.time_delay = 0.02;
    state.stateToParams();

    EXPECT_EQ(frame.linearized_pose, first_pose);
    EXPECT_EQ(frame.linearized_speed_bias, first_speed_bias);
    EXPECT_DOUBLE_EQ(*state.getLinearizedTimeDelay(), 0.01);
    EXPECT_DOUBLE_EQ(*state.getCurrentTimeDelay(), 0.02);
    EXPECT_DOUBLE_EQ(frame.getCurrentPose()[0], 7.0);
    EXPECT_DOUBLE_EQ(frame.getLinearizedPose()[0], 1.0);
}

TEST(StateTest, ExplicitTimeDelayCaptureRejectsInvalidAndDoesNotOverwrite) {
    State state(1);
    state.param_time_delay = 0.012;
    state.captureLinearizedTimeDelay();
    state.param_time_delay = 0.031;
    state.captureLinearizedTimeDelay();
    EXPECT_DOUBLE_EQ(*state.getLinearizedTimeDelay(), 0.012);

    State invalid(1);
    invalid.param_time_delay = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(invalid.captureLinearizedTimeDelay(), std::logic_error);
}

TEST(StateTest, SeedingNewFrameClearsInheritedLinearizationIdentity) {
    State state(2);
    state.frames[0].frame_id = 10;
    state.frames[0].stateToParam();
    ASSERT_TRUE(state.frames[0].has_linearized);

    state.seedFrameState(0, 1);

    EXPECT_FALSE(state.frames[1].has_linearized);
    EXPECT_THROW(state.frames[1].getLinearizedPose(), std::logic_error);
    EXPECT_TRUE(state.frames[1].pos_w_i.isApprox(state.frames[0].pos_w_i));
}

TEST(StateTest, FrameClonePreservesDynamicType) {
    const std::array<std::unique_ptr<Frame>, 3> frames = {
        std::make_unique<NormalFrame>(), std::make_unique<KeyFrame>(),
        std::make_unique<RetainedFrame>()};

    EXPECT_NE(dynamic_cast<NormalFrame*>(frames[0]->clone().get()), nullptr);
    EXPECT_NE(dynamic_cast<KeyFrame*>(frames[1]->clone().get()), nullptr);
    EXPECT_NE(dynamic_cast<RetainedFrame*>(frames[2]->clone().get()), nullptr);
}

TEST(StateTest, FrameStorageReplaceTypePreservesState) {
    State state(1);
    state.frames[0].frame_id = 77;
    state.frames[0].pos_w_i = Eigen::Vector3d(1.0, 2.0, 3.0);

    state.frames.replaceType(0, state.frames[0], true);
    EXPECT_NE(dynamic_cast<KeyFrame*>(&state.frames[0]), nullptr);
    EXPECT_EQ(state.frames[0].frame_id, 77);
    EXPECT_EQ(state.frames[0].pos_w_i, Eigen::Vector3d(1.0, 2.0, 3.0));

    state.frames.replaceType(0, state.frames[0], false);
    EXPECT_NE(dynamic_cast<NormalFrame*>(&state.frames[0]), nullptr);
    EXPECT_EQ(state.frames[0].frame_id, 77);
    EXPECT_EQ(state.frames[0].pos_w_i, Eigen::Vector3d(1.0, 2.0, 3.0));
}

TEST(StateTest, RetainedFrameIsOwnedIndependentlyFromActiveMirror) {
    State state(2);
    state.frames[1].frame_id = 42;
    state.frames[1].pos_w_i = Eigen::Vector3d(1.0, 2.0, 3.0);

    state.replaceRetainedFrame(state.frames[1]);
    state.copyRetainedToFrame(0);
    state.frames[0].pos_w_i.x() = 9.0;

    EXPECT_EQ(state.retainedFrame().frame_id, 42);
    EXPECT_DOUBLE_EQ(state.retainedFrame().pos_w_i.x(), 1.0);
    EXPECT_DOUBLE_EQ(state.frames[0].pos_w_i.x(), 9.0);
    EXPECT_NE(dynamic_cast<const RetainedFrame*>(&state.retainedFrame()), nullptr);
}

}  // namespace
}  // namespace tassel_core
