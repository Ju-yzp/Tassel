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
        state.frames[0].R = Sophus::SO3d::exp(phi).matrix();
        state.stateToParam(0);
        const Eigen::Vector3d converted(
            state.frames[0].pose[3], state.frames[0].pose[4], state.frames[0].pose[5]);
        const Eigen::Matrix3d reconstructed = Sophus::SO3d::exp(converted).matrix();
        EXPECT_TRUE(reconstructed.isApprox(state.frames[0].R, 1e-10))
            << "phi=" << phi.transpose() << " converted=" << converted.transpose();
    }
}

TEST(StateTest, ResetClearsFrameIds) {
    State state(3);
    for (int i = 0; i < 3; ++i) {
        state.frames[i].timestamp_ns = 10 * (i + 1);
    }

    state.reset();

    for (const auto& frame : state.frames) {
        EXPECT_EQ(frame.timestamp_ns, tassel_utils::kInvalidFrameId);
    }
}

TEST(StateTest, RejectsInvalidWindowSizeBeforeAllocation) {
    EXPECT_THROW(State(-1), std::runtime_error);
    EXPECT_THROW(State(0), std::runtime_error);
}

TEST(StateTest, CopyFrameStateCopiesCompletePhysicalState) {
    State state(2);
    state.frames[1].R = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)).matrix();
    state.frames[1].P = Eigen::Vector3d(1.0, 2.0, 3.0);
    state.frames[1].V = Eigen::Vector3d(4.0, 5.0, 6.0);
    state.frames[1].Ba = Eigen::Vector3d(0.1, 0.2, 0.3);
    state.frames[1].Bg = Eigen::Vector3d(0.01, 0.02, 0.03);
    state.frames[1].sync_delay = 0.004;
    state.frames[1].timestamp_ns = 123456789;
    state.frames[1].type = FrameType::KeyFrame;

    state.copyFrameState(1, 0);

    EXPECT_TRUE(state.frames[0].R.isApprox(state.frames[1].R));
    EXPECT_EQ(state.frames[0].P, state.frames[1].P);
    EXPECT_EQ(state.frames[0].V, state.frames[1].V);
    EXPECT_EQ(state.frames[0].Ba, state.frames[1].Ba);
    EXPECT_EQ(state.frames[0].Bg, state.frames[1].Bg);
    EXPECT_EQ(state.frames[0].sync_delay, state.frames[1].sync_delay);
    EXPECT_EQ(state.frames[0].timestamp_ns, state.frames[1].timestamp_ns);
    EXPECT_EQ(state.frames[0].type, FrameType::KeyFrame);
}

}  // namespace
}  // namespace tassel_core
