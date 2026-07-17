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
        state.Rs[0] = Sophus::SO3d::exp(phi).matrix();
        state.stateToParam(0);
        const Eigen::Vector3d converted(
            state.params_pose[0][3], state.params_pose[0][4], state.params_pose[0][5]);
        const Eigen::Matrix3d reconstructed = Sophus::SO3d::exp(converted).matrix();
        EXPECT_TRUE(reconstructed.isApprox(state.Rs[0], 1e-10))
            << "phi=" << phi.transpose() << " converted=" << converted.transpose();
    }
}

TEST(StateTest, ResetClearsFrameIds) {
    State state(3);
    state.frame_ids = {10, 20, 30};

    state.reset();

    for (const auto id : state.frame_ids) {
        EXPECT_EQ(id, tassel_utils::kInvalidFrameId);
    }
}

TEST(StateTest, RejectsInvalidWindowSizeBeforeAllocation) {
    EXPECT_THROW(State(-1), std::runtime_error);
    EXPECT_THROW(State(0), std::runtime_error);
}

TEST(StateTest, InvalidFrameIdNeverResolvesToEmptySlot) {
    State state(3);
    state.cur_frame_count = 2;
    state.frame_ids = {100, 200, tassel_utils::kInvalidFrameId};
    EXPECT_EQ(state.findFrameSlot(tassel_utils::kInvalidFrameId), -1);
}

TEST(StateTest, CopyFrameSlotCopiesCompletePhysicalState) {
    State state(2);
    state.Rs[1] = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)).matrix();
    state.Ps[1] = Eigen::Vector3d(1.0, 2.0, 3.0);
    state.Vs[1] = Eigen::Vector3d(4.0, 5.0, 6.0);
    state.Bas[1] = Eigen::Vector3d(0.1, 0.2, 0.3);
    state.Bgs[1] = Eigen::Vector3d(0.01, 0.02, 0.03);
    state.frame_delays[1] = 0.004;
    state.frame_ids[1] = 123456789;

    state.copyFrameSlot(1, 0);

    EXPECT_TRUE(state.Rs[0].isApprox(state.Rs[1]));
    EXPECT_EQ(state.Ps[0], state.Ps[1]);
    EXPECT_EQ(state.Vs[0], state.Vs[1]);
    EXPECT_EQ(state.Bas[0], state.Bas[1]);
    EXPECT_EQ(state.Bgs[0], state.Bgs[1]);
    EXPECT_EQ(state.frame_delays[0], state.frame_delays[1]);
    EXPECT_EQ(state.frame_ids[0], state.frame_ids[1]);
}

TEST(StateTest, ActiveImuRangeSkipsRetainedHostPlaceholder) {
    State state(3);
    EXPECT_EQ(state.firstActiveImuSlot(), 0);

    state.has_retained_host = true;
    EXPECT_EQ(state.firstActiveImuSlot(), 1);
}

}  // namespace
}  // namespace tassel_core
