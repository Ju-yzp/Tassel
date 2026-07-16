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

}  // namespace
}  // namespace tassel_core
