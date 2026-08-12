#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <vector>

#include "initial/initial_alignment.h"

namespace tassel_core {
namespace {

struct SyntheticAlignmentData {
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3d> velocities;
    std::vector<Eigen::Vector3d> delta_velocities;
    std::vector<Eigen::Vector3d> delta_positions;
    std::vector<double> dts;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    Eigen::Vector3d gravity;
    double scale;
};

SyntheticAlignmentData makeSyntheticAlignmentData() {
    SyntheticAlignmentData data;
    constexpr int kFrameCount = 8;
    data.ric = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix() *
               Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()).toRotationMatrix() *
               Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    data.tic = Eigen::Vector3d(0.08, -0.04, 0.06);
    data.gravity = Eigen::Vector3d(0.0, 0.0, 9.81);
    data.scale = 2.3;

    for (int i = 0; i < kFrameCount; ++i) {
        const double t = static_cast<double>(i);
        data.rotations.push_back(
            Eigen::AngleAxisd(0.11 * t, Eigen::Vector3d::UnitX()).toRotationMatrix() *
            Eigen::AngleAxisd(-0.07 * t * t / kFrameCount, Eigen::Vector3d::UnitY())
                .toRotationMatrix() *
            Eigen::AngleAxisd(0.05 * std::sin(t), Eigen::Vector3d::UnitZ()).toRotationMatrix());
        data.positions.emplace_back(
            0.4 * t + 0.03 * t * t, 0.2 * std::sin(0.7 * t), 0.15 * std::cos(0.4 * t));
        data.velocities.emplace_back(0.5 + 0.12 * t, -0.3 + 0.08 * t * t, 0.2 * std::cos(0.3 * t));
    }

    for (int i = 0; i + 1 < kFrameCount; ++i) {
        const double dt = 0.08 + 0.01 * std::sin(static_cast<double>(i));
        const Eigen::Matrix3d q_i = data.ric * data.rotations[i].transpose() * data.ric.transpose();
        const Eigen::Matrix3d r_i = q_i.transpose();
        const Eigen::Matrix3d r_j = data.ric * data.rotations[i + 1] * data.ric.transpose();
        const Eigen::Vector3d p_i =
            data.ric * (data.scale * data.positions[i]) - r_i * data.tic + data.tic;
        const Eigen::Vector3d p_j =
            data.ric * (data.scale * data.positions[i + 1]) - r_j * data.tic + data.tic;

        data.dts.push_back(dt);
        data.delta_velocities.push_back(
            q_i * (data.velocities[i + 1] - data.velocities[i] + data.gravity * dt));
        data.delta_positions.push_back(
            q_i * (p_j - p_i - data.velocities[i] * dt + 0.5 * data.gravity * dt * dt));
    }
    return data;
}

TEST(InitialAlignmentTest, RecoversLeverArmWithRotatedExtrinsics) {
    SyntheticAlignmentData data = makeSyntheticAlignmentData();
    std::vector<Eigen::Vector3d> velocities(data.velocities.size(), Eigen::Vector3d::Zero());
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    double scale = 0.0;

    ASSERT_TRUE(linearAlignment(
        data.rotations, data.positions, velocities, data.delta_velocities, data.delta_positions,
        data.dts, gravity, scale, data.ric, data.tic, 1e-8, 9.81));
    EXPECT_NEAR(scale, data.scale, 1e-8);
    EXPECT_TRUE(gravity.isApprox(data.gravity, 1e-8));
    for (size_t i = 0; i < velocities.size(); ++i) {
        EXPECT_TRUE(velocities[i].isApprox(data.velocities[i], 1e-8));
    }

    velocities.assign(data.velocities.size(), Eigen::Vector3d::Zero());
    gravity = Eigen::Vector3d(0.2, -0.15, 9.65);
    scale = 1.0;
    ASSERT_TRUE(refineGravitySpeeds(
        velocities, data.rotations, data.positions, data.delta_velocities, data.delta_positions,
        data.dts, gravity, scale, data.ric, data.tic, 9.81));
    EXPECT_NEAR(scale, data.scale, 1e-7);
    EXPECT_TRUE(gravity.isApprox(data.gravity, 1e-7));
    for (size_t i = 0; i < velocities.size(); ++i) {
        EXPECT_TRUE(velocities[i].isApprox(data.velocities[i], 1e-7));
    }
}

}  // namespace
}  // namespace tassel_core
