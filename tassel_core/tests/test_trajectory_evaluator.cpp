#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "evaluation/trajectory_evaluator.h"

namespace tassel_core::evaluation {
namespace {

TEST(TrajectoryEvaluator, InterpolatesPositionAndRotationAtRequestedTimestamp) {
    const Sophus::SE3d start;
    const Sophus::SE3d end(
        Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
        Eigen::Vector3d(2.0, 4.0, 6.0));
    const std::vector<TimedPose> poses{{10.0, start}, {12.0, end}};

    const auto pose = interpolatePose(poses, 10.5);

    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->translation().isApprox(Eigen::Vector3d(0.5, 1.0, 1.5), 1e-12));
    EXPECT_NEAR(pose->so3().log().norm(), M_PI / 4.0, 1e-12);
}

TEST(TrajectoryEvaluator, RemovesKnownYawAndTranslationWithoutChangingScale) {
    const Sophus::SE3d truth_from_estimate(
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
        Eigen::Vector3d(3.0, -2.0, 0.7));
    std::vector<PosePair> poses;
    for (int i = 0; i < 4; ++i) {
        const Sophus::SE3d estimate(
            Eigen::AngleAxisd(0.1 * i, Eigen::Vector3d::UnitX()).toRotationMatrix(),
            Eigen::Vector3d(0.5 * i, 0.2 * i * i, -0.1 * i));
        poses.push_back({static_cast<double>(i), estimate, truth_from_estimate * estimate});
    }

    const TrajectoryError error = evaluateTrajectory(poses);

    EXPECT_NEAR(error.position_rmse, 0.0, 1e-12);
    EXPECT_NEAR(error.terminal_position_error, 0.0, 1e-12);
    EXPECT_NEAR(error.rotation_rmse, 0.0, 1e-12);
    EXPECT_TRUE(error.alignment.matrix().isApprox(truth_from_estimate.matrix(), 1e-12));
}

}  // namespace
}  // namespace tassel_core::evaluation
