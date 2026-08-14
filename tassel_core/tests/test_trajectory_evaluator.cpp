#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <limits>
#include <stdexcept>

#include "evaluation/trajectory_evaluator.h"

namespace tassel_core::evaluation {
namespace {

TEST(TrajectoryEvaluator, InterpolatesPositionAndRotationAtRequestedTimestamp) {
    const Sophus::SE3d start;
    const Sophus::SE3d end(
        Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
        Eigen::Vector3d(2.0, 4.0, 6.0));
    const std::vector<TimedPose> poses{{10.0, start}, {12.0, end}};

    const auto pose = interpolatePose(poses, 10.5, 2.0);

    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->translation().isApprox(Eigen::Vector3d(0.5, 1.0, 1.5), 1e-12));
    EXPECT_NEAR(pose->so3().log().norm(), M_PI / 4.0, 1e-12);
}

TEST(TrajectoryEvaluator, RejectsInterpolationAcrossGroundTruthGap) {
    const std::vector<TimedPose> poses{{10.0, Sophus::SE3d()}, {12.0, Sophus::SE3d()}};

    EXPECT_FALSE(interpolatePose(poses, 11.0, 0.05).has_value());
}

TEST(TrajectoryEvaluator, AcceptsExactSampleBesideGroundTruthGap) {
    const Sophus::SE3d expected(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 2.0, 3.0));
    const std::vector<TimedPose> poses{{10.0, Sophus::SE3d()}, {12.0, expected}};

    const auto pose = interpolatePose(poses, 12.0, 0.05);

    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->matrix().isApprox(expected.matrix(), 1e-12));
}

TEST(TrajectoryEvaluator, RejectsInvalidMaximumInterpolationInterval) {
    const std::vector<TimedPose> poses{{10.0, Sophus::SE3d()}, {12.0, Sophus::SE3d()}};

    EXPECT_THROW(interpolatePose(poses, 11.0, 0.0), std::invalid_argument);
    EXPECT_THROW(
        interpolatePose(poses, std::numeric_limits<double>::quiet_NaN(), 0.05),
        std::invalid_argument);
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
