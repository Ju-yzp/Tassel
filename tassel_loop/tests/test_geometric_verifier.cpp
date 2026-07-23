#include "geometric_verifier.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <vector>

TEST(GeometricVerifierTest, RecoversCandidateFromCurrentMotion) {
    const Eigen::Matrix3d current_R_candidate =
        Eigen::AngleAxisd(0.12, Eigen::Vector3d(0.2, -0.5, 0.8).normalized()).toRotationMatrix();
    const Eigen::Vector3d current_t_candidate(0.4, -0.1, 0.2);
    std::vector<Eigen::Vector2d> candidate_points;
    std::vector<Eigen::Vector2d> current_points;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 10; ++col) {
            const Eigen::Vector3d point_candidate(
                -1.2 + 0.25 * col, -0.8 + 0.22 * row, 4.0 + 0.03 * row * col);
            const Eigen::Vector3d point_current =
                current_R_candidate * point_candidate + current_t_candidate;
            candidate_points.push_back(point_candidate.hnormalized());
            current_points.push_back(point_current.hnormalized());
        }
    }

    const tassel_loop::GeometricVerification result =
        tassel_loop::GeometricVerifier(30, 0.5, 1e-4).verify(candidate_points, current_points);

    ASSERT_TRUE(result.accepted);
    EXPECT_GE(result.inlier_count, 70);
    EXPECT_NEAR(
        Eigen::AngleAxisd(result.candidate_R_current * current_R_candidate).angle(), 0.0, 1e-4);
    const Eigen::Vector3d expected_direction =
        (-current_R_candidate.transpose() * current_t_candidate).normalized();
    EXPECT_GT(result.candidate_t_current.dot(expected_direction), 0.999);
}

TEST(GeometricVerifierTest, RejectsInsufficientMatches) {
    const std::vector<Eigen::Vector2d> points(4, Eigen::Vector2d::Zero());
    const auto result = tassel_loop::GeometricVerifier().verify(points, points);
    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(result.match_count, 4);
}

TEST(GeometricVerifierTest, RejectsMismatchedPointCounts) {
    EXPECT_THROW(
        tassel_loop::GeometricVerifier().verify(
            std::vector<Eigen::Vector2d>(5), std::vector<Eigen::Vector2d>(6)),
        std::invalid_argument);
}
