#include "pnp_verifier.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>

TEST(PnpVerifierTest, RecoversHostFromCurrentPoseWithOutliers) {
    const Eigen::Matrix3d host_R_current =
        Eigen::AngleAxisd(0.12, Eigen::Vector3d(0.2, -0.4, 0.9).normalized()).toRotationMatrix();
    const Eigen::Vector3d host_t_current(0.7, -0.25, 0.3);

    std::vector<Eigen::Vector3d> host_points;
    std::vector<Eigen::Vector2d> current_points;
    for (int y = -3; y <= 3; ++y) {
        for (int x = -4; x <= 4; ++x) {
            const Eigen::Vector3d point_current(0.18 * x, 0.16 * y, 3.5 + 0.07 * (x + y + 7));
            host_points.push_back(host_R_current * point_current + host_t_current);
            current_points.push_back(point_current.head<2>() / point_current.z());
        }
    }
    // 保持足够高的内点比例，同时用离群点验证鲁棒估计。
    for (size_t index = current_points.size() - 12; index < current_points.size(); ++index) {
        current_points[index] += Eigen::Vector2d(0.15, -0.12);
    }

    const tassel_loop::PnpVerification result =
        tassel_loop::PnpVerifier(20, 0.5, 0.006).verify(host_points, current_points);
    ASSERT_TRUE(result.accepted);
    EXPECT_GE(result.inlier_count, 45);
    EXPECT_LT(result.mean_reprojection_error, 1e-4);
    EXPECT_TRUE(std::isfinite(result.translation_variance));
    EXPECT_TRUE(std::isfinite(result.rotation_variance));
    EXPECT_GT(result.translation_variance, 0.0);
    EXPECT_GT(result.rotation_variance, 0.0);
    EXPECT_TRUE(result.host_R_current.isApprox(host_R_current, 1e-5));
    EXPECT_TRUE(result.host_t_current.isApprox(host_t_current, 1e-5));
}

TEST(PnpVerifierTest, RejectsInsufficientAndMismatchedCorrespondences) {
    tassel_loop::PnpVerifier verifier;
    EXPECT_FALSE(verifier.verify({}, {}).accepted);
    EXPECT_THROW(verifier.verify({Eigen::Vector3d::Ones()}, {}), std::invalid_argument);
}

TEST(PnpVerifierTest, RejectsInvalidOptions) {
    EXPECT_THROW(tassel_loop::PnpVerifier(5), std::invalid_argument);
    EXPECT_THROW(tassel_loop::PnpVerifier(20, 1.1), std::invalid_argument);
    EXPECT_THROW(tassel_loop::PnpVerifier(20, 0.5, 0.006, 1000, 0.999, 1), std::invalid_argument);
    EXPECT_THROW(
        tassel_loop::PnpVerifier(20, 0.5, 0.006, 1000, 0.999, 4, -1.0), std::invalid_argument);
}

TEST(PnpVerifierTest, ComputesRtabmapResidualQuantileVariance) {
    std::vector<Eigen::Vector3d> host_points;
    std::vector<Eigen::Vector2d> current_points;
    std::vector<double> translation_errors;
    for (int y = -3; y <= 3; ++y) {
        for (int x = -4; x <= 4; ++x) {
            const Eigen::Vector3d point(0.2 * x, 0.18 * y, 4.0 + 0.05 * (x - y));
            host_points.push_back(point);
            current_points.push_back(point.head<2>() / point.z());
            translation_errors.push_back(0.01 * point.squaredNorm());
        }
    }
    std::sort(translation_errors.begin(), translation_errors.end());
    const double expected_variance = 2.1981 * translation_errors[translation_errors.size() / 4];

    const tassel_loop::PnpVerification result =
        tassel_loop::PnpVerifier(20, 0.5, 0.006).verify(host_points, current_points);

    ASSERT_TRUE(result.accepted);
    EXPECT_NEAR(result.translation_variance, expected_variance, 1e-6);
    EXPECT_LT(result.rotation_variance, 1e-6);
}
