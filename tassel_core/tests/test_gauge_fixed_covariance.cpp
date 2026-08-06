#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "marg/gauge_fixed_covariance.h"
#include "marg/schmidt/schmidt_update.h"

namespace tassel_core {
namespace {

TEST(GaugeFixedCovarianceTest, RemovesOnlyAnchorPositionAndGlobalYaw) {
    constexpr int state_size = 10;
    const Eigen::Matrix3d rotation = (Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()) *
                                      Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
                                      Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()))
                                         .toRotationMatrix();
    const GaugeFixedBasis basis = buildPositionYawGaugeBasis(state_size, 0, rotation);

    ASSERT_EQ(basis.tangent.rows(), state_size);
    ASSERT_EQ(basis.tangent.cols(), state_size - 4);
    EXPECT_TRUE((basis.tangent.transpose() * basis.tangent)
                    .isApprox(Eigen::MatrixXd::Identity(state_size - 4, state_size - 4), 1e-12));
    EXPECT_TRUE(basis.tangent.topRows(3).isZero(1e-12));
    const Eigen::Vector3d yaw_direction = rotation.transpose() * Eigen::Vector3d::UnitZ();
    EXPECT_TRUE(
        (yaw_direction.transpose() * basis.tangent.block(3, 0, 3, state_size - 4)).isZero(1e-12));
    EXPECT_EQ(basis.full_to_reduced[6], 2);
}

TEST(GaugeFixedCovarianceTest, ComputesCovarianceOnExplicitGaugeSlice) {
    constexpr int state_size = 10;
    const GaugeFixedBasis basis =
        buildPositionYawGaugeBasis(state_size, 0, Eigen::Matrix3d::Identity());
    Eigen::MatrixXd reduced_sqrt = Eigen::MatrixXd::Identity(state_size - 4, state_size - 4);
    reduced_sqrt.diagonal() << 1.0, 2.0, 1.5, 0.8, 1.2, 2.5;
    const Eigen::MatrixXd full_sqrt = reduced_sqrt * basis.tangent.transpose();

    EXPECT_THROW(covarianceFromFullRankSqrtInformation(full_sqrt), std::invalid_argument);
    const GaugeFixedCovariance covariance = covarianceInFixedGauge(full_sqrt, basis);
    const Eigen::MatrixXd expected_reduced =
        reduced_sqrt.diagonal().cwiseAbs2().cwiseInverse().asDiagonal();
    EXPECT_TRUE(covariance.reduced.isApprox(expected_reduced, 1e-12));
    EXPECT_TRUE(covariance.lifted.isApprox(
        basis.tangent * expected_reduced * basis.tangent.transpose(), 1e-12));
    EXPECT_TRUE(covariance.lifted.topRows(3).isZero(1e-12));
    EXPECT_TRUE(covariance.lifted.leftCols(3).isZero(1e-12));
}

TEST(GaugeFixedCovarianceTest, RejectsUnfixedObservableDeficiency) {
    const GaugeFixedBasis basis = buildPositionYawGaugeBasis(10, 0, Eigen::Matrix3d::Identity());
    Eigen::MatrixXd deficient = Eigen::MatrixXd::Zero(5, 10);
    deficient.leftCols(5).setIdentity();

    EXPECT_THROW(covarianceInFixedGauge(deficient, basis), std::invalid_argument);
}

}  // namespace
}  // namespace tassel_core
