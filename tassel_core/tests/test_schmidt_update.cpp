#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <array>
#include <stdexcept>

#include "marg/schmidt/schmidt_update.h"

namespace tassel_core {
namespace {

TEST(SchmidtUpdateTest, KeepsSchmidtMarginalAndUpdatesCrossCovariance) {
    Eigen::MatrixXd paa(2, 2);
    paa << 2.0, 0.2, 0.2, 1.5;
    Eigen::MatrixXd pas(2, 1);
    pas << 0.4, -0.1;
    Eigen::MatrixXd pss(1, 1);
    pss << 0.8;
    Eigen::MatrixXd ha(1, 2);
    ha << 1.0, 0.3;
    Eigen::MatrixXd hs(1, 1);
    hs << 1.2;
    Eigen::MatrixXd r(1, 1);
    r << 0.25;
    Eigen::VectorXd residual(1);
    residual << 0.7;

    const SchmidtUpdateResult result = applySchmidtUpdate(paa, pas, pss, ha, hs, r, residual);

    EXPECT_TRUE(result.active_delta.allFinite());
    EXPECT_TRUE(result.active_covariance.allFinite());
    EXPECT_TRUE(result.active_schmidt_cross_covariance.allFinite());
    EXPECT_TRUE(result.schmidt_covariance.isApprox(pss, 1e-12));
    EXPECT_FALSE(result.active_schmidt_cross_covariance.isApprox(pas, 1e-12));
}

TEST(SchmidtUpdateTest, ActiveMarginalMatchesFullKalmanUpdate) {
    Eigen::MatrixXd paa(2, 2);
    paa << 2.0, 0.2, 0.2, 1.5;
    Eigen::MatrixXd pas(2, 1);
    pas << 0.4, -0.1;
    Eigen::MatrixXd pss(1, 1);
    pss << 0.8;
    Eigen::MatrixXd ha(1, 2);
    ha << 1.0, 0.3;
    Eigen::MatrixXd hs(1, 1);
    hs << 1.2;
    Eigen::MatrixXd r(1, 1);
    r << 0.25;
    Eigen::VectorXd residual(1);
    residual << 0.7;

    const SchmidtUpdateResult result = applySchmidtUpdate(paa, pas, pss, ha, hs, r, residual);

    Eigen::MatrixXd full_covariance(3, 3);
    full_covariance << paa, pas, pas.transpose(), pss;
    Eigen::MatrixXd full_jacobian(1, 3);
    full_jacobian << ha, hs;
    const Eigen::MatrixXd innovation =
        full_jacobian * full_covariance * full_jacobian.transpose() + r;
    const Eigen::MatrixXd full_cross = full_covariance * full_jacobian.transpose();
    const Eigen::MatrixXd full_gain = innovation.ldlt().solve(full_cross.transpose()).transpose();
    const Eigen::MatrixXd full_covariance_updated =
        full_covariance - full_gain * full_cross.transpose();

    EXPECT_TRUE(
        result.active_covariance.isApprox(full_covariance_updated.topLeftCorner(2, 2), 1e-12));
    EXPECT_TRUE(result.active_schmidt_cross_covariance.isApprox(
        full_covariance_updated.topRightCorner(2, 1), 1e-12));
}

TEST(SchmidtUpdateTest, RejectsInconsistentDimensions) {
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(2, 1);
    Eigen::MatrixXd schmidt = Eigen::MatrixXd::Identity(1, 1);
    Eigen::MatrixXd active_jacobian = Eigen::MatrixXd::Zero(1, 2);
    Eigen::MatrixXd schmidt_jacobian = Eigen::MatrixXd::Zero(2, 1);
    Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(1, 1);
    Eigen::VectorXd residual = Eigen::VectorXd::Zero(1);

    EXPECT_THROW(
        applySchmidtUpdate(
            covariance, cross, schmidt, active_jacobian, schmidt_jacobian, noise, residual),
        std::invalid_argument);
}

TEST(SchmidtUpdateTest, ConvertsRankDeficientSquareRootInformation) {
    Eigen::MatrixXd sqrt_information(1, 2);
    sqrt_information << 2.0, 0.0;

    const Eigen::MatrixXd covariance = pseudoCovarianceFromSqrtInformation(sqrt_information);
    ASSERT_EQ(covariance.rows(), 2);
    ASSERT_EQ(covariance.cols(), 2);
    EXPECT_NEAR(covariance(0, 0), 0.25, 1e-12);
    EXPECT_NEAR(covariance(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(covariance(1, 1), 0.0, 1e-12);

    const Eigen::MatrixXd reconstructed = sqrtInformationFromPseudoCovariance(covariance);
    EXPECT_TRUE((reconstructed.transpose() * reconstructed)
                    .isApprox(sqrt_information.transpose() * sqrt_information, 1e-12));
}

TEST(SchmidtUpdateTest, RejectsRankDeficientPriorAsProbabilisticCovariance) {
    Eigen::MatrixXd sqrt_information(1, 2);
    sqrt_information << 2.0, 0.0;

    EXPECT_THROW(covarianceFromFullRankSqrtInformation(sqrt_information), std::invalid_argument);
}

TEST(SchmidtUpdateTest, BuildsConsistentFullRankSqrtPrior) {
    Eigen::MatrixXd prior_information(3, 3);
    prior_information << 2.0, 0.2, 0.1, 0.2, 1.7, -0.15, 0.1, -0.15, 1.2;
    const Eigen::MatrixXd prior_covariance = prior_information.inverse();
    const Eigen::MatrixXd prior_sqrt =
        sqrtInformationFromPositiveDefiniteCovariance(prior_covariance);
    Eigen::VectorXd prior_mean(3);
    prior_mean << 0.2, -0.1, 0.3;
    const Eigen::VectorXd prior_residual = -prior_sqrt * prior_mean;
    Eigen::MatrixXd likelihood(2, 3);
    likelihood << 1.0, 0.3, 0.8, -0.2, 1.1, -0.4;
    Eigen::VectorXd likelihood_residual(2);
    likelihood_residual << -0.5, 0.25;

    const SchmidtSqrtPrior result = buildSchmidtUpdatedSqrtPrior(
        prior_sqrt, prior_residual, likelihood, likelihood_residual, {2});
    const Eigen::MatrixXd result_covariance = covarianceFromFullRankSqrtInformation(result.H);
    const Eigen::VectorXd result_mean = -result_covariance * result.H.transpose() * result.b;

    Eigen::MatrixXd ha = likelihood.leftCols(2);
    Eigen::MatrixXd hs = likelihood.rightCols(1);
    const Eigen::VectorXd innovation = -(likelihood * prior_mean + likelihood_residual);
    const SchmidtUpdateResult expected = applySchmidtUpdate(
        prior_covariance.topLeftCorner(2, 2), prior_covariance.topRightCorner(2, 1),
        prior_covariance.bottomRightCorner(1, 1), ha, hs, Eigen::Matrix2d::Identity(), innovation);

    EXPECT_TRUE(result_mean.head(2).isApprox(prior_mean.head(2) + expected.active_delta, 1e-11));
    EXPECT_NEAR(result_mean[2], prior_mean[2], 1e-11);
    EXPECT_TRUE(result_covariance.topLeftCorner(2, 2).isApprox(expected.active_covariance, 1e-11));
    EXPECT_TRUE(result_covariance.topRightCorner(2, 1).isApprox(
        expected.active_schmidt_cross_covariance, 1e-11));
    EXPECT_TRUE(result_covariance.bottomRightCorner(1, 1).isApprox(
        prior_covariance.bottomRightCorner(1, 1), 1e-11));
}

TEST(SchmidtUpdateTest, UpdatesJointCovarianceWithNonContiguousSchmidtColumns) {
    Eigen::MatrixXd covariance(4, 4);
    covariance << 2.0, 0.2, 0.3, -0.1, 0.2, 1.5, 0.1, 0.25, 0.3, 0.1, 1.2, 0.15, -0.1, 0.25, 0.15,
        1.0;
    Eigen::MatrixXd jacobian(2, 4);
    jacobian << 1.0, 0.4, -0.2, 0.1, 0.3, -0.5, 0.8, 0.6;
    Eigen::Vector2d residual(0.2, -0.4);

    const SchmidtJointUpdateResult result = applySchmidtUpdateToJointCovariance(
        covariance, jacobian, Eigen::Matrix2d::Identity(), residual, {1, 3});

    const std::array<int, 2> active{0, 2};
    const std::array<int, 2> schmidt{1, 3};
    Eigen::Matrix2d paa;
    Eigen::Matrix2d pas;
    Eigen::Matrix2d pss;
    Eigen::Matrix2d ha;
    Eigen::Matrix2d hs;
    for (int i = 0; i < 2; ++i) {
        ha.col(i) = jacobian.col(active[i]);
        hs.col(i) = jacobian.col(schmidt[i]);
        for (int j = 0; j < 2; ++j) {
            paa(i, j) = covariance(active[i], active[j]);
            pas(i, j) = covariance(active[i], schmidt[j]);
            pss(i, j) = covariance(schmidt[i], schmidt[j]);
        }
    }
    const SchmidtUpdateResult expected =
        applySchmidtUpdate(paa, pas, pss, ha, hs, Eigen::Matrix2d::Identity(), residual);

    EXPECT_NEAR(result.delta[1], 0.0, 1e-12);
    EXPECT_NEAR(result.delta[3], 0.0, 1e-12);
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(result.delta[active[i]], expected.active_delta[i], 1e-12);
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(
                result.covariance(schmidt[i], schmidt[j]), covariance(schmidt[i], schmidt[j]),
                1e-12);
            EXPECT_NEAR(
                result.covariance(active[i], schmidt[j]),
                expected.active_schmidt_cross_covariance(i, j), 1e-12);
        }
    }
}

}  // namespace
}  // namespace tassel_core
