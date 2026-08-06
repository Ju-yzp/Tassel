#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "estimator/ba_update_evidence.h"

namespace tassel_core {
namespace {

TEST(BaUpdateEvidenceTest, RemovesNuisanceExplanationAndRecoversBiasIncrement) {
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(7, 5);
    jacobian.block<4, 2>(0, 0) << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0;
    jacobian.block<4, 3>(0, 2) << 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0, 3.0, 0.0,
        1.0;
    jacobian.block<3, 3>(4, 2) = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d expected_increment(0.2, -0.1, 0.05);
    Eigen::VectorXd residual = -jacobian.middleCols<3>(2) * expected_increment;
    residual.head<4>() += jacobian.leftCols<2>() * Eigen::Vector2d(0.4, -0.3);

    const BaUpdateEvidence evidence = computeBaUpdateEvidence(jacobian, residual, {2, 3, 4});

    EXPECT_EQ(evidence.rank, 3);
    EXPECT_TRUE(evidence.increment.isApprox(expected_increment, 1e-10));
    EXPECT_TRUE(evidence.information.isApprox(evidence.information.transpose(), 1e-12));
    EXPECT_GT(evidence.cost_reduction, 0.0);
    EXPECT_TRUE(std::isfinite(evidence.condition));
}

TEST(BaUpdateEvidenceTest, ReportsUnobservableDirectionsWithoutInventingInformation) {
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 4);
    jacobian.col(0) << 1.0, 0.0, 0.0;
    jacobian.col(1) << 1.0, 0.0, 0.0;
    jacobian.col(2) << 0.0, 1.0, 0.0;
    const Eigen::VectorXd residual = Eigen::Vector3d(0.2, -0.3, 0.0);

    const BaUpdateEvidence evidence = computeBaUpdateEvidence(jacobian, residual, {1, 2, 3});

    EXPECT_EQ(evidence.rank, 1);
    EXPECT_TRUE(std::isinf(evidence.condition));
    EXPECT_NEAR(evidence.information(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(evidence.information(2, 2), 0.0, 1e-12);
}

TEST(BaUpdateEvidenceTest, RejectsInvalidBiasColumnMapping) {
    const Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(4, 4);
    const Eigen::VectorXd residual = Eigen::VectorXd::Zero(4);
    EXPECT_THROW(computeBaUpdateEvidence(jacobian, residual, {1, 1, 3}), std::invalid_argument);
}

TEST(BaUpdateEvidenceTest, SupportsCommonBiasWithDifferentialBiasAsNuisance) {
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(8, 7);
    jacobian.block<4, 3>(0, 1) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0;
    jacobian.block<4, 3>(0, 4) << 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, -0.5,
        0.5, 0.0;
    jacobian.block<4, 3>(4, 1) = Eigen::Matrix3d::Identity().replicate<2, 1>().topRows<4>();
    jacobian.block<4, 3>(4, 4) = -jacobian.block<4, 3>(4, 1);

    Eigen::MatrixXd candidate_basis = Eigen::MatrixXd::Zero(7, 3);
    candidate_basis.block<3, 3>(1, 0) = Eigen::Matrix3d::Identity();
    candidate_basis.block<3, 3>(4, 0) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd nuisance_basis = Eigen::MatrixXd::Zero(7, 4);
    nuisance_basis(0, 0) = 1.0;
    nuisance_basis.block<3, 3>(1, 1) = Eigen::Matrix3d::Identity();
    nuisance_basis.block<3, 3>(4, 1) = -Eigen::Matrix3d::Identity();
    const Eigen::Vector3d expected_increment(0.1, -0.2, 0.3);
    const Eigen::VectorXd residual = -(jacobian * candidate_basis) * expected_increment;

    const BaUpdateEvidence evidence =
        computeBaUpdateEvidence(jacobian, residual, candidate_basis, nuisance_basis);

    EXPECT_EQ(evidence.rank, 3);
    EXPECT_TRUE(evidence.increment.isApprox(expected_increment, 1e-10));
}

}  // namespace
}  // namespace tassel_core
