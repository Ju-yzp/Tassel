#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "marg/imu_covariance_propagation.h"

namespace tassel_core {
namespace {

TEST(ImuCovariancePropagationTest, RecoversTransitionFromWhitenedFactor) {
    Eigen::Matrix<double, 15, 15> expected_f = Eigen::Matrix<double, 15, 15>::Identity();
    expected_f.block<3, 3>(0, 6) = 0.1 * Eigen::Matrix3d::Identity();
    expected_f.block<3, 3>(6, 9) = -0.03 * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 15, 15> expected_q = Eigen::Matrix<double, 15, 15>::Identity();
    expected_q.diagonal().array() += Eigen::Array<double, 15, 1>::LinSpaced(15, 0.1, 1.5);
    Eigen::LLT<Eigen::Matrix<double, 15, 15>> q_factor(expected_q);
    const Eigen::Matrix<double, 15, 15> whitening =
        q_factor.matrixL().solve(Eigen::Matrix<double, 15, 15>::Identity());
    Eigen::Matrix<double, 15, 1> expected_offset;
    expected_offset.setLinSpaced(-0.2, 0.3);
    const Eigen::Matrix<double, 15, 15> jacobian_i = -whitening * expected_f;
    const Eigen::Matrix<double, 15, 15> jacobian_j = whitening;
    const Eigen::Matrix<double, 15, 1> residual = -whitening * expected_offset;

    const ImuCovariancePropagation propagation =
        buildImuCovariancePropagation(jacobian_i, jacobian_j, residual);

    EXPECT_TRUE(propagation.F.isApprox(expected_f, 1e-12));
    EXPECT_TRUE(propagation.Q.isApprox(expected_q, 1e-12));
    EXPECT_TRUE(propagation.offset.isApprox(expected_offset, 1e-12));
}

TEST(ImuCovariancePropagationTest, AppendsEveryHistoricalCrossCovariance) {
    constexpr int prior_size = 20;
    constexpr int parent_column = 3;
    Eigen::Matrix<double, prior_size, prior_size> seed =
        Eigen::Matrix<double, prior_size, prior_size>::Random();
    const Eigen::MatrixXd prior =
        seed * seed.transpose() + 0.5 * Eigen::MatrixXd::Identity(prior_size, prior_size);
    ImuCovariancePropagation propagation;
    propagation.F = Eigen::Matrix<double, 15, 15>::Identity();
    propagation.F.block<3, 3>(0, 6) = 0.2 * Eigen::Matrix3d::Identity();
    propagation.Q = 0.1 * Eigen::Matrix<double, 15, 15>::Identity();
    propagation.offset.setZero();

    const Eigen::MatrixXd result = appendImuPropagatedCovariance(prior, parent_column, propagation);
    const Eigen::MatrixXd expected_cross =
        prior.middleCols(parent_column, 15) * propagation.F.transpose();
    const Eigen::MatrixXd expected_child = propagation.F *
                                               prior.block(parent_column, parent_column, 15, 15) *
                                               propagation.F.transpose() +
                                           propagation.Q;

    ASSERT_EQ(result.rows(), prior_size + 15);
    EXPECT_TRUE(result.topLeftCorner(prior_size, prior_size).isApprox(prior, 1e-12));
    EXPECT_TRUE(result.topRightCorner(prior_size, 15).isApprox(expected_cross, 1e-12));
    EXPECT_TRUE(result.bottomRightCorner(15, 15).isApprox(expected_child, 1e-12));
    Eigen::LLT<Eigen::MatrixXd> factor(result);
    EXPECT_EQ(factor.info(), Eigen::Success);
}

TEST(ImuCovariancePropagationTest, RejectsSingularChildJacobian) {
    Eigen::Matrix<double, 15, 15> jacobian_i = Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix<double, 15, 15> jacobian_j = Eigen::Matrix<double, 15, 15>::Identity();
    jacobian_j.row(4).setZero();

    EXPECT_THROW(
        buildImuCovariancePropagation(jacobian_i, jacobian_j, Eigen::Matrix<double, 15, 1>::Zero()),
        std::invalid_argument);
}

TEST(ImuCovariancePropagationTest, InsertsChildBeforeTrailingDelayVariable) {
    constexpr int prior_size = 17;
    constexpr int parent_column = 1;
    constexpr int delay_column = 16;
    Eigen::MatrixXd prior = Eigen::MatrixXd::Identity(prior_size, prior_size);
    prior(parent_column, delay_column) = 0.2;
    prior(delay_column, parent_column) = 0.2;
    ImuCovariancePropagation propagation;
    propagation.F = Eigen::Matrix<double, 15, 15>::Identity();
    propagation.Q = 0.1 * Eigen::Matrix<double, 15, 15>::Identity();
    propagation.offset.setZero();

    const Eigen::MatrixXd result =
        insertImuPropagatedCovariance(prior, parent_column, delay_column, propagation);

    ASSERT_EQ(result.rows(), prior_size + 15);
    EXPECT_NEAR(
        result(delay_column + 15, delay_column + 15), prior(delay_column, delay_column), 1e-12);
    EXPECT_NEAR(result(delay_column + 15, delay_column), 0.2, 1e-12);
    EXPECT_TRUE(result.block(delay_column, delay_column, 15, 15)
                    .isApprox(1.1 * Eigen::Matrix<double, 15, 15>::Identity(), 1e-12));
}

}  // namespace
}  // namespace tassel_core
