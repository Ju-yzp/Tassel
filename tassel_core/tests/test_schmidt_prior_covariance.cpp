#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>

#include "marg/gauge_fixed_covariance.h"
#include "marg/schmidt/schmidt_prior_covariance.h"
#include "marg/state_layout.h"

namespace tassel_core {
namespace {

TEST(SchmidtPriorCovarianceTest, FixesGaugeAndPropagatesMixedLayoutBeforeDelay) {
    constexpr int old_state_count = 2;
    constexpr int old_columns = 6 + 15 + 1;
    MargLinData prior;
    prior.linearization_poses.resize(old_state_count);
    prior.linearization_speed_bias.resize(old_state_count);
    const GaugeFixedBasis gauge =
        buildPositionYawGaugeBasis(old_columns, 0, Eigen::Matrix3d::Identity());
    Eigen::MatrixXd reduced_sqrt = Eigen::MatrixXd::Identity(old_columns - 4, old_columns - 4);
    reduced_sqrt.diagonal().array() += Eigen::ArrayXd::LinSpaced(old_columns - 4, 0.1, 1.8);
    prior.H = reduced_sqrt * gauge.tangent.transpose();
    prior.b = Eigen::VectorXd::Zero(prior.H.rows());

    const SchmidtPriorCovariance fixed =
        buildGaugeFixedPriorCovariance(prior, Eigen::Matrix3d::Identity());
    ImuCovariancePropagation propagation;
    propagation.F = Eigen::Matrix<double, 15, 15>::Identity();
    propagation.Q = 0.05 * Eigen::Matrix<double, 15, 15>::Identity();
    propagation.offset.setZero();
    const SchmidtPriorCovariance propagated =
        propagatePriorCovarianceWithImu(fixed, 1, propagation);

    const PriorStateLayout new_layout(3, static_cast<int>(propagated.covariance.cols()));
    EXPECT_EQ(new_layout.kind(), PriorLayoutKind::PoseOnlyHost);
    EXPECT_EQ(new_layout.delayColumn(), 36);
    EXPECT_EQ(new_layout.poseColumn(2), 21);
    EXPECT_EQ(new_layout.baColumns(2), (std::array<int, 3>{30, 31, 32}));
    const std::vector<std::array<int, 3>> ba_columns = priorBaColumns(propagated);
    ASSERT_EQ(ba_columns.size(), 2);
    EXPECT_EQ(ba_columns[0], (std::array<int, 3>{15, 16, 17}));
    EXPECT_EQ(ba_columns[1], (std::array<int, 3>{30, 31, 32}));
    EXPECT_TRUE(propagated.covariance.topRows(3).isZero(1e-12));
    EXPECT_TRUE(propagated.covariance.leftCols(3).isZero(1e-12));
}

TEST(SchmidtPriorCovarianceTest, RejectsUnobservablePriorBeyondGauge) {
    MargLinData prior;
    prior.linearization_poses.resize(2);
    prior.linearization_speed_bias.resize(2);
    prior.H = Eigen::MatrixXd::Zero(4, 22);
    prior.b = Eigen::VectorXd::Zero(4);

    EXPECT_THROW(
        buildGaugeFixedPriorCovariance(prior, Eigen::Matrix3d::Identity()), std::invalid_argument);
    EXPECT_FALSE(priorHasFiniteGaugeFixedCovariance(prior, Eigen::Matrix3d::Identity()));
}

TEST(SchmidtPriorCovarianceTest, RecenterUsesInverseRightTangentMap) {
    constexpr int state_count = 2;
    constexpr int column_count = 22;
    MargLinData prior;
    prior.linearization_poses.resize(state_count);
    prior.linearization_speed_bias.resize(state_count);
    prior.linearization_poses[0][3] = 0.2;
    prior.linearization_poses[1][4] = -0.3;
    const GaugeFixedBasis gauge = buildPositionYawGaugeBasis(
        column_count, 0, Sophus::SO3d::exp(Eigen::Vector3d(0.2, 0, 0)).matrix());
    prior.H =
        Eigen::MatrixXd::Identity(column_count - 4, column_count - 4) * gauge.tangent.transpose();
    prior.b = Eigen::VectorXd::Zero(prior.H.rows());
    ASSERT_TRUE(priorHasFiniteGaugeFixedCovariance(
        prior, Sophus::SO3d::exp(Eigen::Vector3d(0.2, 0, 0)).matrix()));
    SchmidtPriorCovariance covariance = buildGaugeFixedPriorCovariance(
        prior, Sophus::SO3d::exp(Eigen::Vector3d(0.2, 0, 0)).matrix());
    const Eigen::MatrixXd old_covariance = covariance.covariance;

    std::vector<std::array<double, 6>> new_poses = prior.linearization_poses;
    const Eigen::Vector3d delta_0(0.08, -0.03, 0.04);
    const Eigen::Vector3d delta_1(-0.02, 0.07, -0.05);
    const Eigen::Vector3d new_phi_0 =
        (Sophus::SO3d::exp(Eigen::Vector3d(0.2, 0, 0)) * Sophus::SO3d::exp(delta_0)).log();
    const Eigen::Vector3d new_phi_1 =
        (Sophus::SO3d::exp(Eigen::Vector3d(0, -0.3, 0)) * Sophus::SO3d::exp(delta_1)).log();
    for (int d = 0; d < 3; ++d) {
        new_poses[0][3 + d] = new_phi_0[d];
        new_poses[1][3 + d] = new_phi_1[d];
    }
    Eigen::MatrixXd inverse_map = Eigen::MatrixXd::Identity(column_count, column_count);
    inverse_map.block<3, 3>(3, 3) = Sophus::SO3d::leftJacobianInverse(-delta_0).inverse();
    inverse_map.block<3, 3>(9, 9) = Sophus::SO3d::leftJacobianInverse(-delta_1).inverse();

    recenterPriorCovariance(covariance, prior, new_poses);

    EXPECT_TRUE(covariance.covariance.isApprox(
        inverse_map * old_covariance * inverse_map.transpose(), 1e-12));
}

TEST(SchmidtPriorCovarianceTest, GaugeRotatesOnlyWorldPositionAndVelocityCoordinates) {
    constexpr int state_count = 2;
    constexpr int column_count = 22;
    SchmidtPriorCovariance covariance;
    covariance.state_count = state_count;
    Eigen::MatrixXd seed = Eigen::MatrixXd::Identity(column_count, column_count);
    seed(0, 12) = 0.4;
    seed(12, 0) = -0.2;
    seed(6, 15) = 0.3;
    covariance.covariance = seed * seed.transpose();
    const Eigen::MatrixXd old_covariance = covariance.covariance;
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::MatrixXd gauge_map = Eigen::MatrixXd::Identity(column_count, column_count);
    gauge_map.block<3, 3>(0, 0) = rotation;
    gauge_map.block<3, 3>(6, 6) = rotation;
    gauge_map.block<3, 3>(12, 12) = rotation;

    transformPriorCovarianceGauge(covariance, rotation);

    EXPECT_TRUE(
        covariance.covariance.isApprox(gauge_map * old_covariance * gauge_map.transpose(), 1e-12));
    EXPECT_TRUE((covariance.covariance.block<3, 3>(15, 15).isApprox(
        old_covariance.block<3, 3>(15, 15), 1e-12)));
    EXPECT_TRUE((covariance.covariance.block<3, 3>(18, 18).isApprox(
        old_covariance.block<3, 3>(18, 18), 1e-12)));
}

TEST(SchmidtPriorCovarianceTest, RetainsColumnsForBothHostActions) {
    SchmidtPriorCovariance current;
    current.state_count = 3;
    current.covariance = Eigen::MatrixXd::Zero(37, 37);
    current.covariance.diagonal() = Eigen::VectorXd::LinSpaced(37, 1.0, 37.0);

    const SchmidtPriorCovariance keep_host =
        retainMarginalizedPriorCovariance(current, RetainedHostAction::MarginalizeOldestFrame);
    const SchmidtPriorCovariance replace_host =
        retainMarginalizedPriorCovariance(current, RetainedHostAction::ReplaceRetainedSlot);

    ASSERT_EQ(keep_host.covariance.rows(), 22);
    ASSERT_EQ(replace_host.covariance.rows(), 22);
    EXPECT_DOUBLE_EQ(keep_host.covariance(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(replace_host.covariance(0, 0), 7.0);
    EXPECT_DOUBLE_EQ(keep_host.covariance(6, 6), 22.0);
    EXPECT_DOUBLE_EQ(replace_host.covariance(6, 6), 22.0);
    EXPECT_DOUBLE_EQ(keep_host.covariance(21, 21), 37.0);
}

TEST(SchmidtPriorCovarianceTest, PropagatesThenKeepsBaMarginalDuringVisualUpdate) {
    constexpr int old_state_count = 2;
    constexpr int old_columns = 22;
    constexpr int window_state_count = 3;
    MargLinData prior;
    prior.linearization_poses.resize(old_state_count);
    prior.linearization_speed_bias.resize(old_state_count);
    const GaugeFixedBasis gauge =
        buildPositionYawGaugeBasis(old_columns, 0, Eigen::Matrix3d::Identity());
    prior.H =
        Eigen::MatrixXd::Identity(old_columns - 4, old_columns - 4) * gauge.tangent.transpose();
    prior.b = Eigen::VectorXd::Zero(prior.H.rows());
    const SchmidtPriorCovariance fixed =
        buildGaugeFixedPriorCovariance(prior, Eigen::Matrix3d::Identity());
    ImuCovariancePropagation propagation;
    propagation.F = Eigen::Matrix<double, 15, 15>::Identity();
    propagation.Q = 0.1 * Eigen::Matrix<double, 15, 15>::Identity();
    propagation.offset.setZero();
    const SchmidtPriorCovariance predicted = propagatePriorCovarianceWithImu(fixed, 1, propagation);
    Eigen::MatrixXd visual_jacobian = Eigen::MatrixXd::Zero(3, window_state_count * 15 + 1);
    visual_jacobian.block<3, 3>(0, 15).setIdentity();
    visual_jacobian.block<3, 3>(0, 30) = -Eigen::Matrix3d::Identity();
    visual_jacobian.block<3, 3>(0, 39) = 0.4 * Eigen::Matrix3d::Identity();
    const SchmidtPriorCovariance updated = propagateAndUpdateSchmidtPrior(
        fixed, propagation, visual_jacobian, Eigen::Vector3d(0.2, -0.1, 0.3), window_state_count,
        RetainedHostAction::MarginalizeOldestFrame);

    const PriorStateLayout predicted_layout(3, static_cast<int>(predicted.covariance.cols()));
    const std::array<int, 3> predicted_latest_ba = predicted_layout.baColumns(2);
    const PriorStateLayout updated_layout(2, static_cast<int>(updated.covariance.cols()));
    const std::array<int, 3> updated_ba = updated_layout.baColumns(1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(
                updated.covariance(updated_ba[i], updated_ba[j]),
                predicted.covariance(predicted_latest_ba[i], predicted_latest_ba[j]), 1e-12);
        }
    }
}

}  // namespace
}  // namespace tassel_core
