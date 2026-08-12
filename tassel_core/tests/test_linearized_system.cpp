#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "solver/linearized_system.h"

namespace tassel_core {
namespace {

TEST(LinearizedSystemTest, SolvesResidualPlusJacobianDeltaConvention) {
    Eigen::Matrix<double, 3, 2> jacobian;
    jacobian << 1.0, 0.0, 0.0, 2.0, 0.0, 0.5;
    Eigen::Vector3d residual(-3.0, -4.0, -1.0);
    const LinearizedSystem system(jacobian, residual);

    const LinearStep step = solveDampedNormalStep(system);

    EXPECT_TRUE(step.delta.isApprox(Eigen::Vector2d(3.0, 2.0), 1e-12));
    EXPECT_NEAR(step.initial_cost, 13.0, 1e-12);
    EXPECT_NEAR(step.model_cost, 0.0, 1e-12);
    EXPECT_NEAR(step.predicted_reduction, 13.0, 1e-12);
}

TEST(LinearizedSystemTest, AppliesBasaltStyleRelativeDiagonalDamping) {
    Eigen::Matrix2d jacobian = Eigen::Vector2d(2.0, 1.0).asDiagonal();
    Eigen::Vector2d residual(-2.0, -1.0);
    const LinearizedSystem system(jacobian, residual);

    const LinearStep step = solveDampedNormalStep(system, {0.5, 0.1});

    EXPECT_TRUE(step.damping_diagonal.isApprox(Eigen::Vector2d(2.0, 0.5), 1e-12));
    EXPECT_TRUE(step.delta.isApprox(Eigen::Vector2d(2.0 / 3.0, 2.0 / 3.0), 1e-12));
    const double reduction_identity =
        0.5 * step.delta.dot(step.damping_diagonal.cwiseProduct(step.delta) - system.gradient());
    EXPECT_NEAR(step.predicted_reduction, reduction_identity, 1e-12);
}

TEST(LinearizedSystemTest, RelativeDampingStepIsInvariantToResidualScale) {
    Eigen::Matrix<double, 3, 2> jacobian;
    jacobian << 1.0, 2.0, -2.0, 1.0, 0.5, -1.0;
    Eigen::Vector3d residual(0.4, -1.2, 2.0);
    const LinearStep reference =
        solveDampedNormalStep(LinearizedSystem(jacobian, residual), {0.2, 0.0});

    for (double scale : {1e-6, 1e6}) {
        const LinearStep scaled =
            solveDampedNormalStep(LinearizedSystem(scale * jacobian, scale * residual), {0.2, 0.0});
        EXPECT_TRUE(scaled.delta.isApprox(reference.delta, 1e-10));
    }
}

TEST(LinearizedSystemTest, RejectsRankDeficiencyUnlessDampingMakesSystemDefinite) {
    Eigen::Matrix2d jacobian;
    jacobian << 1.0, 1.0, 2.0, 2.0;
    const LinearizedSystem system(jacobian, Eigen::Vector2d(1.0, -1.0));

    EXPECT_THROW(solveDampedNormalStep(system), std::runtime_error);
    EXPECT_NO_THROW(solveDampedNormalStep(system, {0.1, 0.0}));
}

TEST(LinearizedSystemTest, RejectsInvalidDimensionsAndNumbers) {
    EXPECT_THROW(
        LinearizedSystem(Eigen::MatrixXd::Zero(2, 2), Eigen::VectorXd::Zero(3)),
        std::invalid_argument);
    Eigen::MatrixXd non_finite = Eigen::MatrixXd::Zero(1, 1);
    non_finite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(LinearizedSystem(non_finite, Eigen::VectorXd::Zero(1)), std::invalid_argument);

    const LinearizedSystem system(Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Ones(2));
    EXPECT_THROW(solveDampedNormalStep(system, {-1.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(system.costAt(Eigen::VectorXd::Zero(3)), std::invalid_argument);
}

}  // namespace
}  // namespace tassel_core
