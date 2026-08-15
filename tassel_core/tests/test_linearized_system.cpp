#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "solver/linearized_system.h"
#include "solver/variable_layout.h"

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

TEST(VariableLayoutTest, PreservesDeclaredJointLayoutAndSeparatesRoles) {
    const VariableKey pose{20, VariableKind::Pose};
    const VariableKey ba{20, VariableKind::AccelBias};
    const VariableKey velocity{10, VariableKind::Velocity};
    const VariableLayout layout({
        {pose, 6, VariableRole::Active},
        {ba, 3, VariableRole::Schmidt},
        {velocity, 3, VariableRole::Fixed},
    });

    EXPECT_EQ(layout.totalSize(), 12);
    EXPECT_EQ(layout.block(pose).offset, 0);
    EXPECT_EQ(layout.block(ba).offset, 6);
    EXPECT_EQ(layout.block(velocity).offset, 9);
    EXPECT_EQ(layout.columns(VariableRole::Active), (std::vector<int>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(layout.columns(VariableRole::Schmidt), (std::vector<int>{6, 7, 8}));
    EXPECT_EQ(layout.columns(VariableRole::Fixed), (std::vector<int>{9, 10, 11}));
}

TEST(VariableLayoutTest, RejectsInvalidContracts) {
    const VariableKey pose{1, VariableKind::Pose};
    EXPECT_THROW(VariableLayout({{pose, 0, VariableRole::Active}}), std::invalid_argument);
    EXPECT_THROW(
        VariableLayout({{pose, 6, VariableRole::Active}, {pose, 6, VariableRole::Schmidt}}),
        std::invalid_argument);

    const VariableLayout layout({{pose, 6, VariableRole::Active}});
    EXPECT_THROW(layout.block({2, VariableKind::Pose}), std::out_of_range);
}

}  // namespace
}  // namespace tassel_core
