#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Householder>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "marg/marg_helper.h"

namespace tassel_core {
namespace {

void marginalizeSquareRootSystemReference(
    size_t marginalized_size, size_t retained_size, Eigen::MatrixXd& jacobian,
    Eigen::VectorXd& residual, Eigen::MatrixXd& prior_jacobian, Eigen::VectorXd& prior_residual) {
    ASSERT_EQ(Eigen::Index(marginalized_size + retained_size), jacobian.cols());
    ASSERT_EQ(jacobian.rows(), residual.rows());

    if (jacobian.rows() == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        return;
    }

    Eigen::Index marginalized_rank = 0;
    Eigen::Index total_rank = 0;
    const Eigen::Index rows = jacobian.rows();
    const Eigen::Index cols = jacobian.cols();
    Eigen::VectorXd temp_vec(cols + 1);
    double* temp_data = temp_vec.data();
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon());

    for (Eigen::Index i = 0; i < cols && total_rank < rows; ++i) {
        const Eigen::Index remaining_rows = rows - total_rank;
        const Eigen::Index remaining_cols = cols - i - 1;

        double beta;
        double h_coeff;
        jacobian.col(i).tail(remaining_rows).makeHouseholderInPlace(h_coeff, beta);
        if (std::abs(beta) > rank_threshold) {
            jacobian.coeffRef(total_rank, i) = beta;
            jacobian.bottomRightCorner(remaining_rows, remaining_cols)
                .applyHouseholderOnTheLeft(
                    jacobian.col(i).tail(remaining_rows - 1), h_coeff, temp_data + i + 1);
            residual.tail(remaining_rows)
                .applyHouseholderOnTheLeft(
                    jacobian.col(i).tail(remaining_rows - 1), h_coeff, temp_data + cols);
            ++total_rank;
        } else {
            jacobian.coeffRef(total_rank, i) = 0;
        }

        jacobian.col(i).tail(remaining_rows - 1).setZero();

        if (i == Eigen::Index(marginalized_size) - 1) {
            marginalized_rank = total_rank;
        }
    }

    const Eigen::Index retained_rank = total_rank - marginalized_rank;
    if (retained_rank == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        jacobian.resize(0, 0);
        residual.resize(0);
        return;
    }

    prior_jacobian =
        jacobian.block(marginalized_rank, marginalized_size, retained_rank, retained_size);
    prior_residual = residual.segment(marginalized_rank, retained_rank);
    jacobian.resize(0, 0);
    residual.resize(0);
}

void expectPriorNear(
    const Eigen::MatrixXd& actual_jacobian, const Eigen::VectorXd& actual_residual,
    const Eigen::MatrixXd& expected_jacobian, const Eigen::VectorXd& expected_residual) {
    ASSERT_EQ(actual_jacobian.rows(), expected_jacobian.rows());
    ASSERT_EQ(actual_jacobian.cols(), expected_jacobian.cols());
    ASSERT_EQ(actual_residual.rows(), expected_residual.rows());

    const double jacobian_scale =
        std::max(1.0, std::max(actual_jacobian.norm(), expected_jacobian.norm()));
    const double residual_scale =
        std::max(1.0, std::max(actual_residual.norm(), expected_residual.norm()));
    EXPECT_LE((actual_jacobian - expected_jacobian).norm(), 1e-12 * jacobian_scale);
    EXPECT_LE((actual_residual - expected_residual).norm(), 1e-12 * residual_scale);
}

void expectMarginalizationMatchesReference(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& residual, int marginalized_size,
    int retained_size) {
    Eigen::MatrixXd actual_input = jacobian;
    Eigen::VectorXd actual_residual_input = residual;
    Eigen::MatrixXd expected_input = jacobian;
    Eigen::VectorXd expected_residual_input = residual;
    Eigen::MatrixXd actual_jacobian;
    Eigen::VectorXd actual_residual;
    Eigen::MatrixXd expected_jacobian;
    Eigen::VectorXd expected_residual;

    MargHelper::marginalizeSquareRootSystem(
        static_cast<size_t>(marginalized_size), static_cast<size_t>(retained_size), actual_input,
        actual_residual_input, actual_jacobian, actual_residual);
    marginalizeSquareRootSystemReference(
        static_cast<size_t>(marginalized_size), static_cast<size_t>(retained_size), expected_input,
        expected_residual_input, expected_jacobian, expected_residual);

    // 调用方依赖输入系统被释放；候选实现也必须保持该所有权合同。
    EXPECT_EQ(actual_input.rows(), 0);
    EXPECT_EQ(actual_input.cols(), 0);
    EXPECT_EQ(actual_residual_input.rows(), 0);
    EXPECT_EQ(expected_input.rows(), 0);
    EXPECT_EQ(expected_input.cols(), 0);
    EXPECT_EQ(expected_residual_input.rows(), 0);
    expectPriorNear(actual_jacobian, actual_residual, expected_jacobian, expected_residual);
}

Eigen::MatrixXd randomJacobian(
    Eigen::Index rows, Eigen::Index cols, std::mt19937& rng,
    std::normal_distribution<double>& normal) {
    Eigen::MatrixXd jacobian(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r) {
        for (Eigen::Index c = 0; c < cols; ++c) {
            jacobian(r, c) = normal(rng);
        }
    }
    return jacobian;
}

Eigen::VectorXd randomResidual(
    Eigen::Index rows, std::mt19937& rng, std::normal_distribution<double>& normal) {
    Eigen::VectorXd residual(rows);
    for (Eigen::Index r = 0; r < rows; ++r) {
        residual(r) = normal(rng);
    }
    return residual;
}

TEST(MargHelperTest, MovesMarginalizedColumnsBeforeRetainedColumns) {
    Eigen::MatrixXd columns(1, 46);
    for (int i = 0; i < columns.cols(); ++i) {
        columns(0, i) = i;
    }

    const Eigen::MatrixXd create_order =
        MargHelper::reorderForMarginalization(columns, RetainedHostAction::InitializeRetainedSlot);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(create_order(0, i), i);
    }
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(create_order(0, i + 15), i + 21);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(create_order(0, i + 24), i + 15);
    }

    const Eigen::MatrixXd keep_order =
        MargHelper::reorderForMarginalization(columns, RetainedHostAction::MarginalizeOldestFrame);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(keep_order(0, i), i + 15);
        EXPECT_EQ(keep_order(0, i + 15), i);
    }

    const Eigen::MatrixXd replace_order =
        MargHelper::reorderForMarginalization(columns, RetainedHostAction::ReplaceRetainedSlot);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(replace_order(0, i), i);
    }
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(replace_order(0, i + 15), i + 21);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(replace_order(0, i + 24), i + 15);
    }

    EXPECT_EQ(create_order(0, 45), 45);
    EXPECT_EQ(keep_order(0, 45), 45);
    EXPECT_EQ(replace_order(0, 45), 45);
}

TEST(MargHelperTest, SquareRootMarginalizationMatchesReferenceQrContract) {
    std::mt19937 rng(7);
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::vector<std::tuple<int, int, int>> cases = {
        {6, 2, 5}, {12, 5, 9}, {35, 15, 31}, {80, 24, 52}};

    for (const auto& [rows, marginalized_size, retained_size] : cases) {
        Eigen::MatrixXd jacobian =
            randomJacobian(rows, marginalized_size + retained_size, rng, normal);
        Eigen::VectorXd residual = randomResidual(rows, rng, normal);

        // 构造确定性的相关列，覆盖 rank-skip 与后续列继续消元的合同。
        if (jacobian.cols() > 4) {
            jacobian.col(2) = 2.0 * jacobian.col(0) - jacobian.col(1);
            jacobian.col(jacobian.cols() - 1) = jacobian.col(1) + 0.5 * jacobian.col(3);
        }

        expectMarginalizationMatchesReference(jacobian, residual, marginalized_size, retained_size);
    }
}

TEST(MargHelperTest, SquareRootMarginalizationMatchesReferenceAcrossPanelBoundaries) {
    std::mt19937 rng(19);
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::vector<std::tuple<int, int, int>> cases = {
        {24, 7, 18}, {30, 8, 23}, {36, 9, 28}, {72, 16, 47}, {96, 24, 65}};

    for (const auto& [rows, marginalized_size, retained_size] : cases) {
        Eigen::MatrixXd jacobian =
            randomJacobian(rows, marginalized_size + retained_size, rng, normal);
        Eigen::VectorXd residual = randomResidual(rows, rng, normal);

        // 这些列位置专门卡 blocked QR 的 panel 边界、边缘化边界和 retained 尾部。
        const Eigen::Index cols = jacobian.cols();
        if (cols > 10) {
            jacobian.col(7) = jacobian.col(0) - 0.25 * jacobian.col(3);
            jacobian.col(8) = -2.0 * jacobian.col(1) + jacobian.col(4);
            jacobian.col(9) = 0.5 * jacobian.col(2) + jacobian.col(5);
        }
        if (marginalized_size + 1 < cols) {
            jacobian.col(marginalized_size) =
                jacobian.col(marginalized_size - 1) + 0.1 * jacobian.col(0);
        }
        if (cols > 4) {
            jacobian.col(cols - 1) = jacobian.col(cols - 2) - jacobian.col(1);
        }

        for (const double scale : {1e-6, 1.0, 1e6}) {
            expectMarginalizationMatchesReference(
                scale * jacobian, scale * residual, marginalized_size, retained_size);
        }
    }
}

}  // namespace
}  // namespace tassel_core
