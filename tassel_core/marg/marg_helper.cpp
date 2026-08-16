#include <Eigen/Householder>

#include <cmath>
#include <limits>
#include <stdexcept>
#include "factor/marginalization_prior_factor.h"
#include "marg/marg_helper.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

namespace {

std::vector<int> marginalizationColumnOrder(int window_state_count, RetainedHostAction action) {
    if (window_state_count < 2) {
        throw std::invalid_argument("Marginalization requires at least two window states");
    }
    constexpr int pose_size = MargHelper::kPoseSize;
    constexpr int speed_bias_size = MargHelper::kSpeedBiasSize;
    constexpr int state_size = MargHelper::kFullStateSize;
    const int total_columns = window_state_count * state_size + 1;
    std::vector<int> order;
    order.reserve(static_cast<size_t>(total_columns));
    const auto append = [&order](int first, int count) {
        for (int i = 0; i < count; ++i) {
            order.push_back(first + i);
        }
    };

    switch (action) {
        case RetainedHostAction::InitializeRetainedSlot:
        case RetainedHostAction::ReplaceRetainedSlot:
            append(0, state_size);
            append(state_size + pose_size, speed_bias_size);
            append(state_size, pose_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
        case RetainedHostAction::MarginalizeOldestFrame:
            append(state_size, state_size);
            append(0, state_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
    }
    if (static_cast<int>(order.size()) != total_columns) {
        throw std::logic_error("Marginalization column mapping is incomplete");
    }
    return order;
}

}  // namespace

Eigen::VectorXd MargHelper::evaluatePriorResidual(
    const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double time_delay) {
    const int num_kept = static_cast<int>(prior.linearization_poses.size());
    TASSEL_ASSERT(static_cast<int>(poses.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(speed_bias.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(prior.linearization_speed_bias.size()) == num_kept);
    prior.validate();

    MarginalizationPriorFactor factor(prior);
    std::vector<const double*> parameters;
    parameters.reserve(num_kept * 2 + 1);
    for (int i = 0; i < num_kept; ++i) {
        parameters.push_back(poses[i].data());
        if (i > 0) {
            parameters.push_back(speed_bias[i].data());
        }
    }
    parameters.push_back(&time_delay);

    Eigen::VectorXd residual(factor.num_residuals());
    TASSEL_ASSERT(factor.Evaluate(parameters.data(), residual.data(), nullptr));
    return residual;
}

void MargHelper::marginalizeSquareRootSystem(
    size_t marginalized_size, size_t retained_size, Eigen::MatrixXd& jacobian,
    Eigen::VectorXd& residual, Eigen::MatrixXd& prior_jacobian, Eigen::VectorXd& prior_residual) {
    TASSEL_ASSERT(Eigen::Index(marginalized_size + retained_size) == jacobian.cols());
    TASSEL_ASSERT(jacobian.rows() == residual.rows());

    if (jacobian.rows() == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        return;
    }

    // 丢弃缺秩行，避免将无效约束写入先验。
    Eigen::Index marginalized_rank = 0;
    Eigen::Index total_rank = 0;
    const Eigen::Index rows = jacobian.rows();
    const Eigen::Index cols = jacobian.cols();

    Eigen::VectorXd temp_vec(cols + 1);
    double* temp_data = temp_vec.data();
    // Householder beta 与输入同尺度，秩阈值必须随整个平方根系统缩放。
    const double system_scale = jacobian.cwiseAbs().maxCoeff();
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon()) * system_scale;
    for (Eigen::Index i = 0; i < cols && total_rank < rows; ++i) {
        Eigen::Index remainingRows = rows - total_rank;
        Eigen::Index remainingCols = cols - i - 1;

        double beta;
        double hCoeff;
        jacobian.col(i).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
        if (std::abs(beta) > rank_threshold) {
            jacobian.coeffRef(total_rank, i) = beta;

            jacobian.bottomRightCorner(remainingRows, remainingCols)
                .applyHouseholderOnTheLeft(
                    jacobian.col(i).tail(remainingRows - 1), hCoeff, temp_data + i + 1);
            residual.tail(remainingRows)
                .applyHouseholderOnTheLeft(
                    jacobian.col(i).tail(remainingRows - 1), hCoeff, temp_data + cols);
            total_rank++;
        } else {
            jacobian.coeffRef(total_rank, i) = 0;
        }

        jacobian.col(i).tail(remainingRows - 1).setZero();

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

Eigen::MatrixXd MargHelper::reorderForMarginalization(
    const Eigen::MatrixXd& jacobian, RetainedHostAction action) {
    TASSEL_ASSERT(jacobian.cols() >= 2 * kFullStateSize + 1);
    TASSEL_ASSERT((jacobian.cols() - 1) % kFullStateSize == 0);
    const int window_state_count = (static_cast<int>(jacobian.cols()) - 1) / kFullStateSize;
    const std::vector<int> order = marginalizationColumnOrder(window_state_count, action);
    Eigen::MatrixXd reordered(jacobian.rows(), jacobian.cols());
    for (int destination = 0; destination < static_cast<int>(order.size()); ++destination) {
        reordered.col(destination) = jacobian.col(order[static_cast<size_t>(destination)]);
    }
    return reordered;
}

}  // namespace tassel_core
