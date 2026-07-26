#include <Eigen/Householder>

#include <limits>
#include "factor/marginalization_prior_factor.h"
#include "marg_helper.h"
#include "tassel_utils/macros.h"

#include <sophus/so3.hpp>

namespace tassel_core {

Eigen::VectorXd MargHelper::evaluatePriorResidual(
    const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double delay_time) {
    const int num_kept = static_cast<int>(prior.linearization_poses.size());
    TASSEL_ASSERT(static_cast<int>(poses.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(speed_bias.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(prior.linearization_speed_bias.size()) == num_kept);
    const int full_cols = num_kept * kFullStateSize + 1;
    const int mixed_cols = kPoseSize + (num_kept - 1) * kFullStateSize + 1;
    TASSEL_ASSERT(prior.H.cols() == full_cols || prior.H.cols() == mixed_cols);

    MarginalizationPriorFactor factor(prior);
    std::vector<const double*> parameters;
    parameters.reserve(num_kept * 2 + 1);
    const bool pose_only_host = prior.H.cols() == mixed_cols;
    parameters.push_back(poses[0].data());
    if (!pose_only_host) {
        parameters.push_back(speed_bias[0].data());
    }
    for (int i = 1; i < num_kept; ++i) {
        parameters.push_back(poses[i].data());
        parameters.push_back(speed_bias[i].data());
    }
    parameters.push_back(&delay_time);

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
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon());
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

    // Keep/Replace 输入从保留宿主开始；Create 输入为 [空保留槽, 首帧状态, 后续状态]。
    // 平方根边缘化要求待边缘化列位于保留列之前。
    const auto host_pose = jacobian.leftCols(kPoseSize);
    const auto host_motion = jacobian.middleCols(kPoseSize, kSpeedBiasSize);
    const auto next_state = jacobian.middleCols(kFullStateSize, kFullStateSize);
    const auto trailing = jacobian.rightCols(jacobian.cols() - 2 * kFullStateSize);
    Eigen::MatrixXd reordered(jacobian.rows(), jacobian.cols());

    switch (action) {
        case RetainedHostAction::Create:
            // 首次建立先验：消去空槽和首帧运动状态，保留首帧位姿。
            reordered << jacobian.leftCols(kFullStateSize),
                jacobian.middleCols(kFullStateSize + kPoseSize, kSpeedBiasSize),
                jacobian.middleCols(kFullStateSize, kPoseSize),
                jacobian.rightCols(jacobian.cols() - 2 * kFullStateSize);
            break;
        case RetainedHostAction::Keep:
            // 继续使用当前宿主：边缘化下一帧状态，保留完整宿主状态布局。
            reordered << next_state, host_pose, host_motion, trailing;
            break;
        case RetainedHostAction::Replace:
            // 替换宿主：边缘化旧宿主和新宿主运动，仅保留新宿主位姿。
            reordered << host_pose, host_motion, next_state.rightCols(kSpeedBiasSize),
                next_state.leftCols(kPoseSize), trailing;
            break;
    }
    return reordered;
}

}  // namespace tassel_core
