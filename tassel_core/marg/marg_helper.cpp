#include <Eigen/Householder>

#include "factor/marginalization_prior_factor.h"
#include "marg_helper.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"
#include "tassel_utils/types.h"

#include <cmath>
#include <limits>

namespace tassel_core {

MargLinData MargHelper::transportPriorToCurrentTangent(
    const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double delay_time) {
    const int num_kept = static_cast<int>(prior.linearization_poses.size());
    TASSEL_ASSERT(static_cast<int>(poses.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(speed_bias.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(prior.linearization_speed_bias.size()) == num_kept);
    TASSEL_ASSERT(prior.H.cols() == num_kept * 15 + 1);

    MarginalizationPriorFactor factor(prior);
    std::vector<const double*> parameters;
    parameters.reserve(num_kept * 2 + 1);
    for (int i = 0; i < num_kept; ++i) {
        parameters.push_back(poses[i].data());
        parameters.push_back(speed_bias[i].data());
    }
    parameters.push_back(&delay_time);

    const int rows = factor.num_residuals();
    std::vector<tassel_utils::MatrixRowMajor> jacobian_blocks;
    std::vector<double*> jacobians;
    jacobian_blocks.reserve(num_kept * 2 + 1);
    jacobians.reserve(num_kept * 2 + 1);
    for (int i = 0; i < num_kept; ++i) {
        jacobian_blocks.emplace_back(rows, 6);
        jacobians.push_back(jacobian_blocks.back().data());
        jacobian_blocks.emplace_back(rows, 9);
        jacobians.push_back(jacobian_blocks.back().data());
    }
    jacobian_blocks.emplace_back(rows, 1);
    jacobians.push_back(jacobian_blocks.back().data());

    MargLinData prior_in_current_tangent;
    prior_in_current_tangent.b.resize(rows);
    factor.Evaluate(parameters.data(), prior_in_current_tangent.b.data(), jacobians.data());
    prior_in_current_tangent.H.resize(rows, num_kept * 15 + 1);
    prior_in_current_tangent.linearization_poses = poses;
    prior_in_current_tangent.linearization_speed_bias = speed_bias;
    prior_in_current_tangent.linearization_delay_time = delay_time;

    SE3RightManifold manifold;
    for (int i = 0; i < num_kept; ++i) {
        const auto& pose_jacobian = jacobian_blocks[2 * i];
        const auto& speed_bias_jacobian = jacobian_blocks[2 * i + 1];
        double plus_data[36];
        manifold.PlusJacobian(poses[i].data(), plus_data);
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
        prior_in_current_tangent.H.block(0, i * 15, rows, 6) = pose_jacobian * plus;
        prior_in_current_tangent.H.block(0, i * 15 + 6, rows, 9) = speed_bias_jacobian;
    }
    prior_in_current_tangent.H.col(prior_in_current_tangent.H.cols() - 1) =
        jacobian_blocks.back().col(0);
    return prior_in_current_tangent;
}

void MargHelper::eliminateSquareRootSystem(
    size_t eliminated_size, size_t retained_size, Eigen::MatrixXd& jacobian,
    Eigen::VectorXd& residual, Eigen::MatrixXd& prior_jacobian, Eigen::VectorXd& prior_residual) {
    TASSEL_ASSERT(Eigen::Index(eliminated_size + retained_size) == jacobian.cols());
    TASSEL_ASSERT(jacobian.rows() == residual.rows());

    if (jacobian.rows() == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        return;
    }

    // 丢弃缺秩行，避免将无效约束写入先验。
    Eigen::Index eliminated_rank = 0;
    Eigen::Index total_rank = 0;
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon());

    const Eigen::Index rows = jacobian.rows();
    const Eigen::Index cols = jacobian.cols();

    Eigen::VectorXd temp_vec(cols + 1);
    double* temp_data = temp_vec.data();

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

        if (i == Eigen::Index(eliminated_size) - 1) {
            eliminated_rank = total_rank;
        }
    }

    const Eigen::Index retained_rank = total_rank - eliminated_rank;

    if (retained_rank == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        jacobian.resize(0, 0);
        residual.resize(0);
        return;
    }

    prior_jacobian = jacobian.block(eliminated_rank, eliminated_size, retained_rank, retained_size);
    prior_residual = residual.segment(eliminated_rank, retained_rank);

    jacobian.resize(0, 0);
    residual.resize(0);
}

}  // namespace tassel_core
