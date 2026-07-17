#include <Eigen/Householder>

#include "factor/marginalization_prior_factor.h"
#include "marg_helper.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

#include <cmath>
#include <limits>

namespace tassel_core {

MargLinData MargHelper::rebasePrior(
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
    std::vector<std::vector<double>> jacobian_storage;
    std::vector<double*> jacobians;
    jacobian_storage.reserve(num_kept * 2 + 1);
    jacobians.reserve(num_kept * 2 + 1);
    for (int i = 0; i < num_kept; ++i) {
        jacobian_storage.emplace_back(rows * 6);
        jacobians.push_back(jacobian_storage.back().data());
        jacobian_storage.emplace_back(rows * 9);
        jacobians.push_back(jacobian_storage.back().data());
    }
    jacobian_storage.emplace_back(rows);
    jacobians.push_back(jacobian_storage.back().data());

    MargLinData rebased;
    rebased.b.resize(rows);
    factor.Evaluate(parameters.data(), rebased.b.data(), jacobians.data());
    rebased.H.resize(rows, num_kept * 15 + 1);
    rebased.linearization_poses = poses;
    rebased.linearization_speed_bias = speed_bias;
    rebased.linearization_delay_time = delay_time;

    SE3RightManifold manifold;
    for (int i = 0; i < num_kept; ++i) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> pose_jacobian(
            jacobians[2 * i], rows, 6);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> sb_jacobian(
            jacobians[2 * i + 1], rows, 9);
        double plus_data[36];
        manifold.PlusJacobian(poses[i].data(), plus_data);
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
        rebased.H.block(0, i * 15, rows, 6) = pose_jacobian * plus;
        rebased.H.block(0, i * 15 + 6, rows, 9) = sb_jacobian;
    }
    Eigen::Map<Eigen::VectorXd> delay_jacobian(jacobians.back(), rows);
    rebased.H.col(rebased.H.cols() - 1) = delay_jacobian;
    return rebased;
}

void MargHelper::marginalizeSqrtToSqrt(
    size_t marg_size, size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r,
    Eigen::MatrixXd& marg_sqrt_H, Eigen::VectorXd& marg_sqrt_b) {
    TASSEL_ASSERT(Eigen::Index(marg_size + keep_size) == Q2Jp.cols());
    TASSEL_ASSERT(Q2Jp.rows() == Q2r.rows());

    if (Q2Jp.rows() == 0) {
        marg_sqrt_H.resize(0, static_cast<Eigen::Index>(keep_size));
        marg_sqrt_b.resize(0);
        return;
    }

    // 只保留强约束，弱约束因缺秩自动丢弃
    Eigen::Index marg_rank = 0;
    Eigen::Index total_rank = 0;
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon());

    const Eigen::Index rows = Q2Jp.rows();
    const Eigen::Index cols = Q2Jp.cols();

    Eigen::VectorXd temp_vec(cols + 1);
    double* temp_data = temp_vec.data();

    for (Eigen::Index i = 0; i < cols && total_rank < rows; ++i) {
        Eigen::Index remainingRows = rows - total_rank;
        Eigen::Index remainingCols = cols - i - 1;

        double beta;
        double hCoeff;
        Q2Jp.col(i).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
        if (std::abs(beta) > rank_threshold) {
            Q2Jp.coeffRef(total_rank, i) = beta;

            Q2Jp.bottomRightCorner(remainingRows, remainingCols)
                .applyHouseholderOnTheLeft(
                    Q2Jp.col(i).tail(remainingRows - 1), hCoeff, temp_data + i + 1);
            Q2r.tail(remainingRows)
                .applyHouseholderOnTheLeft(
                    Q2Jp.col(i).tail(remainingRows - 1), hCoeff, temp_data + cols);
            total_rank++;
        } else {
            Q2Jp.coeffRef(total_rank, i) = 0;
        }

        Q2Jp.col(i).tail(remainingRows - 1).setZero();

        if (i == Eigen::Index(marg_size) - 1) {
            marg_rank = total_rank;
        }
    }

    const Eigen::Index keep_valid_rows = total_rank - marg_rank;

    if (keep_valid_rows == 0) {
        marg_sqrt_H.resize(0, static_cast<Eigen::Index>(keep_size));
        marg_sqrt_b.resize(0);
        Q2Jp.resize(0, 0);
        Q2r.resize(0);
        return;
    }

    marg_sqrt_H = Q2Jp.block(marg_rank, marg_size, keep_valid_rows, keep_size);
    marg_sqrt_b = Q2r.segment(marg_rank, keep_valid_rows);

    Q2Jp.resize(0, 0);
    Q2r.resize(0);
}

}  // namespace tassel_core
