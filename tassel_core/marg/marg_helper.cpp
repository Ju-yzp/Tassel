#include <Eigen/Householder>

#include "marg_helper.h"
#include "tassel_utils/macros.h"

#include <cmath>
#include <limits>

namespace tassel_core {

void MargHelper::marginalizeSqrtToSqrt(
    size_t marg_size, size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r,
    Eigen::MatrixXd& marg_sqrt_H, Eigen::VectorXd& marg_sqrt_b) {
    TASSEL_ASSERT(Eigen::Index(marg_size + keep_size) == Q2Jp.cols());
    TASSEL_ASSERT(Q2Jp.rows() == Q2r.rows());

    if (Q2Jp.rows() == 0) {
        marg_sqrt_H.resize(0, 0);
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

    Eigen::Index keep_valid_rows = std::max(total_rank - marg_rank, Eigen::Index(1));

    marg_sqrt_H = Q2Jp.block(marg_rank, marg_size, keep_valid_rows, keep_size);
    marg_sqrt_b = Q2r.segment(marg_rank, keep_valid_rows);

    Q2Jp.resize(0, 0);
    Q2r.resize(0);
}

}  // namespace tassel_core
