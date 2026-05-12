#include "marg_helper.h"
#include "tassel_utils/macros.h"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace tassel_core {

void MargHelper::computeDelta(const AbsOrderMap& order, Eigen::VectorXd& delta) const {
    size_t marg_size = order.total_size;
    delta.setZero(marg_size);
    for (const auto& kv : order.abs_order_map) {
        if (kv.second.second == tassel_utils::POSE_SIZE) {
            TASSEL_ASSERT(frame_poses_.at(kv.first).isLinearized());
            delta.template segment<tassel_utils::POSE_SIZE>(kv.second.first) =
                frame_poses_.at(kv.first).get_delta();
        } else {
            TASSEL_ASSERT(false);
        }
    }
}

void MargHelper::linearizeMargPrior(
    const MargLinData& mld, const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
    double& marg_prior_error) {
    TASSEL_ASSERT(static_cast<size_t>(mld.H.cols()) == mld.order.total_size);

    for (const auto& kv : mld.order.abs_order_map) {
        TASSEL_ASSERT(aom.abs_order_map.at(kv.first) == kv.second);
        TASSEL_ASSERT(static_cast<size_t>(kv.second.first) < mld.order.total_size);
    }

    const size_t marg_size = mld.order.total_size;

    // 计算增量
    Eigen::VectorXd delta;
    computeDelta(mld.order, delta);

    // 默认使用平方根形式
    abs_H.topLeftCorner(marg_size, marg_size) += mld.H.transpose() * mld.H;
    abs_b.head(marg_size) += mld.H.transpose() * (mld.b + mld.H * delta);

    marg_prior_error = delta.transpose() * mld.H.transpose() * (0.5 * mld.H * delta + mld.b);
}

void MargHelper::computeMargPriorError(const MargLinData& mld, double& marg_prior_error) const {
    TASSEL_ASSERT(size_t(mld.H.cols()) == mld.order.total_size);
    // 当前代价的计算
    //
    //    P(0) = 0.5 || J*delta + r ||^2
    //         = 0.5 delta^T J^T J delta + delta^T J^T r + 0.5 r^T r
    //         = 0.5 delta^T Hmarg delta + delta^T bmarg + 0.5 r^T r.
    //
    // Note: Since the r^T r term does not change with delta, we drop it from the
    // error computation. The main need for the error value is for comparing
    // the cost before and after an update to delta in the optimization loop. This
    // also means the computed error can be negative.

    Eigen::VectorXd delta;
    computeDelta(mld.order, delta);

    // 默认使用平方根形式
    marg_prior_error = delta.transpose() * mld.H.transpose() * (0.5 * mld.H * delta + mld.b);
}

void MargHelper::marginalizeOldest(
    size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, Eigen::MatrixXd& marg_sqrt_H,
    Eigen::VectorXd& marg_sqrt_b) {
    TASSEL_ASSERT(Eigen::Index(1 + keep_size) == Q2Jp.cols());
    TASSEL_ASSERT(Q2Jp.rows() == Q2r.rows());

    // 只保留强约束以及先验，弱约束由于缺秩会自动丢弃
    Eigen::Index marg_rank = 0;
    Eigen::Index total_rank = 0;

    const double rank_theshold = std::sqrt(std::numeric_limits<double>::epsilon());

    const Eigen::Index rows = Q2Jp.rows();
    const Eigen::Index cols = Q2Jp.cols();

    Eigen::VectorXd temp_vec;
    temp_vec.resize(cols + 1);
    double* temp_data = temp_vec.data();

    for (Eigen::Index i = 0; i < cols && total_rank < rows; ++i) {
        Eigen::Index remainingRows = rows - total_rank;
        Eigen::Index remainingCols = cols - i - 1;

        double beta;
        double hCoeff;
        Q2Jp.col(i).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
        if (std::abs(beta) > rank_theshold) {
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

        if (i == 0) {
            marg_rank = total_rank;
        }
    }

    Eigen::Index keep_valid_rows = std::max(total_rank - marg_rank, Eigen::Index(1));

    marg_sqrt_H = Q2Jp.block(marg_rank, 1, keep_valid_rows, keep_size);
    marg_sqrt_b = Q2r.segment(marg_rank, keep_valid_rows);

    Q2Jp.resize(0, 0);
    Q2r.resize(0);
}
}  // namespace tassel_core
