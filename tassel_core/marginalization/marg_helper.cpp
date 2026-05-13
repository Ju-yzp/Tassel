#include "marg_helper.h"
#include "tassel_utils/macros.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace tassel_core {

void MargHelper::computeDelta(const State& state, Eigen::VectorXd& delta) {
    size_t marg_size = (state.max_frame_count - 1) * tassel_utils::POSE_SIZE;
    delta.setZero(marg_size);
    // Extract kept frames 0..(n-2), matching the prior's column mapping.
    // Frame n-1 is the newest frame, not yet in the prior.
    for (int i = 0; i < state.max_frame_count - 1; i++) {
        TASSEL_ASSERT(state.poses[i].isLinearized());
        delta.segment<tassel_utils::POSE_SIZE>(i * tassel_utils::POSE_SIZE) =
            state.poses[i].get_delta();
    }
}

void MargHelper::linearizeMargPrior(
    const MargLinData& mld, const State& cur_state, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
    double& marg_prior_error) {
    size_t total_size = (cur_state.max_frame_count - 1) * tassel_utils::POSE_SIZE;
    size_t marg_size = total_size;

    TASSEL_ASSERT(static_cast<size_t>(mld.H.cols()) == total_size);

    // 计算增量
    Eigen::VectorXd delta;
    computeDelta(cur_state, delta);

    // 默认使用平方根形式
    abs_H.topLeftCorner(marg_size, marg_size) += mld.H.transpose() * mld.H;
    abs_b.head(marg_size) += mld.H.transpose() * (mld.b + mld.H * delta);

    marg_prior_error = delta.transpose() * mld.H.transpose() * (0.5 * mld.H * delta + mld.b);
}

void MargHelper::computeMargPriorError(
    const MargLinData& mld, const State& cur_state, double& marg_prior_error) {
    size_t total_size = (cur_state.max_frame_count - 1) * tassel_utils::POSE_SIZE;
    TASSEL_ASSERT(size_t(mld.H.cols()) == total_size);
    // 当前代价的计算
    //
    //    P(0) = 0.5 || J*delta + r ||^2
    //         = 0.5 delta^T J^T J delta + delta^T J^T r + 0.5 r^T r
    //         = 0.5 delta^T Hmarg delta + delta^T bmarg + 0.5 r^T r.
    //
    Eigen::VectorXd delta;
    computeDelta(cur_state, delta);

    // 默认使用平方根形式
    marg_prior_error = delta.transpose() * mld.H.transpose() * (0.5 * mld.H * delta + mld.b);
}

void MargHelper::marginalizeOldest(
    size_t marg_size, size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r,
    Eigen::MatrixXd& marg_sqrt_H, Eigen::VectorXd& marg_sqrt_b) {
    TASSEL_ASSERT(Eigen::Index(marg_size + keep_size) == Q2Jp.cols());
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

double MargHelper::computeMargPriorModelCostChange(
    const MargLinData& mld, const State& cur_state, const Eigen::VectorXd& marg_scaling,
    const Eigen::VectorXd& marg_pose_inc) {
    Eigen::VectorXd delta;
    computeDelta(cur_state, delta);

    Eigen::VectorXd J_inc = marg_pose_inc;
    if (marg_scaling.rows() > 0) {
        J_inc = marg_scaling.asDiagonal() * J_inc;
    }

    double l_diff;

    const Eigen::VectorXd b_Jdelta = mld.H * delta + mld.b;

    J_inc = mld.H * J_inc;
    l_diff = -J_inc.transpose() * (b_Jdelta + 0.5 * J_inc);
    return l_diff;
}
}  // namespace tassel_core
