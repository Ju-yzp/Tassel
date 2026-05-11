#include "linearization_abs_qr.h"
#include "tassel_utils/macros.h"

// tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// eigen
#include <Eigen/Core>

namespace tassel_core {

double LinearizationAbsQR::linearizeProbelm() {
    pose_damping_diagonal_ = 0.0;
    pose_damping_diagonal_sqrt_ = 0.0;

    auto features = feature_manager_->collectOptimizationFeatures();
    landmark_blocks_.resize(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        landmark_blocks_[i].allocate(features[i], state_.get());
    }

    size_t num_landmarks = landmark_blocks_.size();

    double initial_value = 0.0;

    auto body = [&](const tbb::blocked_range<size_t>& range, double error) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
            error += landmark_blocks_[r].linearize();
        }
        return error;
    };

    auto join = [](double a, double b) { return a + b; };

    tbb::blocked_range<size_t> range(0, num_landmarks);
    return thread_pool_.execute(
        [&] { return tbb::parallel_reduce(range, initial_value, body, join); });
}

void LinearizationAbsQR::performQR() {
    auto body = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
            landmark_blocks_[r].performQR();
        }
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    thread_pool_.execute([&] { tbb::parallel_for(range, body); });
}

void LinearizationAbsQR::setPoseDamping(const double lambda) {
    TASSEL_ASSERT(lambda >= 0);

    pose_damping_diagonal_ = lambda;
    pose_damping_diagonal_sqrt_ = std::sqrt(lambda);
}
void LinearizationAbsQR::get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r) const {
    size_t total_size = num_rows_Q2r_;
    size_t poses_size;

    size_t lm_start_idx = 0;

    // 为边缘化部分预留位置，暂时不填充
    size_t marg_start_idx = total_size;
    if (marg_lin_data_) total_size += marg_lin_data_->H.rows();

    size_t damping_start_idx = total_size;
    if (hasPoseDamping()) {
        total_size += poses_size;
    }

    Q2Jp.setZero(total_size, poses_size);
    Q2r.setZero(total_size);
    auto body = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
            const auto& lb = landmark_blocks_[r];
            lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, lm_start_idx + landmark_block_idx_[r]);
        }
    };

    tbb::blocked_range<size_t> range(0, landmark_block_idx_.size());
    tbb::parallel_for(range, body);

    // Add damping
    get_dense_Q2Jp_Q2r_pose_damping(Q2Jp, damping_start_idx);

    // Add marginalization
    get_dense_Q2Jp_Q2r_marg_prior(Q2Jp, Q2r, marg_start_idx);
}

void LinearizationAbsQR::get_dense_H_b(Eigen::MatrixXd& H, Eigen::VectorXd& b) const {
    struct Reductor {
        Reductor(size_t opt_size, const std::vector<LandmarkBlock>& landmark_blocks)
            : opt_size_(opt_size), landmark_blocks_(landmark_blocks) {
            H_.setZero(opt_size_, opt_size_);
            b_.setZero(opt_size_);
        }

        void operator()(const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                auto& lb = landmark_blocks_[r];
                lb.add_dense_H_b(H_, b_);
            }
        }

        Reductor(Reductor& a, tbb::split)
            : opt_size_(a.opt_size_), landmark_blocks_(a.landmark_blocks_) {
            H_.setZero(opt_size_, opt_size_);
            b_.setZero(opt_size_);
        };

        inline void join(Reductor& b) {
            H_ += b.H_;
            b_ += b.b_;
        }

        size_t opt_size_;
        const std::vector<LandmarkBlock>& landmark_blocks_;

        Eigen::MatrixXd H_;
        Eigen::VectorXd b_;
    };

    size_t opt_size = aom_->total_size;

    Reductor r(opt_size, landmark_blocks_);

    // go over all host frames
    tbb::blocked_range<size_t> range(0, landmark_block_idx_.size());
    tbb::parallel_reduce(range, r);

    // Add damping
    add_dense_H_b_pose_damping(r.H_);

    // Add marginalization
    add_dense_H_b_marg_prior(r.H_, r.b_);

    H = std::move(r.H_);
    b = std::move(r.b_);
}

void LinearizationAbsQR::get_dense_Q2Jp_Q2r_pose_damping(
    Eigen::MatrixXd& Q2Jp, size_t start_idx) const {
    size_t poses_size = tassel_utils::POSE_SIZE;
    if (hasPoseDamping()) {
        Q2Jp.block(start_idx, 0, poses_size, poses_size).diagonal().array() =
            pose_damping_diagonal_sqrt_;
    }
}

void LinearizationAbsQR::get_dense_Q2Jp_Q2r_marg_prior(
    Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, size_t start_idx) const {
    if (!marg_lin_data_) return;

    size_t marg_rows = marg_lin_data_->H.rows();
    size_t marg_cols = marg_lin_data_->H.cols();

    Eigen::VectorXd delta;
    marg_helper_->computeDelta(marg_lin_data_->order, delta);

    if (marg_scaling_.rows() > 0) {
        Q2Jp.block(start_idx, 0, marg_rows, marg_cols) =
            marg_lin_data_->H * marg_scaling_.asDiagonal();
    } else {
        Q2Jp.block(start_idx, 0, marg_rows, marg_cols) = marg_lin_data_->H;
    }

    Q2r.segment(start_idx, marg_rows) = marg_lin_data_->H * delta + marg_lin_data_->b;
}

void LinearizationAbsQR::add_dense_H_b_pose_damping(Eigen::MatrixXd& H) const {
    if (hasPoseDamping()) {
        H.diagonal().array() += pose_damping_diagonal_;
    }
}

void LinearizationAbsQR::add_dense_H_b_marg_prior(Eigen::MatrixXd& H, Eigen::VectorXd& b) const {
    if (!marg_lin_data_) return;

    // Scaling not supported ATM
    TASSEL_ASSERT(marg_scaling_.rows() == 0);

    double marg_prior_error;
    marg_helper_->linearizeMargPrior(*marg_lin_data_, *aom_.get(), H, b, marg_prior_error);
}
}  // namespace tassel_core
