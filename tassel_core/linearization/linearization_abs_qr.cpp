#include "linearization_abs_qr.h"
#include "tassel_utils/macros.h"

// tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// eigen
#include <Eigen/Core>

namespace tassel_core {

int LinearizationAbsQR::getPoseDim() const {
    return state_->max_frame_count * tassel_utils::POSE_SIZE;
}

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
    std::vector<size_t> row_offsets(landmark_blocks_.size());
    size_t rows = 0;
    for (size_t i = 0; i < landmark_blocks_.size(); ++i) {
        row_offsets[i] = rows;
        rows += landmark_blocks_[i].getNumRows() - 1;
    }

    size_t cols = state_->max_frame_count * tassel_utils::POSE_SIZE;

    // 为边缘化部分预留位置，暂时不填充
    size_t marg_start_idx = rows;
    if (marg_lin_data_) rows += marg_lin_data_->H.rows();

    size_t damping_start_idx = rows;
    if (hasPoseDamping()) {
        rows += cols;
    }

    Q2Jp.setZero(rows, cols);
    Q2r.setZero(rows);
    auto body = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
            const auto& lb = landmark_blocks_[r];
            lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, row_offsets[r]);
        }
    };

    tbb::blocked_range<size_t> range(0, row_offsets.size());
    thread_pool_.execute([&] { tbb::parallel_for(range, body); });

    // Add damping
    if (hasPoseDamping()) {
        get_dense_Q2Jp_Q2r_pose_damping(Q2Jp, damping_start_idx);
    }

    // Add marginalization
    if (marg_lin_data_) {
        get_dense_Q2Jp_Q2r_marg_prior(Q2Jp, Q2r, marg_start_idx);
    }
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

    size_t opt_size = state_->max_frame_count * tassel_utils::POSE_SIZE + landmark_blocks_.size();

    Reductor r(opt_size, landmark_blocks_);

    // go over all host frames
    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    thread_pool_.execute([&] { tbb::parallel_reduce(range, r); });

    // Add damping
    if (hasPoseDamping()) {
        add_dense_H_b_pose_damping(r.H_);
    }
    // Add marginalization
    if (marg_lin_data_) {
        add_dense_H_b_marg_prior(r.H_, r.b_);
    }

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
    marg_helper_->computeDelta(marg_lin_data_->old_state, delta);

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
    marg_helper_->linearizeMargPrior(*marg_lin_data_, *state_, H, b, marg_prior_error);
}

double LinearizationAbsQR::computeError() const {
    auto features = feature_manager_->collectOptimizationFeatures();
    double error = 0.0;

    for (auto* f : features) {
        int host_id = f->start_frame_id;
        Pose T_w_h = state_->poses[host_id].get_pose();

        for (int offset = 1; offset < static_cast<int>(f->observations.size()); offset++) {
            int target_id = host_id + offset;
            Pose T_w_t = state_->poses[target_id].get_pose();

            Eigen::Matrix<double, 2, 3> tangent_base =
                compute_tangent_base(f->observations[offset].uv);

            Eigen::Vector3d pt_in_H = f->observations[0].uv * f->estimated_depth;
            Eigen::Vector3d pt_in_W = T_w_h * pt_in_H;
            Eigen::Vector3d pt_in_T = T_w_t.inverse() * pt_in_W;

            Eigen::Vector2d residual =
                tangent_base * (pt_in_T.normalized() - f->observations[offset].uv);
            error += 0.5 * residual.squaredNorm();
        }
    }
    return error;
}

void LinearizationAbsQR::saveState() {
    for (auto& p : state_->poses) p.save();

    saved_feature_depths_.clear();
    auto features = feature_manager_->collectOptimizationFeatures();
    for (auto* f : features) {
        saved_feature_depths_[f] = f->estimated_depth;
    }
}

void LinearizationAbsQR::restoreState() {
    for (auto& p : state_->poses) p.restore();

    for (auto& [f, depth] : saved_feature_depths_) {
        f->estimated_depth = depth;
    }
}

void LinearizationAbsQR::applyPoseInc(const Eigen::VectorXd& inc) {
    for (int k = 0; k < state_->cur_frame_count; k++) {
        Eigen::Vector<double, 6> delta = inc.segment<6>(k * tassel_utils::POSE_SIZE);
        state_->poses[k].applyDelta(delta);
    }
}

double LinearizationAbsQR::backSubstitute(const Eigen::VectorXd& pose_inc) {
    TASSEL_ASSERT(pose_inc.size() == state_->max_frame_count * tassel_utils::POSE_SIZE);

    auto body = [&](const tbb::blocked_range<size_t>& range, double l_diff) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
            landmark_blocks_[r].backSubstitute(pose_inc, l_diff);
        }
        return l_diff;
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    double l_diff = thread_pool_.execute(
        [&] { return tbb::parallel_reduce(range, 0.0, body, std::plus<double>()); });

    if (marg_lin_data_) {
        size_t marg_size = marg_lin_data_->H.cols();
        Eigen::VectorXd pose_inc_marg = pose_inc.head(marg_size);

        // l_diff += estimator->computeMargPriorModelCostChange(
        //     *marg_lin_data, marg_scaling, pose_inc_marg);
    }

    return l_diff;
}
}  // namespace tassel_core
