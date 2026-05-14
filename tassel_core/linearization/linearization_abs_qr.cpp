#include "linearization_abs_qr.h"
#include "linearization/landmark_block.h"
#include "marginalization/marg_helper.h"
#include "tassel_utils/macros.h"

// tbb
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// eigen
#include <Eigen/Core>
#include <utility>

namespace tassel_core {

LinearizationAbsQR::LinearizationAbsQR(
    int num_threads, std::shared_ptr<State> state, std::shared_ptr<FeatureManager> fm,
    LossVariant reprojection_loss, DepthLoss depth_loss, double min_depth, double max_depth,
    std::shared_ptr<MargLinData> marg_lin_data)
    : thread_pool_(num_threads),
      cur_state_(std::move(state)),
      feature_manager_(std::move(fm)),
      marg_lin_data_(std::move(marg_lin_data)),
      reprojection_loss_(std::move(reprojection_loss)),
      depth_loss_(std::move(depth_loss)),
      min_depth_(min_depth),
      max_depth_(max_depth) {
    auto features = feature_manager_->collectOptimizationFeatures();
    spdlog::info("{} landmarks need to optimize", static_cast<int>(features.size()));
    landmark_blocks_.resize(features.size(), LandmarkBlock(min_depth_, max_depth_));
    for (size_t i = 0; i < features.size(); ++i) {
        landmark_blocks_[i].allocate(
            features[i], cur_state_.get(), reprojection_loss_, depth_loss_);
    }
}

int LinearizationAbsQR::getPoseDim() const {
    return cur_state_->max_frame_count * tassel_utils::POSE_SIZE;
}

double LinearizationAbsQR::linearizeProbelm(bool* numerically_valid) {
    pose_damping_diagonal_ = 0.0;
    pose_damping_diagonal_sqrt_ = 0.0;

    size_t num_landmarks = landmark_blocks_.size();

    auto body = [&](const tbb::blocked_range<size_t>& range, std::pair<double, bool> error_valid) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
            error_valid.first += landmark_blocks_[r].linearize();
            error_valid.second = error_valid.second && !landmark_blocks_[r].isNumericalFailure();
        }
        return error_valid;
    };

    std::pair<double, bool> initial_value = {0.0, true};
    auto join = [](std::pair<double, bool> a, std::pair<double, bool> b) {
        a.first += b.first;
        a.second = a.second && b.second;
        return a;
    };

    tbb::blocked_range<size_t> range(0, num_landmarks);
    auto reduction_res = thread_pool_.execute(
        [&] { return tbb::parallel_reduce(range, initial_value, body, join); });

    if (numerically_valid) {
        *numerically_valid = reduction_res.second;
    }

    if (marg_lin_data_) {
        double marg_prior_error;
        MargHelper::computeMargPriorError(*marg_lin_data_, *cur_state_, marg_prior_error);
        reduction_res.first += marg_prior_error;
    }

    return reduction_res.first;
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

    size_t cols = cur_state_->max_frame_count * tassel_utils::POSE_SIZE;

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

    size_t opt_size =
        cur_state_->max_frame_count * tassel_utils::POSE_SIZE + landmark_blocks_.size();

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
    MargHelper::computeDelta(*cur_state_, delta);

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
    MargHelper::linearizeMargPrior(*marg_lin_data_, *cur_state_, H, b, marg_prior_error);
}

double LinearizationAbsQR::computeError() const {
    auto features = feature_manager_->collectOptimizationFeatures();
    double error = 0.0;

    for (auto* f : features) {
        int host_id = f->start_frame_id;
        Pose T_w_h = cur_state_->poses[host_id].get_pose();

        for (int offset = 1; offset < static_cast<int>(f->observations.size()); offset++) {
            int target_id = host_id + offset;
            Pose T_w_t = cur_state_->poses[target_id].get_pose();

            Eigen::Matrix<double, 2, 3> tangent_base =
                compute_tangent_base(f->observations[offset].uv);

            Eigen::Vector3d pt_in_H = f->observations[0].uv * f->estimated_depth;
            Eigen::Vector3d pt_in_W = T_w_h * pt_in_H;
            Eigen::Vector3d pt_in_T = T_w_t.inverse() * pt_in_W;

            Eigen::Vector2d residual =
                tangent_base * (pt_in_T.normalized() - f->observations[offset].uv);
            error += computeRho(reprojection_loss_, residual.norm());
        }
    }

    if (marg_lin_data_) {
        double marg_prior_error = 0.0;
        MargHelper::computeMargPriorError(*marg_lin_data_, *cur_state_, marg_prior_error);
        error += marg_prior_error;
    }
    return error;
}

void LinearizationAbsQR::saveState() {
    for (auto& p : cur_state_->poses) p.save();

    saved_feature_depths_.clear();
    auto features = feature_manager_->collectOptimizationFeatures();
    for (auto* f : features) {
        saved_feature_depths_[f] = f->estimated_depth;
    }

    saved_lms_states_.resize(landmark_blocks_.size());
    for (size_t i = 0; i < landmark_blocks_.size(); ++i) {
        saved_lms_states_[i] = landmark_blocks_[i].getState();
    }
}

void LinearizationAbsQR::restoreState() {
    for (auto& p : cur_state_->poses) p.restore();

    for (auto& [f, depth] : saved_feature_depths_) {
        f->estimated_depth = depth;
    }

    for (size_t i = 0; i < saved_lms_states_.size() && i < landmark_blocks_.size(); ++i) {
        landmark_blocks_[i].setState(saved_lms_states_[i]);
    }
}

void LinearizationAbsQR::applyPoseInc(const Eigen::VectorXd& inc) {
    for (int k = 0; k < cur_state_->cur_frame_count; k++) {
        Eigen::Vector<double, 6> delta = inc.segment<6>(k * tassel_utils::POSE_SIZE);
        cur_state_->poses[k].applyDelta(delta);
    }
}

double LinearizationAbsQR::backSubstitute(const Eigen::VectorXd& pose_inc) {
    TASSEL_ASSERT(pose_inc.size() == cur_state_->max_frame_count * tassel_utils::POSE_SIZE);

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

        l_diff += MargHelper::computeMargPriorModelCostChange(
            *marg_lin_data_, *cur_state_, marg_scaling_, pose_inc_marg);
    }

    return l_diff;
}
}  // namespace tassel_core
