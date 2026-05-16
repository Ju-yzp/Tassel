#include "landmark_block.h"
#include <spdlog/spdlog.h>
#include "frond_end/feature.h"
#include "tassel_utils/macros.h"

#include <sophus/se3.hpp>

namespace tassel_core {
Eigen::Matrix<double, 2, 3> compute_tangent_base(Eigen::Vector3d uv) {
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = uv.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp) tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    Eigen::Matrix<double, 2, 3> tangent_base;
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
    return tangent_base;
}

bool compute_feature_linearization_block(
    const Sophus::SE3d& T_w_h, const Sophus::SE3d& T_w_t, double depth,
    const Eigen::Vector3d& uv_host, const Eigen::Vector3d& uv_target,
    const Eigen::Matrix<double, 2, 3>& tangent_base, Eigen::Matrix<double, 2, 6>& jacobian_H,
    Eigen::Matrix<double, 2, 6>& jacobian_T, Eigen::Matrix<double, 2, 1>& jacobian_L,
    Eigen::Matrix<double, 2, 1>& residual) {
    Eigen::Matrix3d R_h = T_w_h.rotationMatrix();
    Eigen::Matrix3d R_t = T_w_t.rotationMatrix();
    Eigen::Matrix3d R_t_inv = R_t.transpose();

    Eigen::Vector3d pt_in_H = uv_host * depth;
    Eigen::Vector3d pt_in_W = T_w_h * pt_in_H;
    Eigen::Vector3d pt_in_T = T_w_t.inverse() * pt_in_W;

    double norm = pt_in_T.norm();
    if (pt_in_T.z() < 0.0) {
        return false;
    }

    residual = tangent_base * (pt_in_T / norm - uv_target);

    Eigen::Matrix3d jaco_norm =
        1.0 / norm *
        (Eigen::Matrix3d::Identity() - (pt_in_T * pt_in_T.transpose()) / (norm * norm));

    Eigen::Matrix<double, 2, 3> reduce = tangent_base * jaco_norm;

    Eigen::Matrix<double, 3, 6> J_host_3d;
    J_host_3d.block<3, 3>(0, 0) = R_t_inv * R_h;
    J_host_3d.block<3, 3>(0, 3) = -R_t_inv * R_h * Sophus::SO3d::hat(pt_in_H);
    jacobian_H = reduce * J_host_3d;

    Eigen::Matrix<double, 3, 6> J_target_3d;
    J_target_3d.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    J_target_3d.block<3, 3>(0, 3) = Sophus::SO3d::hat(pt_in_T);
    jacobian_T = reduce * J_target_3d;

    double inv_depth = 1.0 / depth;
    jacobian_L = -reduce * R_t_inv * R_h * (uv_host / (inv_depth * inv_depth));

    return true;
}

LandmarkBlock::LandmarkBlock(double min_depth, double max_depth)
    : lm_idx_(0),
      res_idx_(0),
      padding_idx_(0),
      num_cols_(0),
      num_rows_(0),
      state_(nullptr),
      feature_(nullptr),
      Jl_col_scale_(1.0),
      lms_(LandmarkState::Uninitialized),
      min_depth_(min_depth),
      max_depth_(max_depth) {}

void LandmarkBlock::allocate(
    Feature* const feature, State* const state, const LossVariant& reprojection_loss) {
    TASSEL_ASSERT(lms_ == LandmarkState::Uninitialized);
    state_ = state;
    feature_ = feature;
    reprojection_loss_ = reprojection_loss;

    int num_frames = state_->max_frame_count;
    int start_frame_id = feature_->start_frame_id;

    tangent_base_vec_.resize(num_frames);
    for (int i = 1; i < static_cast<int>(feature_->observations.size()); ++i) {
        tangent_base_vec_[start_frame_id + i] = compute_tangent_base(feature_->observations[i].uv);
    }

    padding_idx_ = num_frames * tassel_utils::POSE_SIZE;

    int num_obs = static_cast<int>(feature_->observations.size()) - 1;
    num_rows_ = num_obs * 2 + 1;

    int pad = padding_idx_ % 4;
    int padding_cols = (pad != 0) ? (4 - pad) : 0;

    lm_idx_ = padding_idx_ + padding_cols;  // landmark column (inverse depth)
    res_idx_ = lm_idx_ + 1;                 // residual column
    num_cols_ = res_idx_ + 1;

    storage_.resize(num_rows_, num_cols_);
    storage_.setZero();

    lms_ = LandmarkState::Allocated;
}

bool LandmarkBlock::evaluate(
    int frame_id_H, int frame_id_T, Eigen::Matrix<double, 2, 6>& jacobian_H,
    Eigen::Matrix<double, 2, 6>& jacobian_T, Eigen::Matrix<double, 2, 1>& jacobian_L,
    Eigen::Matrix<double, 2, 1>& residual) {
    Pose T_w_h = state_->poses[frame_id_H].get_pose();
    Pose T_w_t = state_->poses[frame_id_T].get_pose();

    const Eigen::Matrix<double, 2, 3>& tangent_base = tangent_base_vec_[frame_id_T];
    Eigen::Vector3d uv_host = feature_->observations[0].uv;
    Eigen::Vector3d uv_target = feature_->observations[frame_id_T - frame_id_H].uv;
    double depth = feature_->estimated_depth;

    return compute_feature_linearization_block(
        T_w_h, T_w_t, depth, uv_host, uv_target, tangent_base, jacobian_H, jacobian_T, jacobian_L,
        residual);
}

double LandmarkBlock::linearize() {
    TASSEL_ASSERT(
        lms_ == LandmarkState::Allocated || lms_ == LandmarkState::Linearized ||
        lms_ == LandmarkState::Marginalized || lms_ == LandmarkState::NumericalFailure);
    storage_.setZero();
    damping_rotations_.clear();
    damping_rotations_.reserve(6);

    bool numerically_valid = true;
    int start_frame_id = feature_->start_frame_id;
    auto& observations = feature_->observations;

    Eigen::Matrix<double, 2, 6> jacobian_H, jacobian_T;
    Eigen::Matrix<double, 2, 1> jacobian_L;
    Eigen::Matrix<double, 2, 1> residual;

    if (static_cast<int>(observations.size()) <= 2) {
        spdlog::error("Observation size is less than 2");
    }

    double error_sum = 0.0;
    for (int offset = 1; offset < static_cast<int>(observations.size()); ++offset) {
        int frame_id_T = start_frame_id + offset;
        int obs_idx = offset - 1;
        int row = obs_idx * 2;

        if (evaluate(start_frame_id, frame_id_T, jacobian_H, jacobian_T, jacobian_L, residual)) {
            numerically_valid = numerically_valid && jacobian_H.array().isFinite().all() &&
                                jacobian_L.array().isFinite().all();
            double s = residual.norm();
            double w = computeWeight(reprojection_loss_, s);
            double scale = std::sqrt(w);

            error_sum += computeRho(reprojection_loss_, s);
            if (start_frame_id > 0) {
                storage_.block<2, 6>(row, start_frame_id * tassel_utils::POSE_SIZE) +=
                    scale * jacobian_H;
            } else {
                storage_.block<2, 6>(row, frame_id_T * tassel_utils::POSE_SIZE) =
                    Eigen::Matrix<double, 2, 6>::Zero();
            }
            storage_.block<2, 6>(row, frame_id_T * tassel_utils::POSE_SIZE) += scale * jacobian_T;
            storage_.block<2, 1>(row, lm_idx_) += scale * jacobian_L;
            storage_.block<2, 1>(row, res_idx_) += scale * residual;
        }
    }
    if (numerically_valid) {
        lms_ = LandmarkState::Linearized;
        scaleJl_cols();
    } else {
        lms_ = LandmarkState::NumericalFailure;
    }
    return error_sum;
}

void LandmarkBlock::performQR() {
    Eigen::JacobiRotation<double> gr;
    for (size_t m = num_rows_ - 2; m > 0; m--) {
        gr.makeGivens(storage_(m - 1, lm_idx_), storage_(m, lm_idx_));
        storage_.applyOnTheLeft(m, m - 1, gr);
    }

    lms_ = LandmarkState::Marginalized;
}

void LandmarkBlock::get_dense_Q2Jp_Q2r(
    Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, size_t start_idx) const {
    Q2r.segment(start_idx, num_rows_ - 1) = storage_.col(res_idx_).tail(num_rows_ - 1);

    Q2Jp.block(start_idx, 0, num_rows_ - 1, padding_idx_) =
        storage_.block(1, 0, num_rows_ - 1, padding_idx_);
}

void LandmarkBlock::add_dense_H_b(Eigen::MatrixXd& H, Eigen::VectorXd& b) const {
    const auto r = storage_.col(res_idx_).tail(num_rows_ - 1);
    const auto J = storage_.block(1, 0, num_rows_ - 1, padding_idx_);

    H.noalias() += J.transpose() * J;
    b.noalias() += J.transpose() * r;
}

void LandmarkBlock::addJp_diag2(Eigen::VectorXd& res) const {
    TASSEL_ASSERT(lms_ == LandmarkState::Linearized);
    int start_frame_id = feature_->start_frame_id;
    const auto& observations = feature_->observations;

    for (int offset = 1; offset < static_cast<int>(observations.size()); ++offset) {
        int obs_idx = offset - 1;
        int frame_id_T = start_frame_id + offset;

        // host frame Jp
        res.segment<tassel_utils::POSE_SIZE>(start_frame_id * tassel_utils::POSE_SIZE) +=
            storage_
                .block<2, tassel_utils::POSE_SIZE>(
                    obs_idx * 2, start_frame_id * tassel_utils::POSE_SIZE)
                .colwise()
                .squaredNorm();

        // target frame Jp
        res.segment<tassel_utils::POSE_SIZE>(frame_id_T * tassel_utils::POSE_SIZE) +=
            storage_
                .block<2, tassel_utils::POSE_SIZE>(
                    obs_idx * 2, frame_id_T * tassel_utils::POSE_SIZE)
                .colwise()
                .squaredNorm();
    }
}

void LandmarkBlock::scaleJl_cols() {
    TASSEL_ASSERT(lms_ == LandmarkState::Linearized);

    const double eps = 1e-6;
    double col_norm = storage_.col(lm_idx_).head(num_rows_ - 1).norm();
    Jl_col_scale_ = 1.0 / (eps + col_norm);

    storage_.col(lm_idx_).head(num_rows_ - 1) *= Jl_col_scale_;
}

void LandmarkBlock::scaleJp_cols(const Eigen::VectorXd& jacobian_scaling) {
    TASSEL_ASSERT(lms_ == LandmarkState::Marginalized);

    // we assume we apply scaling before damping (we exclude the last 3 rows)
    TASSEL_ASSERT(!hasLandmarkDamping());

    storage_.topLeftCorner(num_rows_ - 1, padding_idx_) *= jacobian_scaling.asDiagonal();
}

void LandmarkBlock::setLandmarkDamping(double lambda) {
    TASSEL_ASSERT(lms_ == LandmarkState::Marginalized);
    TASSEL_ASSERT(lambda >= 0);

    if (hasLandmarkDamping()) {
        TASSEL_ASSERT(damping_rotations_.size() == 1);

        storage_.applyOnTheLeft(num_rows_ - 1, 0, damping_rotations_.back().adjoint());
        damping_rotations_.pop_back();
    }

    if (lambda == 0) {
        storage_(num_rows_ - 1, lm_idx_) = 0;
    } else {
        TASSEL_ASSERT(std::isfinite(Jl_col_scale_));

        storage_(num_rows_ - 1, lm_idx_) = std::sqrt(lambda);

        TASSEL_ASSERT(damping_rotations_.empty());

        damping_rotations_.emplace_back();
        damping_rotations_.back().makeGivens(
            storage_(0, lm_idx_), storage_(num_rows_ - 1, lm_idx_));
        storage_.applyOnTheLeft(num_rows_ - 1, 0, damping_rotations_.back());
    }
}

void LandmarkBlock::backSubstitute(const Eigen::VectorXd& pose_inc, double& l_diff) {
    if (lms_ == LandmarkState::NumericalFailure) {
        return;
    }
    TASSEL_ASSERT(lms_ == LandmarkState::Marginalized);

    // For now we include all columns in LMB
    TASSEL_ASSERT(pose_inc.size() == static_cast<Eigen::Index>(padding_idx_));

    const double Q1Jl = storage_(0, lm_idx_);
    const double Q1Jr = storage_(0, res_idx_);
    const auto Q1Jp = storage_.row(0).head(padding_idx_);

    // Guard against near-zero Q1Jl — would produce infinite depth update
    // and poison subsequent Jacobian computations with NaN.
    if (std::abs(Q1Jl) < 1e-12) {
        setLandmarkDamping(0);
        return;
    }

    double inc = -(Q1Jr + Q1Jp.dot(pose_inc)) / Q1Jl;

    // We want to compute the model cost change. The model function is
    //
    //     L(inc) = F(x) + inc^T J^T r + 0.5 inc^T J^T J inc
    //
    // and thus the expected decrease in cost for the computed increment is
    //
    //     l_diff = L(0) - L(inc)
    //            = - inc^T J^T r - 0.5 inc^T J^T J inc
    //            = - inc^T J^T (r + 0.5 J inc)
    //            = - (J inc)^T (r + 0.5 (J inc)).
    //
    // Here we have J = [Jp, Jl] under the orthogonal projection Q = [Q1, Q2],
    // i.e. the linearized system (model cost) is
    //
    //    L(inc) = 0.5 || J inc + r ||^2 = 0.5 || Q^T J inc + Q^T r ||^2
    //
    // and below we thus compute
    //
    //    l_diff = - (Q^T J inc)^T (Q^T r + 0.5 (Q^T J inc)).
    //
    // We have
    //             | Q1^T |            | Q1^T Jp   Q1^T Jl |
    //    Q^T J =  |      | [Jp, Jl] = |                   |
    //             | Q2^T |            | Q2^T Jp      0    |.
    //
    // Note that Q2 is the nullspace of Jl, and Q1^T Jl == R. So with inc =
    // [incp^T, incl^T]^T we have
    //
    //                | Q1^T Jp incp + Q1^T Jl incl |
    //    Q^T J inc = |                             |
    //                | Q2^T Jp incp                |
    //

    // undo damping before we compute the model cost difference
    setLandmarkDamping(0);

    // compute "Q^T J incp"
    Eigen::VectorXd QJinc = storage_.topLeftCorner(num_rows_ - 1, padding_idx_) * pose_inc;

    // add "Q1^T Jl incl" to the first 1 rows
    QJinc(0) += Q1Jl * inc;

    auto Qr = storage_.col(res_idx_).head(num_rows_ - 1);
    l_diff -= QJinc.transpose() * (0.5 * QJinc + Qr);

    // TODO: detect and handle case like ceres, allowing a few iterations but
    // stopping eventually
    // if (!inc.array().isFinite().all() ||
    //     !lm_ptr->direction.array().isFinite().all() ||
    //     !std::isfinite(lm_ptr->inv_dist)) {
    //   std::cerr << "Numerical failure in backsubstitution\n";
    // }

    // Note: scale only after computing model cost change
    inc *= Jl_col_scale_;

    feature_->estimated_depth = 1.0 / (1.0 / feature_->estimated_depth + inc);
    if (feature_->estimated_depth < min_depth_ || feature_->estimated_depth > max_depth_) {
        lms_ = LandmarkState::NumericalFailure;
    }
}
}  // namespace tassel_core
