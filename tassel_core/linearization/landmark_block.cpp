#include "landmark_block.h"

#include "tassel_utils/utility.h"

namespace tassel_core {

Eigen::Matrix<double, 2, 3> LandmarkBlock::compute_tangent_base(Eigen::Vector3d uv) {
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

LandmarkBlock::LandmarkBlock()
    : lm_idx_(0),
      res_idx_(0),
      padding_idx_(0),
      num_cols_(0),
      num_rows_(0),
      state_(nullptr),
      feature_(nullptr) {}

void LandmarkBlock::allocate(Feature* const feature, State* const state) {
    state_ = state;
    feature_ = feature;

    int num_frames = state_->max_frame_count + 1;
    int start_frame_id = feature_->start_frame_id;

    tangent_base_vec_.resize(num_frames);
    for (int i = 1; i < static_cast<int>(feature_->observations.size()); ++i) {
        tangent_base_vec_[start_frame_id + i] = compute_tangent_base(feature_->observations[i].uv);
    }

    // Total pose Jacobian columns
    padding_idx_ = num_frames * POSE_SIZE;

    // Each target-frame observation contributes 2 rows; +1 damping row (1-DOF inverse depth)
    int num_obs = static_cast<int>(feature_->observations.size()) - 1;
    num_rows_ = num_obs * 2 + 1;

    // Padding for 4-column alignment
    int pad = padding_idx_ % 4;
    int padding_cols = (pad != 0) ? (4 - pad) : 0;

    lm_idx_ = padding_idx_ + padding_cols;  // landmark column (inverse depth)
    res_idx_ = lm_idx_ + 1;                 // residual column
    num_cols_ = res_idx_ + 1;

    storage_.resize(num_rows_, num_cols_);
    storage_.setZero();
}

void LandmarkBlock::compute_feature_linearization_block(
    const Eigen::Matrix3d& host_R, const Eigen::Vector3d& host_P, const Eigen::Matrix3d& target_R,
    const Eigen::Vector3d& target_P, const double inv_depth, const Eigen::Vector3d& uv_host,
    const Eigen::Vector3d& uv_target, const Eigen::Matrix<double, 2, 3>& tangent_base,
    Eigen::Matrix<double, 2, 6>& jacobian_H, Eigen::Matrix<double, 2, 6>& jacobian_T,
    Eigen::Matrix<double, 2, 1>& jacobian_L, Eigen::Matrix<double, 2, 1>& residual) {
    Eigen::Vector3d pt_in_H = uv_host / inv_depth;
    Eigen::Vector3d pt_in_W = host_R * pt_in_H + host_P;
    Eigen::Vector3d pt_in_T = target_R.transpose() * (pt_in_W - target_P);

    Eigen::Vector2d r = tangent_base * (pt_in_T.normalized() - uv_target);
    double norm = pt_in_T.norm();
    Eigen::Matrix3d jaco_norm =
        1.0 / norm *
        (Eigen::Matrix3d::Identity() - (pt_in_T * pt_in_T.transpose()) / (norm * norm));

    Eigen::Matrix<double, 2, 3> reduce = tangent_base * jaco_norm;

    Eigen::Matrix<double, 3, 6> J_host_3d;
    J_host_3d.block<3, 3>(0, 0) = target_R.transpose();
    J_host_3d.block<3, 3>(0, 3) = -target_R.transpose() * host_R * tassel_utils::skew_x(pt_in_H);
    jacobian_H = reduce * J_host_3d;

    Eigen::Matrix<double, 3, 6> J_target_3d;
    J_target_3d.block<3, 3>(0, 0) = -target_R.transpose();
    J_target_3d.block<3, 3>(0, 3) = tassel_utils::skew_x(pt_in_T);
    jacobian_T = reduce * J_target_3d;

    Eigen::Vector3d jaco_inv_3d =
        -target_R.transpose() * host_R * (uv_host / (inv_depth * inv_depth));
    jacobian_L = reduce * jaco_inv_3d;

    residual = r;
}

void LandmarkBlock::evaluate(
    int frame_id_H, int frame_id_T, DR_DQ& jacobian_H, DR_DQ& jacobian_T, DR_DL& jacobian_L,
    Residual& residual) {
    const Eigen::Matrix3d& host_R = state_->Rs[frame_id_H];
    const Eigen::Matrix3d& target_R = state_->Rs[frame_id_T];
    const Eigen::Vector3d& host_P = state_->Ps[frame_id_H];
    const Eigen::Vector3d& target_P = state_->Ps[frame_id_T];

    const Eigen::Matrix<double, 2, 3>& tangent_base = tangent_base_vec_[frame_id_T];
    Eigen::Vector3d uv_host = feature_->observations[0].uv;
    Eigen::Vector3d uv_target = feature_->observations[frame_id_T - frame_id_H].uv;
    double inv_depth = 1.0 / feature_->estimated_depth;

    compute_feature_linearization_block(
        host_R, host_P, target_R, target_P, inv_depth, uv_host, uv_target, tangent_base, jacobian_H,
        jacobian_T, jacobian_L, residual);
}

double LandmarkBlock::linearize() {
    storage_.setZero();

    int start_frame_id = feature_->start_frame_id;
    auto& observations = feature_->observations;

    DR_DQ jacobian_H, jacobian_T;
    DR_DL jacobian_L;
    Residual residual;

    double error_sum = 0.0;
    for (int offset = 1; offset < static_cast<int>(observations.size()); ++offset) {
        int frame_id_T = start_frame_id + offset;
        int obs_idx = offset - 1;
        int row = obs_idx * 2;

        evaluate(start_frame_id, frame_id_T, jacobian_H, jacobian_T, jacobian_L, residual);

        error_sum += residual.squaredNorm();

        storage_.block<2, 6>(row, start_frame_id * POSE_SIZE) += jacobian_H;
        storage_.block<2, 6>(row, frame_id_T * POSE_SIZE) += jacobian_T;
        storage_.block<2, 1>(row, lm_idx_) += jacobian_L;
        storage_.block<2, 1>(row, res_idx_) += residual;
    }

    return error_sum;
}

void LandmarkBlock::performQR() {
    // Householder QR on single landmark column (1-DOF inverse depth)
    Eigen::VectorXd tempVector1(num_cols_);
    Eigen::VectorXd tempVector2(num_rows_ - 1);

    size_t remainingRows = num_rows_ - 1;

    double beta;
    double tau;
    storage_.col(lm_idx_).head(remainingRows).makeHouseholder(tempVector2, tau, beta);

    storage_.block(0, 0, remainingRows, num_cols_)
        .applyHouseholderOnTheLeft(tempVector2, tau, tempVector1.data());
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

}  // namespace tassel_core
