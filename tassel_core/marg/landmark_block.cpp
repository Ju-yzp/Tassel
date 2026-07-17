#include "landmark_block.h"
#include "factor/visual_factor.h"
#include "state/state.h"
#include "tassel_utils/macros.h"

#include <Eigen/Core>
#include <cmath>
#include <sophus/so3.hpp>
#include <vector>

namespace tassel_core {

LandmarkBlock::LandmarkBlock(int dim, ceres::LossFunction* loss)
    : delay_idx_(0),
      lm_idx_(0),
      res_idx_(0),
      padding_idx_(0),
      num_rows_(0),
      eliminated_landmark_rank_(0),
      qr_performed_(false),
      dim_(dim),
      loss_(loss) {}

void LandmarkBlock::allocate(int num_frames, int num_obs, int dim) {
    padding_idx_ = num_frames * dim;
    num_rows_ = num_obs * 2;
    eliminated_landmark_rank_ = 0;
    qr_performed_ = false;

    int padding_size = padding_idx_ % 4;
    int padding_cols = (padding_size == 0) ? 0 : 4 - padding_size;
    delay_idx_ = padding_idx_ + padding_cols;
    lm_idx_ = delay_idx_ + 1;
    res_idx_ = lm_idx_ + 1;

    storage_.resize(num_rows_, res_idx_ + 1);
    storage_.setZero();
}

void LandmarkBlock::linearize(
    const MarginalizedFeatureObservation& marginalized_observation, const State& state,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    storage_.setZero();

    TASSEL_ASSERT(marginalized_observation.feature != nullptr);
    const Feature& feature = *marginalized_observation.feature;
    const std::vector<FeaturePerFrame>& observations = feature.observations;
    TASSEL_ASSERT(!observations.empty());
    const auto target_observation =
        std::find_if(observations.begin(), observations.end(), [&](const auto& observation) {
            return observation.frame_id == marginalized_observation.target_frame_id;
        });
    TASSEL_ASSERT(target_observation != observations.end());
    TASSEL_ASSERT(target_observation->frame_id != feature.host_frame_id);

    const Eigen::Vector3d uv_i = observations[0].uv;
    const int host_slot = state.findFrameSlot(feature.host_frame_id);
    const int target_slot = state.findFrameSlot(target_observation->frame_id);
    TASSEL_ASSERT(host_slot >= 0);
    TASSEL_ASSERT(target_slot >= 0);
    double depth = feature.estimated_depth;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian_pose_i, jacobian_pose_j;
    Eigen::Matrix<double, 2, 1> jacobian_dt;
    Eigen::Matrix<double, 2, 1> jacobian_landmark;
    Eigen::Matrix<double, 2, 1> residual;

    Eigen::Matrix2d sqrt_info = state.visual_sqrt_info;
    double inv_depth = 1.0 / depth;
    Eigen::Vector2d pt_j(target_observation->pt.x, target_observation->pt.y);
    VisualFactor visual_factor(
        uv_i, pt_j, ric, tic, state.gyro_vec[host_slot], state.gyro_vec[target_slot],
        state.acc_vec[host_slot], state.acc_vec[target_slot],
        state.params_speed_bias[host_slot].data(), state.params_speed_bias[target_slot].data(),
        state.params_speed_bias[host_slot].data() + 6,
        state.params_speed_bias[target_slot].data() + 6,
        state.params_speed_bias[host_slot].data() + 3,
        state.params_speed_bias[target_slot].data() + 3, sqrt_info, state.camera,
        observations[0].applied_delay, target_observation->applied_delay);

    std::vector<double*> jacobians = {
        jacobian_pose_i.data(), jacobian_pose_j.data(), jacobian_dt.data(),
        jacobian_landmark.data()};
    std::vector<double const*> parameters = {
        state.params_pose[host_slot].data(), state.params_pose[target_slot].data(),
        &state.param_delay_time, &inv_depth};
    visual_factor.Evaluate(parameters.data(), residual.data(), jacobians.data());
    jacobian_pose_i.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(-Eigen::Vector3d(
        state.params_pose[host_slot][3], state.params_pose[host_slot][4],
        state.params_pose[host_slot][5]));
    jacobian_pose_j.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(-Eigen::Vector3d(
        state.params_pose[target_slot][3], state.params_pose[target_slot][4],
        state.params_pose[target_slot][5]));

    double scale = 1.0;
    if (loss_) {
        double rho[3];
        loss_->Evaluate(residual.squaredNorm(), rho);
        scale = std::sqrt(rho[1]);
    }

    storage_.block<2, 6>(0, host_slot * dim_) = scale * jacobian_pose_i;
    storage_.block<2, 6>(0, target_slot * dim_) = scale * jacobian_pose_j;
    storage_.block<2, 1>(0, delay_idx_) = scale * jacobian_dt;
    storage_.block<2, 1>(0, lm_idx_) = scale * jacobian_landmark;
    storage_.block<2, 1>(0, res_idx_) = scale * residual;
}

void LandmarkBlock::performQR() {
    int n = num_rows_;
    eliminated_landmark_rank_ = 0;
    qr_performed_ = true;
    if (n <= 0) return;

    Eigen::VectorXd v = storage_.col(lm_idx_).head(n);
    double norm = v.norm();
    if (norm < 1e-12) return;
    eliminated_landmark_rank_ = 1;
    if (n == 1) return;

    double alpha = (v(0) > 0) ? -norm : norm;
    v(0) -= alpha;
    double beta = 2.0 / v.squaredNorm();

    for (int j = 0; j <= res_idx_; ++j) {
        double gamma = beta * v.dot(storage_.col(j).head(n));
        storage_.col(j).head(n) -= gamma * v;
    }
}

void LandmarkBlock::get_dense_Q2Jp_Q2r(
    Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, int start_row) const {
    const int kept_rows = get_kept_rows();
    if (kept_rows <= 0) {
        return;
    }
    Q2r.segment(start_row, kept_rows) =
        storage_.col(res_idx_).segment(eliminated_landmark_rank_, kept_rows);
    Q2Jp.block(start_row, 0, kept_rows, padding_idx_) =
        storage_.block(eliminated_landmark_rank_, 0, kept_rows, padding_idx_);
    Q2Jp.col(Q2Jp.cols() - 1).segment(start_row, kept_rows) =
        storage_.col(delay_idx_).segment(eliminated_landmark_rank_, kept_rows);
}

}  // namespace tassel_core
