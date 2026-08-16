#include "landmark_block.h"
#include "factor/reprojection_factor.h"
#include "state/state.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

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
      marginalized_landmark_rank_(0),
      qr_performed_(false),
      dim_(dim),
      loss_(loss) {}

void LandmarkBlock::allocate(int num_frames, int num_obs, int dim) {
    padding_idx_ = num_frames * dim;
    num_rows_ = num_obs * 2;
    marginalized_landmark_rank_ = 0;
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
    const Feature& feature, int target_frame_index, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    storage_.setZero();
    if (!state.has_linearized_delay || !feature.has_linearized_depth) {
        throw std::logic_error("Visual frozen linearization point is unavailable");
    }
    const int host_frame_index = feature.host_frame_index;
    const int first_observation_index =
        target_frame_index < 0 ? 1 : target_frame_index - host_frame_index;
    const int last_observation_index = target_frame_index < 0
                                           ? static_cast<int>(feature.observations.size())
                                           : first_observation_index + 1;
    if (first_observation_index <= 0 ||
        last_observation_index > static_cast<int>(feature.observations.size()) ||
        (last_observation_index - first_observation_index) * 2 != num_rows_) {
        throw std::logic_error("Invalid visual linearization observation layout");
    }
    const Eigen::Vector3d uv_i = feature.observations[0].uv;
    const double current_inverse_depth = 1.0 / feature.estimated_depth;
    const double linearized_inverse_depth = 1.0 / feature.linearized_depth;
    const double current_delay = *state.getCurrentTimeDelay();
    const double linearized_delay = *state.getLinearizedTimeDelay();
    State linearized_state = state;
    linearized_state.param_time_delay = linearized_delay;
    linearized_state.time_delay = linearized_delay;
    linearized_state.invalidateVisualState();

    for (int observation_index = first_observation_index;
         observation_index < last_observation_index; ++observation_index) {
        const int target_frame = feature.observationFrameIndex(observation_index);
        const FeaturePerFrame& target_observation = feature.observations[observation_index];
        Frame& linearized_host = linearized_state.frames[host_frame_index];
        Frame& linearized_target = linearized_state.frames[target_frame];
        if (!linearized_host.has_linearized || !linearized_target.has_linearized) {
            throw std::logic_error("Visual frozen frame linearization point is unavailable");
        }
        linearized_host.param_pose = linearized_host.linearized_pose;
        linearized_host.param_speed_bias = linearized_host.linearized_speed_bias;
        linearized_target.param_pose = linearized_target.linearized_pose;
        linearized_target.param_speed_bias = linearized_target.linearized_speed_bias;
        linearized_host.paramToState();
        linearized_target.paramToState();

        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian_host;
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian_target;
        Eigen::Vector2d jacobian_delay;
        Eigen::Vector2d jacobian_landmark;
        Eigen::Vector2d current_residual;
        Eigen::Vector2d linearized_residual;
        const Eigen::Vector2d target_pixel(target_observation.pt.x, target_observation.pt.y);

        ReprojectionFactor current_factor(
            uv_i, target_pixel, ric, tic, state.frames[host_frame_index].imu_gyro,
            state.frames[target_frame].imu_gyro, state.frames[host_frame_index].imu_acc,
            state.frames[target_frame].imu_acc,
            state.frames[host_frame_index].param_speed_bias.data(),
            state.frames[target_frame].param_speed_bias.data(),
            state.frames[host_frame_index].param_speed_bias.data() + 6,
            state.frames[target_frame].param_speed_bias.data() + 6,
            state.frames[host_frame_index].param_speed_bias.data() + 3,
            state.frames[target_frame].param_speed_bias.data() + 3, state.visual_sqrt_info,
            state.camera, feature.observations[0].sync_delay, target_observation.sync_delay);
        const double* current_parameters[] = {
            state.frames[host_frame_index].param_pose.data(),
            state.frames[target_frame].param_pose.data(), state.getCurrentTimeDelay(),
            &current_inverse_depth};
        TASSEL_ASSERT(
            current_factor.Evaluate(current_parameters, current_residual.data(), nullptr));

        ReprojectionFactor linearized_factor(
            uv_i, target_pixel, ric, tic, linearized_host.imu_gyro, linearized_target.imu_gyro,
            linearized_host.imu_acc, linearized_target.imu_acc,
            linearized_host.param_speed_bias.data(), linearized_target.param_speed_bias.data(),
            linearized_host.param_speed_bias.data() + 6,
            linearized_target.param_speed_bias.data() + 6,
            linearized_host.param_speed_bias.data() + 3,
            linearized_target.param_speed_bias.data() + 3, state.visual_sqrt_info, state.camera,
            feature.observations[0].sync_delay, target_observation.sync_delay);
        double* linearized_jacobians[] = {
            jacobian_host.data(), jacobian_target.data(), jacobian_delay.data(),
            jacobian_landmark.data()};
        const double* linearized_parameters[] = {
            linearized_host.param_pose.data(), linearized_target.param_pose.data(),
            &linearized_delay, &linearized_inverse_depth};
        TASSEL_ASSERT(linearized_factor.Evaluate(
            linearized_parameters, linearized_residual.data(), linearized_jacobians));
        jacobian_host.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(-Eigen::Vector3d(
            linearized_host.param_pose[3], linearized_host.param_pose[4],
            linearized_host.param_pose[5]));
        jacobian_target.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(-Eigen::Vector3d(
            linearized_target.param_pose[3], linearized_target.param_pose[4],
            linearized_target.param_pose[5]));

        Eigen::Matrix<double, 6, 1> linearized_host_pose;
        linearized_host_pose =
            Eigen::Map<const Eigen::Matrix<double, 6, 1>>(linearized_host.linearized_pose.data());
        Eigen::Matrix<double, 6, 1> current_host_pose;
        current_host_pose = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(
            state.frames[host_frame_index].param_pose.data());
        Eigen::Matrix<double, 6, 1> linearized_target_pose;
        linearized_target_pose =
            Eigen::Map<const Eigen::Matrix<double, 6, 1>>(linearized_target.linearized_pose.data());
        Eigen::Matrix<double, 6, 1> current_target_pose;
        current_target_pose = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(
            state.frames[target_frame].param_pose.data());
        const Eigen::Matrix<double, 6, 1> host_delta =
            rightTangentDelta(linearized_host_pose, current_host_pose);
        const Eigen::Matrix<double, 6, 1> target_delta =
            rightTangentDelta(linearized_target_pose, current_target_pose);
        const Eigen::Vector2d fej_constant =
            current_residual - jacobian_host * host_delta - jacobian_target * target_delta -
            jacobian_delay * (current_delay - linearized_delay) -
            jacobian_landmark * (current_inverse_depth - linearized_inverse_depth);

        double scale = 1.0;
        if (loss_) {
            double rho[3];
            loss_->Evaluate(current_residual.squaredNorm(), rho);
            scale = std::sqrt(rho[1]);
        }
        const int row = (observation_index - first_observation_index) * 2;
        storage_.block<2, 6>(row, host_frame_index * dim_) = scale * jacobian_host;
        storage_.block<2, 6>(row, target_frame * dim_) = scale * jacobian_target;
        storage_.block<2, 1>(row, delay_idx_) = scale * jacobian_delay;
        storage_.block<2, 1>(row, lm_idx_) = scale * jacobian_landmark;
        storage_.block<2, 1>(row, res_idx_) = scale * fej_constant;
    }
}

void LandmarkBlock::marginalizeLandmark() {
    int n = num_rows_;
    marginalized_landmark_rank_ = 0;
    qr_performed_ = true;
    if (n <= 0) {
        return;
    }

    Eigen::VectorXd v = storage_.col(lm_idx_).head(n);
    double norm = v.norm();
    if (norm < 1e-12) {
        return;
    }
    // 用一次 Householder 变换将逆深度雅可比压缩到首行。
    marginalized_landmark_rank_ = 1;
    if (n == 1) {
        return;
    }

    double alpha = (v(0) > 0) ? -norm : norm;
    v(0) -= alpha;
    double beta = 2.0 / v.squaredNorm();

    for (int j = 0; j <= res_idx_; ++j) {
        double gamma = beta * v.dot(storage_.col(j).head(n));
        storage_.col(j).head(n) -= gamma * v;
    }
}

void LandmarkBlock::writeReducedSystem(
    Eigen::MatrixXd& jacobian, Eigen::VectorXd& residual, int start_row) const {
    const int kept_rows = get_kept_rows();
    if (kept_rows <= 0) {
        return;
    }
    residual.segment(start_row, kept_rows) =
        storage_.col(res_idx_).segment(marginalized_landmark_rank_, kept_rows);
    jacobian.block(start_row, 0, kept_rows, padding_idx_) =
        storage_.block(marginalized_landmark_rank_, 0, kept_rows, padding_idx_);
    jacobian.col(jacobian.cols() - 1).segment(start_row, kept_rows) =
        storage_.col(delay_idx_).segment(marginalized_landmark_rank_, kept_rows);
}

}  // namespace tassel_core
