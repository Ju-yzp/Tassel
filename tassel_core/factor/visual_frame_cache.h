#ifndef TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_
#define TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_

#include <ceres/evaluation_callback.h>
#include <ceres/internal/config.h>
#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <cmath>
#include <stdexcept>

#include "state/state.h"

namespace tassel_core {

// 视觉缓存只负责在 Ceres 当前评估点刷新 State；帧对和 landmark 在因子中直接计算。
class VisualFrameCache final : public ceres::EvaluationCallback {
public:
    VisualFrameCache(State& state, const Eigen::Matrix3d& ric) : state_(&state), ric_(ric) {
        if (state.max_frame_count < 1) {
            throw std::invalid_argument("Visual frame cache requires at least one frame");
        }
    }

    void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override {
        if (new_evaluation_point || !state_->visual_values_valid) {
            updateFrames();
            state_->visual_values_valid = true;
            state_->visual_jacobians_valid = false;
        }
        if (evaluate_jacobians) {
            for (int i = 0; i < state_->max_frame_count; ++i) {
                auto& frame = state_->frames[i];
                const Eigen::Map<const Eigen::Vector3d> phi(frame.param_pose.data() + 3);
                frame.visual_rotation_parameter_jacobian = Sophus::SO3d::leftJacobian(-phi);
            }
            state_->visual_jacobians_valid = true;
        }
#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
        if (!accepted_state_) {
            accepted_state_ = *state_;
        }
#endif
    }

#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
    void OnEvaluationAccepted() override { accepted_state_ = *state_; }

    void OnEvaluationRejected() override {
        if (!accepted_state_) {
            state_->visual_values_valid = false;
            state_->visual_jacobians_valid = false;
            return;
        }
        restoreVisualFrames(*accepted_state_);
    }
#endif

private:
    void updateFrames() {
        const double time_delay = *state_->currentTimeDelay();
        if (!std::isfinite(time_delay)) {
            throw std::runtime_error("Visual frame cache delay parameter must be finite");
        }
        const Eigen::Matrix3d ric_transpose = ric_.transpose();
        for (int i = 0; i < state_->max_frame_count; ++i) {
            auto& frame = state_->frames[i];
            const Eigen::Map<const Eigen::Vector3d> position(frame.param_pose.data());
            const Eigen::Map<const Eigen::Vector3d> phi(frame.param_pose.data() + 3);
            const Eigen::Map<const Eigen::Vector3d> velocity(frame.param_speed_bias.data());
            const Eigen::Map<const Eigen::Vector3d> accel_bias(frame.param_speed_bias.data() + 3);
            const Eigen::Map<const Eigen::Vector3d> gyro_bias(frame.param_speed_bias.data() + 6);
            const double dt = time_delay - frame.image_sync_delay;
            if (!position.allFinite() || !phi.allFinite() || !velocity.allFinite() ||
                !accel_bias.allFinite() || !gyro_bias.allFinite() || !frame.imu_gyro.allFinite() ||
                !frame.imu_acc.allFinite() || !std::isfinite(dt)) {
                throw std::runtime_error("Visual frame cache state is invalid");
            }

            frame.visual_position = position;
            frame.visual_base_rotation = Sophus::SO3d::exp(phi).matrix();
            frame.visual_dt = dt;
            frame.visual_omega = frame.imu_gyro - gyro_bias;
            frame.visual_acceleration = frame.imu_acc - accel_bias;
            frame.visual_body_rotational_acceleration =
                Sophus::SO3d::hat(frame.visual_omega) * frame.visual_acceleration;
            frame.visual_delta_rotation = Sophus::SO3d::exp(frame.visual_omega * dt).matrix();
            const double dt2 = dt * dt;
            const Eigen::Vector3d world_acceleration =
                frame.visual_base_rotation * frame.visual_acceleration - tassel_utils::G;
            const Eigen::Vector3d world_rotational_acceleration =
                frame.visual_base_rotation * frame.visual_body_rotational_acceleration;
            frame.visual_compensated_position =
                position + velocity * dt + 0.5 * world_acceleration * dt2 +
                (1.0 / 6.0) * world_rotational_acceleration * dt2 * dt;
            frame.visual_position_delay_jacobian =
                velocity + world_acceleration * dt + 0.5 * world_rotational_acceleration * dt2;
            frame.visual_rotation = frame.visual_base_rotation * frame.visual_delta_rotation;
            frame.visual_inverse_rotation = frame.visual_rotation.transpose();
            frame.visual_camera_inverse_rotation = ric_transpose * frame.visual_inverse_rotation;
        }
    }

#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
    void restoreVisualFrames(const State& accepted) {
        for (int i = 0; i < state_->max_frame_count; ++i) {
            state_->frames[i].visual_rotation = accepted.frames[i].visual_rotation;
            state_->frames[i].visual_base_rotation = accepted.frames[i].visual_base_rotation;
            state_->frames[i].visual_rotation_parameter_jacobian =
                accepted.frames[i].visual_rotation_parameter_jacobian;
            state_->frames[i].visual_delta_rotation = accepted.frames[i].visual_delta_rotation;
            state_->frames[i].visual_inverse_rotation = accepted.frames[i].visual_inverse_rotation;
            state_->frames[i].visual_camera_inverse_rotation =
                accepted.frames[i].visual_camera_inverse_rotation;
            state_->frames[i].visual_position = accepted.frames[i].visual_position;
            state_->frames[i].visual_compensated_position =
                accepted.frames[i].visual_compensated_position;
            state_->frames[i].visual_position_delay_jacobian =
                accepted.frames[i].visual_position_delay_jacobian;
            state_->frames[i].visual_omega = accepted.frames[i].visual_omega;
            state_->frames[i].visual_acceleration = accepted.frames[i].visual_acceleration;
            state_->frames[i].visual_body_rotational_acceleration =
                accepted.frames[i].visual_body_rotational_acceleration;
            state_->frames[i].visual_dt = accepted.frames[i].visual_dt;
        }
    }
#endif

    State* state_;
    Eigen::Matrix3d ric_;
#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
    std::optional<State> accepted_state_;
#endif
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_
