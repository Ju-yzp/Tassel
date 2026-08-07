#ifndef TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_
#define TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_

#include <ceres/evaluation_callback.h>
#include <ceres/internal/config.h>
#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "state/frame_kinematics.h"
#include "tassel_utils/types.h"

namespace tassel_core {

struct VisualFrameCacheInput {
    const double* pose = nullptr;
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    double sync_delay = 0.0;
};

struct VisualFrameCacheEntry {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rotation_parameter_jacobian = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d delta_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d world_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d inverse_compensated_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d camera_inverse_compensated_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d compensated_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d position_delay_jacobian = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d body_rotational_acceleration = Eigen::Vector3d::Zero();
    double dt = 0.0;
};

struct VisualPairCacheEntry {
    Eigen::Matrix3d relative_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d camera_relative_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d relative_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d camera_relative_translation = Eigen::Vector3d::Zero();
    // 将两端 pose ambient 增量映射为 T_Ct_Ch 的左扰动，列布局均为 [dP, dtheta]。
    Eigen::Matrix<double, 6, 6> host_pose_jacobian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> target_pose_jacobian = Eigen::Matrix<double, 6, 6>::Zero();
};

class VisualFrameCache final : public ceres::EvaluationCallback {
public:
    VisualFrameCache(
        std::vector<VisualFrameCacheInput> inputs, const double* delay_time,
        const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic)
        : inputs_(std::move(inputs)), delay_time_(delay_time), ric_(ric), tic_(tic) {
        if (inputs_.empty()) {
            throw std::invalid_argument("Visual frame cache requires at least one frame");
        }
        if (!delay_time_) {
            throw std::invalid_argument("Visual frame cache delay parameter must not be null");
        }
        for (const auto& input : inputs_) {
            if (!input.pose) {
                throw std::invalid_argument("Visual frame cache pose must not be null");
            }
            if (!input.velocity.allFinite() || !input.gyro.allFinite() ||
                !input.acceleration.allFinite() || !input.gyro_bias.allFinite() ||
                !input.accel_bias.allFinite() || !std::isfinite(input.sync_delay)) {
                throw std::invalid_argument("Visual frame cache input must be finite");
            }
        }
        entries_.resize(inputs_.size());
        const size_t pair_count = inputs_.size() * inputs_.size();
        pair_entries_.resize(pair_count);
        pair_active_.assign(pair_count, false);
#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
        accepted_entries_.resize(inputs_.size());
        accepted_pair_entries_.resize(pair_count);
#endif
    }

    void addPair(int host_index, int target_index) {
        const size_t index = pairIndex(host_index, target_index);
        if (!pair_active_[index]) {
            pair_active_[index] = true;
            active_pairs_.emplace_back(host_index, target_index);
        }
    }

    void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override {
        if (new_evaluation_point || !values_valid_) {
            updateValues();
            values_valid_ = true;
            jacobians_valid_ = false;
        }
        if (evaluate_jacobians && !jacobians_valid_) {
            for (size_t i = 0; i < inputs_.size(); ++i) {
                const Eigen::Map<const Eigen::Vector3d> phi(inputs_[i].pose + 3);
                entries_[i].rotation_parameter_jacobian = Sophus::SO3d::leftJacobian(-phi);
            }
            updatePairJacobians();
            jacobians_valid_ = true;
        }
#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
        if (!accepted_values_valid_) {
            commitCurrentEvaluation();
        }
#endif
    }

#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
    void OnEvaluationAccepted() override { commitCurrentEvaluation(); }

    void OnEvaluationRejected() override { restoreAcceptedEvaluation(); }
#endif

    const VisualFrameCacheEntry& frame(int frame_index, bool require_jacobian = false) const {
        if (frame_index < 0 || frame_index >= static_cast<int>(entries_.size()) || !values_valid_) {
            throw std::logic_error("Visual frame cache entry is unavailable");
        }
        if (require_jacobian && !jacobians_valid_) {
            throw std::logic_error("Visual frame cache Jacobian is unavailable");
        }
        return entries_[static_cast<size_t>(frame_index)];
    }

    const VisualPairCacheEntry& pair(
        int host_index, int target_index, bool require_jacobian = false) const {
        const size_t index = pairIndex(host_index, target_index);
        if (!values_valid_ || !pair_active_[index]) {
            throw std::logic_error("Visual frame-pair cache entry is unavailable");
        }
        if (require_jacobian && !jacobians_valid_) {
            throw std::logic_error("Visual frame-pair cache Jacobian is unavailable");
        }
        return pair_entries_[index];
    }

private:
    size_t pairIndex(int host_index, int target_index) const {
        const int count = static_cast<int>(inputs_.size());
        if (host_index < 0 || host_index >= count || target_index < 0 || target_index >= count) {
            throw std::out_of_range("Visual frame-pair index is outside the active window");
        }
        return static_cast<size_t>(host_index) * inputs_.size() + static_cast<size_t>(target_index);
    }

    void updateValues() {
        if (!std::isfinite(*delay_time_)) {
            throw std::runtime_error("Visual frame cache delay parameter must be finite");
        }
        const Eigen::Matrix3d ric_transpose = ric_.transpose();
        for (size_t i = 0; i < inputs_.size(); ++i) {
            const auto& input = inputs_[i];
            auto& entry = entries_[i];
            const Eigen::Map<const Eigen::Vector3d> position(input.pose);
            const Eigen::Map<const Eigen::Vector3d> phi(input.pose + 3);
            if (!position.allFinite() || !phi.allFinite()) {
                throw std::runtime_error("Visual frame cache pose parameter must be finite");
            }

            entry.position = position;
            entry.rotation = Sophus::SO3d::exp(phi).matrix();
            const FrameKinematics kinematics = propagateFrameKinematics(
                entry.rotation, position, input.velocity, input.gyro, input.acceleration,
                input.gyro_bias, input.accel_bias, *delay_time_ - input.sync_delay);
            entry.dt = kinematics.dt;
            entry.omega = kinematics.omega;
            entry.acceleration = kinematics.acceleration;
            entry.body_rotational_acceleration = kinematics.body_rotational_acceleration;
            entry.delta_rotation = kinematics.delta_rotation;
            entry.world_rotation = kinematics.rotation;
            entry.inverse_compensated_rotation = kinematics.inverse_rotation;
            entry.camera_inverse_compensated_rotation =
                ric_transpose * entry.inverse_compensated_rotation;

            const double dt2 = entry.dt * entry.dt;
            entry.compensated_position = kinematics.position;
            entry.position_delay_jacobian = kinematics.velocity;
        }

        for (const auto& [host_index, target_index] : active_pairs_) {
            const auto& host = entries_[static_cast<size_t>(host_index)];
            const auto& target = entries_[static_cast<size_t>(target_index)];
            auto& pair_entry = pair_entries_[pairIndex(host_index, target_index)];
            pair_entry.relative_rotation =
                target.inverse_compensated_rotation * host.world_rotation;
            pair_entry.relative_translation =
                target.inverse_compensated_rotation *
                (host.compensated_position - target.compensated_position);
            pair_entry.camera_relative_rotation =
                ric_transpose * pair_entry.relative_rotation * ric_;
            pair_entry.camera_relative_translation =
                ric_transpose *
                (pair_entry.relative_rotation * tic_ + pair_entry.relative_translation - tic_);
        }
    }

    void updatePairJacobians() {
        for (const auto& [host_index, target_index] : active_pairs_) {
            const auto& host = entries_[static_cast<size_t>(host_index)];
            const auto& target = entries_[static_cast<size_t>(target_index)];
            auto& pair_entry = pair_entries_[pairIndex(host_index, target_index)];

            const double host_dt2 = host.dt * host.dt;
            const double target_dt2 = target.dt * target.dt;
            const Eigen::Vector3d host_rotation_offset =
                0.5 * host.acceleration * host_dt2 +
                (1.0 / 6.0) * host.body_rotational_acceleration * host_dt2 * host.dt +
                host.delta_rotation * tic_;
            const Eigen::Vector3d target_rotation_offset =
                0.5 * target.acceleration * target_dt2 +
                (1.0 / 6.0) * target.body_rotational_acceleration * target_dt2 * target.dt +
                target.delta_rotation * tic_;
            const Eigen::Matrix3d host_rotation =
                target.camera_inverse_compensated_rotation * host.rotation;
            const Eigen::Matrix3d target_rotation =
                target.camera_inverse_compensated_rotation * target.rotation;

            auto& host_jacobian = pair_entry.host_pose_jacobian;
            host_jacobian.setZero();
            host_jacobian.topLeftCorner<3, 3>() = target.camera_inverse_compensated_rotation;
            host_jacobian.topRightCorner<3, 3>() =
                -host_rotation * Sophus::SO3d::hat(host_rotation_offset) +
                Sophus::SO3d::hat(pair_entry.camera_relative_translation) * host_rotation;
            host_jacobian.bottomRightCorner<3, 3>() = host_rotation;
            host_jacobian.rightCols<3>() *= host.rotation_parameter_jacobian;

            auto& target_jacobian = pair_entry.target_pose_jacobian;
            target_jacobian.setZero();
            target_jacobian.topLeftCorner<3, 3>() = -target.camera_inverse_compensated_rotation;
            target_jacobian.topRightCorner<3, 3>() =
                target_rotation * Sophus::SO3d::hat(target_rotation_offset);
            target_jacobian.bottomRightCorner<3, 3>() = -target_rotation;
            target_jacobian.rightCols<3>() *= target.rotation_parameter_jacobian;
        }
    }

#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
    void commitCurrentEvaluation() {
        if (!values_valid_) {
            throw std::logic_error("Visual frame cache cannot commit invalid values");
        }
        accepted_entries_ = entries_;
        accepted_pair_entries_ = pair_entries_;
        accepted_values_valid_ = values_valid_;
        accepted_jacobians_valid_ = jacobians_valid_;
    }

    void restoreAcceptedEvaluation() {
        if (!accepted_values_valid_) {
            values_valid_ = false;
            jacobians_valid_ = false;
            return;
        }
        entries_ = accepted_entries_;
        pair_entries_ = accepted_pair_entries_;
        values_valid_ = accepted_values_valid_;
        jacobians_valid_ = accepted_jacobians_valid_;
    }
#endif

    // Ceres 保证两次 PrepareForEvaluation 之间参数不变；输入指针覆盖整个 Problem 生命周期。
    // 自定义 Ceres 的 step event 表达 trial 点是否进入后验；active cache 只服务本次
    // Evaluate，accepted snapshot 表示上一次被优化器接受的线性化点。

    std::vector<VisualFrameCacheInput> inputs_;
    const double* delay_time_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    std::vector<VisualFrameCacheEntry> entries_;
    std::vector<VisualPairCacheEntry> pair_entries_;
    std::vector<bool> pair_active_;
    std::vector<std::pair<int, int>> active_pairs_;
    bool values_valid_ = false;
    bool jacobians_valid_ = false;
#if defined(CERES_HAS_EVALUATION_STEP_EVENTS)
    std::vector<VisualFrameCacheEntry> accepted_entries_;
    std::vector<VisualPairCacheEntry> accepted_pair_entries_;
    bool accepted_values_valid_ = false;
    bool accepted_jacobians_valid_ = false;
#endif
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_
