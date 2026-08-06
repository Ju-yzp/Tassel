#ifndef TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_
#define TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_

#include <ceres/evaluation_callback.h>
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
    Eigen::Matrix3d camera_inverse_delta_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d compensated_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_rotation_origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d position_delay_jacobian = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d body_rotational_acceleration = Eigen::Vector3d::Zero();
    double dt = 0.0;
};

struct VisualPairCacheEntry {
    Eigen::Matrix3d relative_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d camera_relative_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d host_pose_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d relative_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d camera_relative_translation = Eigen::Vector3d::Zero();
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
            jacobians_valid_ = true;
        }
    }

    const VisualFrameCacheEntry& frame(int frame_index, bool require_jacobian = false) const {
        if (frame_index < 0 || frame_index >= static_cast<int>(entries_.size()) || !values_valid_) {
            throw std::logic_error("Visual frame cache entry is unavailable");
        }
        if (require_jacobian && !jacobians_valid_) {
            throw std::logic_error("Visual frame cache Jacobian is unavailable");
        }
        return entries_[static_cast<size_t>(frame_index)];
    }

    const VisualPairCacheEntry& pair(int host_index, int target_index) const {
        const size_t index = pairIndex(host_index, target_index);
        if (!values_valid_ || !pair_active_[index]) {
            throw std::logic_error("Visual frame-pair cache entry is unavailable");
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
            entry.camera_inverse_delta_rotation = ric_transpose * entry.delta_rotation.transpose();

            const double dt2 = entry.dt * entry.dt;
            entry.compensated_position = kinematics.position;
            entry.target_rotation_origin =
                entry.position + input.velocity * entry.dt - 0.5 * tassel_utils::G * dt2;
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
            pair_entry.host_pose_rotation =
                target.camera_inverse_compensated_rotation * host.rotation;
        }
    }

    // Ceres 保证两次 PrepareForEvaluation 之间参数不变；输入指针覆盖整个 Problem 生命周期。
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
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_VISUAL_FRAME_CACHE_H_
