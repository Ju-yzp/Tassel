// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_core {
class CameraBase;

enum class FrameType {
    Unknown,
    KeyFrame,
    NonKeyFrame,
};

// 同时保存物理 current、Ceres 参数缓存和同一物理帧的冻结 FEJ 点。
struct FrameState {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    Eigen::Matrix3d rot_w_i = Eigen::Matrix3d::Identity();
    Eigen::Vector3d pos_w_i = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gyro = Eigen::Vector3d::Zero();
    double image_sync_delay = 0.0;
    // 参数布局固定为 pose [position(3), rotation_log(3)] 和 speed_bias
    // [velocity(3), accel_bias(3), gyro_bias(3)]。
    std::array<double, 6> param_pose{};
    std::array<double, 9> param_speed_bias{};
    std::array<double, 6> linearized_pose{};
    std::array<double, 9> linearized_speed_bias{};
    bool has_linearized = false;
    FrameType frame_type = FrameType::Unknown;

    // 视觉时间延迟补偿结果；由 Ceres 评估回调写入，供视觉因子共享读取。
    Eigen::Matrix3d visual_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d visual_base_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d visual_rotation_parameter_jacobian = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d visual_delta_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d visual_inverse_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d visual_camera_inverse_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d visual_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d visual_compensated_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d visual_position_delay_jacobian = Eigen::Vector3d::Zero();
    Eigen::Vector3d visual_omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d visual_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d visual_body_rotational_acceleration = Eigen::Vector3d::Zero();
    double visual_dt = 0.0;

    void writeCurrentParams() {
        const Eigen::Vector3d phi = Sophus::SO3d(rot_w_i).log();
        param_pose = {pos_w_i.x(), pos_w_i.y(), pos_w_i.z(), phi.x(), phi.y(), phi.z()};
        for (int d = 0; d < 3; ++d) {
            param_speed_bias[d] = vel_w[d];
            param_speed_bias[d + 3] = accel_bias[d];
            param_speed_bias[d + 6] = gyro_bias[d];
        }
    }

    void readCurrentParams() {
        pos_w_i = Eigen::Vector3d(param_pose[0], param_pose[1], param_pose[2]);
        rot_w_i = Sophus::SO3d::exp(Eigen::Vector3d(param_pose[3], param_pose[4], param_pose[5]))
                      .matrix();
        for (int d = 0; d < 3; ++d) {
            vel_w[d] = param_speed_bias[d];
            accel_bias[d] = param_speed_bias[d + 3];
            gyro_bias[d] = param_speed_bias[d + 6];
        }
    }

    void captureLinearization() {
        if (has_linearized) {
            return;
        }
        if (frame_id == tassel_utils::kInvalidFrameId) {
            throw std::logic_error("Cannot linearize an invalid frame");
        }
        linearized_pose = param_pose;
        linearized_speed_bias = param_speed_bias;
        has_linearized = true;
    }

    void clearLinearization() {
        linearized_pose.fill(0.0);
        linearized_speed_bias.fill(0.0);
        has_linearized = false;
    }

    // 新帧只继承上一已接受 current 的预测初值，不继承 frame_id 或 FEJ 身份。
    void seedFromPosterior(const FrameState& posterior) {
        *this = posterior;
        frame_id = tassel_utils::kInvalidFrameId;
        frame_type = FrameType::Unknown;
        clearLinearization();
    }

    // current 与 frozen FEJ 必须使用同一世界系；该变换只改变坐标表示，不改变物理状态身份。
    void transformGauge(
        const Eigen::Matrix3d& rotation, const Eigen::Vector3d& source_origin,
        const Eigen::Vector3d& target_origin) {
        pos_w_i = rotation * (pos_w_i - source_origin) + target_origin;
        rot_w_i = rotation * rot_w_i;
        vel_w = rotation * vel_w;
        if (has_linearized) {
            const Eigen::Vector3d linearized_position(
                linearized_pose[0], linearized_pose[1], linearized_pose[2]);
            const Eigen::Matrix3d linearized_rotation =
                Sophus::SO3d::exp(
                    Eigen::Vector3d(linearized_pose[3], linearized_pose[4], linearized_pose[5]))
                    .matrix();
            const Eigen::Vector3d transformed_position =
                rotation * (linearized_position - source_origin) + target_origin;
            const Eigen::Vector3d transformed_phi =
                Sophus::SO3d(rotation * linearized_rotation).log();
            const Eigen::Vector3d linearized_velocity(
                linearized_speed_bias[0], linearized_speed_bias[1], linearized_speed_bias[2]);
            const Eigen::Vector3d transformed_velocity = rotation * linearized_velocity;
            for (int axis = 0; axis < 3; ++axis) {
                linearized_pose[axis] = transformed_position[axis];
                linearized_pose[axis + 3] = transformed_phi[axis];
                linearized_speed_bias[axis] = transformed_velocity[axis];
            }
        }
        writeCurrentParams();
    }

    double* currentPose() { return param_pose.data(); }
    double* currentSpeedBias() { return param_speed_bias.data(); }

    const double* linearizedPose() const {
        requireLinearization();
        return linearized_pose.data();
    }

    const double* linearizedSpeedBias() const {
        requireLinearization();
        return linearized_speed_bias.data();
    }

private:
    void requireLinearization() const {
        if (!has_linearized) {
            throw std::logic_error("Frame linearization point is unavailable");
        }
    }
};

struct State {
    explicit State(int frame_count = 10) : max_frame_count(frame_count) {
        if (max_frame_count < 1) {
            throw std::runtime_error("max_frame_count must be greater than 0");
        }
        frames.resize(static_cast<size_t>(max_frame_count));
    }

    // 同步 current 参数，并只为第一次进入优化问题的有效状态捕获 FEJ。
    void prepareOptimization() {
        invalidateVisualState();
        bool has_valid_frame = false;
        for (FrameState& frame : frames) {
            frame.writeCurrentParams();
            if (frame.frame_id != tassel_utils::kInvalidFrameId) {
                frame.captureLinearization();
                has_valid_frame = true;
            }
        }
        param_time_delay = time_delay;
        if (has_valid_frame) {
            captureDelayLinearization();
        }
    }

    void acceptOptimization() {
        if (!std::isfinite(param_time_delay)) {
            throw std::logic_error("Optimized time delay is not finite");
        }
        for (const FrameState& frame : frames) {
            for (double value : frame.param_pose) {
                if (!std::isfinite(value)) {
                    throw std::logic_error("Optimized pose is not finite");
                }
            }
            for (double value : frame.param_speed_bias) {
                if (!std::isfinite(value)) {
                    throw std::logic_error("Optimized speed-bias is not finite");
                }
            }
        }
        invalidateVisualState();
        for (FrameState& frame : frames) {
            frame.readCurrentParams();
        }
        time_delay = param_time_delay;
    }

    // 搬移的是同一物理 frame_id，因此 current 和 linearized 必须一起保留。
    void copyFrame(int source_index, int target_index) {
        requireFrameIndex(source_index);
        requireFrameIndex(target_index);
        frames[static_cast<size_t>(target_index)] = frames[static_cast<size_t>(source_index)];
        invalidateVisualState();
    }

    void seedFrameFromPosterior(int source_index, int target_index) {
        requireFrameIndex(source_index);
        requireFrameIndex(target_index);
        frames[static_cast<size_t>(target_index)].seedFromPosterior(
            frames[static_cast<size_t>(source_index)]);
        invalidateVisualState();
    }

    // 时间延迟与窗口中的首个有效帧共享 FEJ 生命周期；普通优化不得覆盖。
    void captureDelayLinearization() {
        if (has_linearized_delay) {
            return;
        }
        if (!std::isfinite(param_time_delay)) {
            throw std::logic_error("Cannot capture a non-finite time-delay linearization point");
        }
        linearized_time_delay = param_time_delay;
        has_linearized_delay = true;
    }

    double* currentTimeDelay() { return &param_time_delay; }
    const double* currentTimeDelay() const { return &param_time_delay; }

    const double* linearizedTimeDelay() const {
        if (!has_linearized_delay) {
            throw std::logic_error("Time-delay linearization point is unavailable");
        }
        return &linearized_time_delay;
    }

    // 优化参数或窗口布局变化后，旧评估点的视觉中间状态不可继续使用。
    void invalidateVisualState() {
        visual_values_valid = false;
        visual_jacobians_valid = false;
    }

    State get_compensated_state() const {
        State compensated = *this;
        for (int frame_index = 0; frame_index <= latest_active_frame_index; ++frame_index) {
            const auto& frame = frames[static_cast<size_t>(frame_index)];
            auto& output = compensated.frames[static_cast<size_t>(frame_index)];
            const double dt = time_delay - frame.image_sync_delay;
            const Eigen::Vector3d omega = frame.imu_gyro - frame.gyro_bias;
            const Eigen::Vector3d acceleration = frame.imu_acc - frame.accel_bias;
            const Eigen::Matrix3d delta_rotation = Sophus::SO3d::exp(omega * dt).matrix();
            const Eigen::Matrix3d rotation = frame.rot_w_i * delta_rotation;
            const double dt2 = dt * dt;
            const Eigen::Vector3d body_rotational_acceleration =
                Sophus::SO3d::hat(omega) * acceleration;
            const Eigen::Vector3d world_acceleration =
                frame.rot_w_i * acceleration - tassel_utils::G;
            const Eigen::Vector3d world_rotational_acceleration =
                frame.rot_w_i * body_rotational_acceleration;

            // 从该帧实际采用的同步延迟传播到当前全局延迟，运动量均在世界系更新。
            output.rot_w_i = rotation;
            output.vel_w =
                frame.vel_w + world_acceleration * dt + 0.5 * world_rotational_acceleration * dt2;
            output.pos_w_i = frame.pos_w_i + frame.vel_w * dt + 0.5 * world_acceleration * dt2 +
                             (1.0 / 6.0) * world_rotational_acceleration * dt2 * dt;
        }
        return compensated;
    }

    void reset() {
        latest_active_frame_index = 0;
        frames.assign(static_cast<size_t>(max_frame_count), FrameState{});
        time_delay = 0.0;
        param_time_delay = 0.0;
        linearized_time_delay = 0.0;
        has_linearized_delay = false;
        invalidateVisualState();
    }

    std::vector<FrameState> frames;
    int max_frame_count;
    // 最新有效帧的索引；窗口填满时为 max_frame_count - 1。
    int latest_active_frame_index = 0;
    double time_delay = 0.0;
    double param_time_delay = 0.0;
    double linearized_time_delay = 0.0;
    bool has_linearized_delay = false;
    const CameraBase* camera = nullptr;
    Eigen::Matrix2d visual_sqrt_info = Eigen::Matrix2d::Identity();
    // 视觉中间状态只在对应 Ceres 评估点有效。
    bool visual_values_valid = false;
    bool visual_jacobians_valid = false;

private:
    void requireFrameIndex(int frame_index) const {
        if (frame_index < 0 || frame_index >= max_frame_count) {
            throw std::out_of_range("Frame index is outside the state window");
        }
    }
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
