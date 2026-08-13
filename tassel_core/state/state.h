// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <array>
#include <optional>
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

// 同时保存物理状态和优化参数缓存；进入和退出求解器时必须显式同步。
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
    // 参数缓存顺序固定为 [position(3), rotation_log(3)] 和 [velocity(3), accel_bias(3),
    // gyro_bias(3)]。
    std::array<double, 6> param_pose{};
    std::array<double, 9> param_speed_bias{};
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

    void stateToParam() {
        const Eigen::Vector3d phi = Sophus::SO3d(rot_w_i).log();
        param_pose = {pos_w_i.x(), pos_w_i.y(), pos_w_i.z(), phi.x(), phi.y(), phi.z()};
        for (int d = 0; d < 3; ++d) {
            param_speed_bias[d] = vel_w[d];
            param_speed_bias[d + 3] = accel_bias[d];
            param_speed_bias[d + 6] = gyro_bias[d];
        }
    }

    void paramToState() {
        pos_w_i = Eigen::Vector3d(param_pose[0], param_pose[1], param_pose[2]);
        rot_w_i = Sophus::SO3d::exp(Eigen::Vector3d(param_pose[3], param_pose[4], param_pose[5]))
                      .matrix();
        for (int d = 0; d < 3; ++d) {
            vel_w[d] = param_speed_bias[d];
            accel_bias[d] = param_speed_bias[d + 3];
            gyro_bias[d] = param_speed_bias[d + 6];
        }
    }
};

// 当前 retained 关键帧第一次完成优化后的自由度参考。
struct GaugeAnchor {
    tassel_utils::FrameId reference_frame_id = tassel_utils::kInvalidFrameId;
    Eigen::Matrix3d reference_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();
};

struct State {
    explicit State(int max_frame_count_ = 10) : max_frame_count(max_frame_count_) {
        if (max_frame_count < 1) {
            throw std::runtime_error("max_frame_count must be greater than 0");
        }
        frames.resize(max_frame_count);
    }

    void stateToParam(int frame_index) { frames[frame_index].stateToParam(); }
    void paramToState(int frame_index) { frames[frame_index].paramToState(); }

    void stateToParams() {
        invalidateVisualState();
        for (auto& frame : frames) {
            frame.stateToParam();
        }
        param_time_delay = time_delay;
    }

    void paramsToState() {
        invalidateVisualState();
        for (auto& frame : frames) {
            frame.paramToState();
        }
        time_delay = param_time_delay;
    }

    void copyFrameState(int source_index, int target_frame_index) {
        frames[target_frame_index] = frames[source_index];
        invalidateVisualState();
    }

    // 优化参数或窗口布局变化后，旧评估点的视觉中间状态不可继续使用。
    void invalidateVisualState() {
        visual_values_valid = false;
        visual_jacobians_valid = false;
    }

    void captureGauge(int frame_index) {
        if (frame_index < 0 || frame_index > latest_active_frame_index) {
            throw std::out_of_range("Gauge frame is outside the active window");
        }
        const FrameState& frame = frames[frame_index];
        if (frame.frame_id == tassel_utils::kInvalidFrameId || !frame.rot_w_i.allFinite() ||
            !frame.pos_w_i.allFinite()) {
            throw std::logic_error("Cannot capture gauge from an invalid frame");
        }
        gauge_reference = GaugeAnchor{frame.frame_id, frame.rot_w_i, frame.pos_w_i};
    }

    State get_compensated_state() const {
        State compensated = *this;
        for (int frame_index = 0; frame_index <= latest_active_frame_index; ++frame_index) {
            const auto& frame = frames[frame_index];
            auto& output = compensated.frames[frame_index];
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
        frames.assign(max_frame_count, FrameState{});
        time_delay = 0.0;
        param_time_delay = 0.0;
        gauge_reference.reset();
        invalidateVisualState();
    }

    std::vector<FrameState> frames;
    int max_frame_count;
    // 最新有效帧的索引；窗口填满时为 max_frame_count - 1。
    int latest_active_frame_index = 0;
    double time_delay = 0.0;
    double param_time_delay = 0.0;
    std::optional<GaugeAnchor> gauge_reference;
    const CameraBase* camera = nullptr;
    Eigen::Matrix2d visual_sqrt_info = Eigen::Matrix2d::Identity();
    // 视觉中间状态只在对应 Ceres 评估点有效。
    bool visual_values_valid = false;
    bool visual_jacobians_valid = false;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
