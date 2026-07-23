// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <array>
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
    tassel_utils::FrameId timestamp_ns = tassel_utils::kInvalidFrameId;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d V = Eigen::Vector3d::Zero();
    Eigen::Vector3d Ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d Bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    double sync_delay = 0.0;
    std::array<double, 6> pose{};
    std::array<double, 9> speed_bias{};
    FrameType type = FrameType::Unknown;

    void stateToParam() {
        const Eigen::Vector3d phi = Sophus::SO3d(R).log();
        pose = {P.x(), P.y(), P.z(), phi.x(), phi.y(), phi.z()};
        for (int d = 0; d < 3; ++d) {
            speed_bias[d] = V[d];
            speed_bias[d + 3] = Ba[d];
            speed_bias[d + 6] = Bg[d];
        }
    }

    void paramToState() {
        P = Eigen::Vector3d(pose[0], pose[1], pose[2]);
        R = Sophus::SO3d::exp(Eigen::Vector3d(pose[3], pose[4], pose[5])).matrix();
        for (int d = 0; d < 3; ++d) {
            V[d] = speed_bias[d];
            Ba[d] = speed_bias[d + 3];
            Bg[d] = speed_bias[d + 6];
        }
    }
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
        for (auto& frame : frames) {
            frame.stateToParam();
        }
        param_delay_time = delay_time;
    }

    void paramsToState() {
        for (auto& frame : frames) {
            frame.paramToState();
        }
        delay_time = param_delay_time;
    }

    void copyFrameState(int source_index, int target_frame_index) {
        frames[target_frame_index] = frames[source_index];
    }

    State get_compensated_state() const {
        State compensated = *this;
        for (int frame_index = 0; frame_index <= latest_frame_index; ++frame_index) {
            const auto& frame = frames[frame_index];
            auto& output = compensated.frames[frame_index];
            const double dt = delay_time - frame.sync_delay;
            const Eigen::Vector3d omega = frame.gyro - frame.Bg;
            const Eigen::Vector3d acc_body = frame.acc - frame.Ba;
            const Eigen::Vector3d acc_world = frame.R * acc_body - tassel_utils::G;
            const Eigen::Vector3d rotational_acceleration =
                frame.R * Sophus::SO3d::hat(omega) * acc_body;

            // 从该帧实际采用的同步延迟传播到当前全局延迟，运动量均在世界系更新。
            output.R = frame.R * Sophus::SO3d::exp(omega * dt).matrix();
            output.V = frame.V + acc_world * dt + 0.5 * rotational_acceleration * dt * dt;
            output.P = frame.P + frame.V * dt + 0.5 * acc_world * dt * dt +
                       (1.0 / 6.0) * rotational_acceleration * dt * dt * dt;
        }
        return compensated;
    }

    void reset() {
        latest_frame_index = 0;
        frames.assign(max_frame_count, FrameState{});
        delay_time = 0.0;
        param_delay_time = 0.0;
    }

    std::vector<FrameState> frames;
    int max_frame_count;
    // 最新有效帧的索引；窗口填满时为 max_frame_count - 1。
    int latest_frame_index = 0;
    double delay_time = 0.0;
    double param_delay_time = 0.0;
    const CameraBase* camera = nullptr;
    Eigen::Matrix2d visual_sqrt_info = Eigen::Matrix2d::Identity();
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
