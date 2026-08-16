// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <memory>
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
struct Frame {
    virtual ~Frame() = default;

    virtual std::unique_ptr<Frame> clone() const { return std::make_unique<Frame>(*this); }

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

    void stateToParam() {
        const Eigen::Vector3d phi = Sophus::SO3d(rot_w_i).log();
        param_pose = {pos_w_i.x(), pos_w_i.y(), pos_w_i.z(), phi.x(), phi.y(), phi.z()};
        for (int d = 0; d < 3; ++d) {
            param_speed_bias[d] = vel_w[d];
            param_speed_bias[d + 3] = accel_bias[d];
            param_speed_bias[d + 6] = gyro_bias[d];
        }
        if (!has_linearized && frame_id != tassel_utils::kInvalidFrameId) {
            linearized_pose = param_pose;
            linearized_speed_bias = param_speed_bias;
            has_linearized = true;
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

    virtual double* getCurrentPose() { return param_pose.data(); }
    virtual double* getCurrentSpeed() { return param_speed_bias.data(); }
    virtual double* getCurrentAccelBias() { return param_speed_bias.data() + 3; }
    virtual double* getCurrentGyroBias() { return param_speed_bias.data() + 6; }

    virtual const double* getLinearizedPose() const {
        requireLinearized();
        return linearized_pose.data();
    }

    virtual const double* getLinearizedSpeed() const {
        requireLinearized();
        return linearized_speed_bias.data();
    }

    virtual const double* getLinearizedAccelBias() const {
        requireLinearized();
        return linearized_speed_bias.data() + 3;
    }

    virtual const double* getLinearizedGyroBias() const {
        requireLinearized();
        return linearized_speed_bias.data() + 6;
    }

    void resetLinearization() {
        linearized_pose.fill(0.0);
        linearized_speed_bias.fill(0.0);
        has_linearized = false;
    }

private:
    void requireLinearized() const {
        if (!has_linearized) {
            throw std::logic_error("Frame linearization point is unavailable");
        }
    }
};

struct NormalFrame final : Frame {
    std::unique_ptr<Frame> clone() const override { return std::make_unique<NormalFrame>(*this); }
};

struct KeyFrame final : Frame {
    std::unique_ptr<Frame> clone() const override { return std::make_unique<KeyFrame>(*this); }
};

struct RetainedFrame final : Frame {
    std::unique_ptr<Frame> clone() const override { return std::make_unique<RetainedFrame>(*this); }
};

class FrameStorage : public std::vector<std::unique_ptr<Frame>> {
public:
    FrameStorage() = default;

    explicit FrameStorage(size_t count) {
        this->reserve(count);
        for (size_t i = 0; i < count; ++i) {
            this->push_back(std::make_unique<NormalFrame>());
        }
    }

    FrameStorage(const FrameStorage& other) {
        this->reserve(other.size());
        for (const auto& frame : other) {
            if (!frame) {
                throw std::logic_error("Frame storage contains a null frame");
            }
            this->push_back(frame->clone());
        }
    }

    FrameStorage& operator=(const FrameStorage& other) {
        if (this != &other) {
            FrameStorage copy(other);
            std::vector<std::unique_ptr<Frame>>::operator=(std::move(copy));
        }
        return *this;
    }

    FrameStorage(FrameStorage&&) noexcept = default;
    FrameStorage& operator=(FrameStorage&&) noexcept = default;

    Frame& operator[](size_t index) {
        return *std::vector<std::unique_ptr<Frame>>::operator[](index);
    }

    const Frame& operator[](size_t index) const {
        return *std::vector<std::unique_ptr<Frame>>::operator[](index);
    }

    void replace(size_t index, const Frame& frame) {
        if (index >= this->size()) {
            throw std::out_of_range("Frame replacement index is outside storage");
        }
        std::vector<std::unique_ptr<Frame>>::operator[](index) = frame.clone();
    }

    void replaceType(size_t index, const Frame& source, bool key_frame) {
        if (index >= this->size()) {
            throw std::out_of_range("Frame type replacement index is outside storage");
        }
        std::unique_ptr<Frame> replacement;
        if (key_frame) {
            replacement = std::make_unique<KeyFrame>();
        } else {
            replacement = std::make_unique<NormalFrame>();
        }
        static_cast<Frame&>(*replacement) = source;
        std::vector<std::unique_ptr<Frame>>::operator[](index) = std::move(replacement);
    }
};

// 迁移期兼容旧算法签名；所有权容器完成迁移后删除该别名。
using FrameState = Frame;

struct State {
    explicit State(int max_frame_count_ = 10)
        : max_frame_count(max_frame_count_), retained_frame(std::make_unique<RetainedFrame>()) {
        if (max_frame_count < 1) {
            throw std::runtime_error("max_frame_count must be greater than 0");
        }
        frames = FrameStorage(static_cast<size_t>(max_frame_count));
    }

    State(const State& other)
        : frames(other.frames),
          max_frame_count(other.max_frame_count),
          latest_active_frame_index(other.latest_active_frame_index),
          time_delay(other.time_delay),
          param_time_delay(other.param_time_delay),
          linearized_time_delay(other.linearized_time_delay),
          has_linearized_delay(other.has_linearized_delay),
          retained_frame(
              other.retained_frame ? std::make_unique<RetainedFrame>(*other.retained_frame)
                                   : nullptr),
          camera(other.camera),
          visual_sqrt_info(other.visual_sqrt_info),
          visual_values_valid(other.visual_values_valid),
          visual_jacobians_valid(other.visual_jacobians_valid) {}

    State& operator=(const State& other) {
        if (this != &other) {
            State copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    RetainedFrame& retainedFrame() {
        if (!retained_frame) {
            throw std::logic_error("State has no retained frame");
        }
        return *retained_frame;
    }

    const RetainedFrame& retainedFrame() const {
        if (!retained_frame) {
            throw std::logic_error("State has no retained frame");
        }
        return *retained_frame;
    }

    void replaceRetainedFrame(const Frame& source) {
        auto replacement = std::make_unique<RetainedFrame>();
        static_cast<Frame&>(*replacement) = source;
        retained_frame = std::move(replacement);
    }

    void copyRetainedToFrame(int target_frame_index) {
        if (target_frame_index < 0 || target_frame_index >= max_frame_count) {
            throw std::out_of_range("Retained frame target is outside the active window");
        }
        frames.replace(static_cast<size_t>(target_frame_index), retainedFrame());
        invalidateVisualState();
    }

    void stateToParam(int frame_index) { frames[frame_index].stateToParam(); }
    void paramToState(int frame_index) { frames[frame_index].paramToState(); }

    void stateToParams() {
        invalidateVisualState();
        bool has_valid_frame = false;
        for (auto& frame_ptr : frames) {
            if (!frame_ptr) {
                throw std::logic_error("Frame storage contains a null frame");
            }
            Frame& frame = *frame_ptr;
            frame.stateToParam();
            if (frame.frame_id != tassel_utils::kInvalidFrameId) {
                has_valid_frame = true;
            }
        }
        param_time_delay = time_delay;
        if (has_valid_frame) {
            captureLinearizedTimeDelay();
        }
    }

    // 时间延迟与窗口中的首个有效帧共享 FEJ 生命周期；后续优化只更新 current 值。
    void captureLinearizedTimeDelay() {
        if (has_linearized_delay) {
            return;
        }
        if (!std::isfinite(param_time_delay)) {
            throw std::logic_error("Cannot capture a non-finite time-delay linearization point");
        }
        linearized_time_delay = param_time_delay;
        has_linearized_delay = true;
    }

    void paramsToState() {
        invalidateVisualState();
        for (auto& frame_ptr : frames) {
            if (!frame_ptr) {
                throw std::logic_error("Frame storage contains a null frame");
            }
            frame_ptr->paramToState();
        }
        time_delay = param_time_delay;
    }

    void copyFrameState(int source_index, int target_frame_index) {
        frames.replace(target_frame_index, frames[source_index]);
        invalidateVisualState();
    }

    // 新 frame_id 只继承上一后验的预测初值，不继承旧状态的 FEJ 身份。
    void seedFrameState(int source_index, int target_frame_index) {
        frames.replace(target_frame_index, frames[source_index]);
        frames[target_frame_index].resetLinearization();
        invalidateVisualState();
    }

    double* getCurrentTimeDelay() { return &param_time_delay; }
    const double* getCurrentTimeDelay() const { return &param_time_delay; }

    const double* getLinearizedTimeDelay() const {
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
        frames = FrameStorage(static_cast<size_t>(max_frame_count));
        retained_frame = std::make_unique<RetainedFrame>();
        time_delay = 0.0;
        param_time_delay = 0.0;
        linearized_time_delay = 0.0;
        has_linearized_delay = false;
        invalidateVisualState();
    }

    FrameStorage frames;
    int max_frame_count;
    // 最新有效帧的索引；窗口填满时为 max_frame_count - 1。
    int latest_active_frame_index = 0;
    double time_delay = 0.0;
    double param_time_delay = 0.0;
    double linearized_time_delay = 0.0;
    bool has_linearized_delay = false;
    std::unique_ptr<RetainedFrame> retained_frame;
    const CameraBase* camera = nullptr;
    Eigen::Matrix2d visual_sqrt_info = Eigen::Matrix2d::Identity();
    // 视觉中间状态只在对应 Ceres 评估点有效。
    bool visual_values_valid = false;
    bool visual_jacobians_valid = false;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
