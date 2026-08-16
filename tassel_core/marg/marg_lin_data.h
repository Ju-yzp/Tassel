#ifndef TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_
#define TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <sophus/so3.hpp>

namespace tassel_core {

// 先验列紧凑排列为 [retained pose(6), frame1 pose+speed_bias(15), ..., delay(1)]；
// frame 0 是保留帧，不包含 speed-bias 先验列。
struct MargLinData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int PoseSize = 6;
    static constexpr int SpeedBiasSize = 9;
    static constexpr int StateSize = PoseSize + SpeedBiasSize;

    int stateCount() const { return static_cast<int>(linearization_poses.size()); }

    int poseColumn(int frame_index) const {
        requireFrame(frame_index);
        return frame_index == 0 ? 0 : PoseSize + (frame_index - 1) * StateSize;
    }

    int speedBiasColumn(int frame_index) const {
        requireFrame(frame_index);
        if (frame_index == 0) {
            throw std::logic_error("Retained prior frame has no speed-bias columns");
        }
        return poseColumn(frame_index) + PoseSize;
    }

    int delayColumn() const { return PoseSize + (stateCount() - 1) * StateSize; }

    int columnCount() const { return delayColumn() + 1; }

    void transformGauge(
        const Eigen::Matrix3d& rotation, const Eigen::Vector3d& source_origin,
        const Eigen::Vector3d& target_origin) {
        validate();
        if (!rotation.allFinite() || !source_origin.allFinite() || !target_origin.allFinite()) {
            throw std::invalid_argument("Prior gauge transform is not finite");
        }
        if (!(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-10) ||
            std::abs(rotation.determinant() - 1.0) > 1e-10) {
            throw std::invalid_argument("Prior gauge rotation is not in SO(3)");
        }

        // 位姿采用右旋转扰动，因此全局左乘旋转不改变姿态切空间；位置和速度增量在世界系，
        // 对应的先验列必须右乘旋转逆矩阵，才能保持 H * delta + b 不变。
        const Eigen::Matrix3d inverse_rotation = rotation.transpose();
        for (int frame_index = 0; frame_index < stateCount(); ++frame_index) {
            auto& pose = linearization_poses[static_cast<size_t>(frame_index)];
            const Eigen::Vector3d position(pose[0], pose[1], pose[2]);
            const Eigen::Matrix3d orientation =
                Sophus::SO3d::exp(Eigen::Vector3d(pose[3], pose[4], pose[5])).matrix();
            const Eigen::Vector3d transformed_position =
                rotation * (position - source_origin) + target_origin;
            const Eigen::Vector3d transformed_phi = Sophus::SO3d(rotation * orientation).log();
            for (int axis = 0; axis < 3; ++axis) {
                pose[axis] = transformed_position[axis];
                pose[axis + 3] = transformed_phi[axis];
            }
            H.block(0, poseColumn(frame_index), H.rows(), 3) *= inverse_rotation;

            auto& speed_bias = linearization_speed_bias[static_cast<size_t>(frame_index)];
            const Eigen::Vector3d velocity(speed_bias[0], speed_bias[1], speed_bias[2]);
            const Eigen::Vector3d transformed_velocity = rotation * velocity;
            for (int axis = 0; axis < 3; ++axis) {
                speed_bias[axis] = transformed_velocity[axis];
            }
            if (frame_index > 0) {
                H.block(0, speedBiasColumn(frame_index), H.rows(), 3) *= inverse_rotation;
            }
        }
    }

    void validate() const {
        if (stateCount() <= 0 ||
            static_cast<int>(linearization_speed_bias.size()) != stateCount() ||
            H.cols() != columnCount() || H.rows() != b.size()) {
            throw std::invalid_argument("Marginalization prior does not match the window state");
        }
    }

    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    std::vector<std::array<double, 6>> linearization_poses;
    std::vector<std::array<double, 9>> linearization_speed_bias;
    double linearization_delay_time = 0.0;

private:
    void requireFrame(int frame_index) const {
        if (frame_index < 0 || frame_index >= stateCount()) {
            throw std::out_of_range("Prior frame index is outside the active state");
        }
    }
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_
