#ifndef TASSEL_CORE_STATE_FRAME_KINEMATICS_H_
#define TASSEL_CORE_STATE_FRAME_KINEMATICS_H_

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include "tassel_utils/types.h"

namespace tassel_core {

struct FrameKinematics {
    Eigen::Matrix3d delta_rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d inverse_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d body_rotational_acceleration = Eigen::Vector3d::Zero();
    double dt = 0.0;
};

inline FrameKinematics propagateFrameKinematics(
    const Eigen::Matrix3d& rotation, const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity, const Eigen::Vector3d& gyro,
    const Eigen::Vector3d& acceleration, const Eigen::Vector3d& gyro_bias,
    const Eigen::Vector3d& accel_bias, double dt) {
    FrameKinematics result;
    result.dt = dt;
    result.omega = gyro - gyro_bias;
    result.acceleration = acceleration - accel_bias;
    result.body_rotational_acceleration = Sophus::SO3d::hat(result.omega) * result.acceleration;
    result.delta_rotation = Sophus::SO3d::exp(result.omega * dt).matrix();
    result.rotation = rotation * result.delta_rotation;
    result.inverse_rotation = result.rotation.transpose();

    const double dt2 = dt * dt;
    const Eigen::Vector3d world_acceleration = rotation * result.acceleration - tassel_utils::G;
    const Eigen::Vector3d world_rotational_acceleration =
        rotation * result.body_rotational_acceleration;
    // 常体坐标角速度和比力下，在当前帧线性化点对世界系平移作三阶展开。
    result.position = position + velocity * dt + 0.5 * world_acceleration * dt2 +
                      (1.0 / 6.0) * world_rotational_acceleration * dt2 * dt;
    result.velocity =
        velocity + world_acceleration * dt + 0.5 * world_rotational_acceleration * dt2;
    return result;
}

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_FRAME_KINEMATICS_H_
