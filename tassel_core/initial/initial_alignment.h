#ifndef TASSEL_CORE_INITIAL_INITIAL_ALIGNMENT_H_
#define TASSEL_CORE_INITIAL_INITIAL_ALIGNMENT_H_

#include <Eigen/Core>
#include <vector>

namespace tassel_core {

// 求解体坐标系速度 V、世界系重力 g、尺度 s
// Pj = Pi + Vi*dt + 0.5*g*dt^2 + Ri*dp
// Vj = Vi + g*dt + Ri*dv
bool linearAlignment(
    const std::vector<Eigen::Matrix3d>& rotations, const std::vector<Eigen::Vector3d>& positions,
    std::vector<Eigen::Vector3d>& velocities, const std::vector<Eigen::Vector3d>& delta_velocities,
    const std::vector<Eigen::Vector3d>& delta_positions, const std::vector<double>& dts,
    Eigen::Vector3d& gravity, double& scale, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double gravity_norm_tolerance, double target_gravity_norm);

// 重力方向 (2-DOF) + 速度 + 尺度的线性迭代精化，固定重力模长。
// rotations/positions 位于首相机参考系，velocities 位于 IMU 体系，ric 为 R_I_C。
bool refineGravitySpeeds(
    std::vector<Eigen::Vector3d>& velocities, const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Vector3d>& delta_velocities,
    const std::vector<Eigen::Vector3d>& delta_positions, const std::vector<double>& dts,
    Eigen::Vector3d& gravity, double& scale, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double gravity_norm);

// 返回相对预积分线性化点的陀螺偏置修正量，而非绝对偏置。
Eigen::Vector3d solveGyroBiasCorrection(
    const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Matrix3d>& rotation_bias_jacobians,
    const std::vector<Eigen::Matrix3d>& delta_rotations, const Eigen::Matrix3d& ric);

}  // namespace tassel_core

#endif  // TASSEL_CORE_INITIAL_INITIAL_ALIGNMENT_H_
