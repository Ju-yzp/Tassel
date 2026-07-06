#ifndef TASSEL_CORE_INITIAL_INITIAL_ALIGNMENT_H_
#define TASSEL_CORE_INITIAL_INITIAL_ALIGNMENT_H_

#include <Eigen/Core>
#include <vector>

namespace tassel_core {

// 求解体坐标系速度 V、世界系重力 g、尺度 s
// Pj = Pi + Vi*dt + 0.5*g*dt^2 + Ri*dp
// Vj = Vi + g*dt + Ri*dv
bool linearAlignment(
    std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps,
    std::vector<Eigen::Vector3d>& Vs, const std::vector<Eigen::Vector3d>& delta_v,
    const std::vector<Eigen::Vector3d>& delta_p, const std::vector<double>& dt,
    Eigen::Vector3d& final_g, double& s, const Eigen::Matrix3d ric, const Eigen::Vector3d tic,
    double g_norm_thres, double target_g_norm);

// 重力方向 (2-DOF) + 速度 + 尺度的线性迭代精化，固定模长 g_mag
// Rs/Ps: 相机系姿态与位置 (C0 参考系), Vs: IMU 体系速度, ric: R_I_C
void refineGravitySpeeds(
    std::vector<Eigen::Vector3d>& Vs, const std::vector<Eigen::Matrix3d>& Rs,
    const std::vector<Eigen::Vector3d>& Ps, const std::vector<Eigen::Vector3d>& delta_vs,
    const std::vector<Eigen::Vector3d>& delta_ps, const std::vector<double>& dts,
    Eigen::Vector3d& G, double& s, const Eigen::Matrix3d ric, const Eigen::Vector3d tic,
    double g_mag);

// 陀螺偏置求解
Eigen::Vector3d solveGyroBias(
    std::vector<Eigen::Matrix3d> Rs, std::vector<Eigen::Matrix3d> dq_dbgs,
    std::vector<Eigen::Matrix3d> delta_qs, Eigen::Matrix3d ric);

}  // namespace tassel_core

#endif  // TASSEL_CORE_INITIAL_INITIAL_ALIGNMENT_H_
