#ifndef TASSEL_CORE_FACTOR_VISUAL_REPROJECTION_H_
#define TASSEL_CORE_FACTOR_VISUAL_REPROJECTION_H_

#include <Eigen/Core>
#include <sophus/so3.hpp>

namespace tassel_core {

// 计算单位球切平面基 (2×3)，在 uv_j 处线性化
inline Eigen::Matrix<double, 2, 3> computeTangentBasis(const Eigen::Vector3d& uv) {
    Eigen::Vector3d a = uv.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp) {
        tmp << 1, 0, 0;
    }
    Eigen::Vector3d b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    Eigen::Vector3d b2 = a.cross(b1);
    Eigen::Matrix<double, 2, 3> tangent_base;
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
    return tangent_base;
}

// 计算单对观测的重投影雅各比和残差（无权重，纯几何）
// 位姿参数化: [φ_0 φ_1 φ_2 P_0 P_1 P_2]（旋转在前，平移在后）
// J_i, J_j 列顺序与 param_poses 对齐
inline bool computeVisualReprojection(
    const Eigen::Vector3d& uv_i, const Eigen::Vector3d& uv_j, const Eigen::Matrix3d& R_i,
    const Eigen::Vector3d& P_i, const Eigen::Matrix3d& R_j, const Eigen::Vector3d& P_j,
    double depth, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    Eigen::Matrix<double, 2, 6>& J_i, Eigen::Matrix<double, 2, 6>& J_j,
    Eigen::Matrix<double, 2, 1>& J_l, Eigen::Vector2d& residual) {
    Eigen::Matrix<double, 2, 3> tangent_base = computeTangentBasis(uv_j);

    // 重投影
    Eigen::Vector3d pi_in_H = uv_i * depth;
    Eigen::Vector3d pi_in_I = ric * pi_in_H + tic;
    Eigen::Vector3d pi_in_W = R_i * pi_in_I + P_i;
    Eigen::Vector3d pj_in_I = R_j.transpose() * (pi_in_W - P_j);
    Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);

    double norm = pj_in_C.norm();
    residual = tangent_base * (pj_in_C / norm - uv_j);

    Eigen::Matrix<double, 2, 3> reduce =
        1.0 / norm * tangent_base *
        (Eigen::Matrix3d::Identity() - (pj_in_C * pj_in_C.transpose()) / (norm * norm));

    // 对 host 位姿 (φ_i, P_i) 的雅各比
    J_i.block<2, 3>(0, 0) =
        reduce * (-ric.transpose() * R_j.transpose() * R_i * Sophus::SO3d::hat(pi_in_I));
    J_i.block<2, 3>(0, 3) = reduce * (ric.transpose() * R_j.transpose());

    // 对 target 位姿 (φ_j, P_j) 的雅各比
    J_j.block<2, 3>(0, 0) = reduce * (ric.transpose() * Sophus::SO3d::hat(pj_in_I));
    J_j.block<2, 3>(0, 3) = reduce * (-ric.transpose() * R_j.transpose());

    // 对逆深度的雅各比
    double inv_depth = 1.0 / depth;
    J_l = reduce *
          (-ric.transpose() * R_j.transpose() * R_i * ric * (uv_i / (inv_depth * inv_depth)));

    return true;
}

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_VISUAL_REPROJECTION_H_
