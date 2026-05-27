#include "visual_factor.h"
#include "visual_reprojection.h"

#include <Eigen/Geometry>
#include <cmath>
#include <sophus/so3.hpp>

namespace tassel_core {

VisualFactor::VisualFactor(
    const Eigen::Vector3d& uv_i_, const Eigen::Vector3d& uv_j_, const Eigen::Matrix3d& ric_,
    const Eigen::Vector3d& tic_, double min_depth)
    : uv_i(uv_i_), uv_j(uv_j_), ric(ric_), tic(tic_), min_depth_(min_depth) {}

bool VisualFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    Eigen::Vector3d phi_h(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d P_h(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Matrix3d R_h = Sophus::SO3d::exp(phi_h).matrix();

    Eigen::Vector3d phi_t(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d P_t(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Matrix3d R_t = Sophus::SO3d::exp(phi_t).matrix();

    double inv_depth = parameters[2][0];
    double depth = 1.0 / inv_depth;

    // 共享雅各比计算
    Eigen::Matrix<double, 2, 6> J_i, J_j;
    Eigen::Matrix<double, 2, 1> J_l;
    Eigen::Vector2d r_raw;
    computeVisualReprojection(
        uv_i, uv_j, R_h, P_h, R_t, P_t, depth, ric, tic, J_i, J_j, J_l, r_raw);

    Eigen::Map<Eigen::Vector2d> r(residuals);
    r = r_raw;
    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_H(jacobians[0]);
            jacobian_H = J_i;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_T(jacobians[1]);
            jacobian_T = J_j;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_L(jacobians[2]);
            jacobian_L = J_l;
        }
    }
    return true;
}

// void jacobian() {
//     Eigen::Matrix3d ric;    // 相机到imu的旋转
//     Eigen::Matrix3d R_G_I;  // imu到全局坐标系的旋转
//     Eigen::Vector3d w;      // 全局坐标系下的角速度
//     Eigen::Vector3d v;      // 全局坐标系下线速度
//     Eigen::Vector3d a;      // 体坐标系下加速度
//     double dt;              // 时间微小量
//     Eigen::Vector3d p_G;    // 特征点在全局坐标系中的位置
//     Eigen::Vector3d P_G_I;  // imu在全局坐标系的位置

//     Eigen::Matrix<double, 2, 3> reduce;
//     Eigen::Vector2d jacobian_delay_t =
//         reduce * ric.transpose() * -1.0 *
//         (Sophus::SO3d::hat(w) * R_G_I.transpose() * p_G -
//          Sophus::SO3d::hat(w) * R_G_I.transpose() * P_G_I +
//          R_G_I.transpose() *
//              (v + (R_G_I * Sophus::SO3d::exp(w * dt).matrix() * a + R_G_I * a) * dt));

//     // 我们现在有两个相机和imu姿态，现在不使用特征点的全局坐标，而是host的相机坐标系下的坐标
//     Eigen::Vector3d pt_i_C;
//     Eigen::Vector3d pt_j_C;
//     // imu全局位姿
//     Eigen::Vector3d P_G_I_i, P_G_I_j;
//     Eigen::Matrix3d R_G_I_i, R_G_I_j;
//     Eigen::Vector3d a_i, a_j;  // 在两个时刻imu的体坐标系下加速度
//     Eigen::Vector3d w_i, w_j;  // 在两个时刻imu的体坐标系下角速度
//     Eigen::Vector3d v_i, v_j;  // 在两个时刻imu的全局坐标下的线速度
//     Eigen::Matrix3d ric_;      // 外参
//     Eigen::Vector3d tic_;      // 外参
//     Eigen::Vector2d jacobian_delay_t_ =
//         reduce * ric_.transpose() * -1.0 *
//         (Sophus::SO3d::hat(w_j) * R_G_I_j.transpose() *
//              (R_G_I_i * (ric_ * pt_i_C + tic_) + P_G_I_i) -
//          R_G_I_j.transpose() *
//              (R_G_I_i * Sophus::SO3d::hat(w_i) * (ric_ * pt_i_C + tic_) + v_i +
//               (R_G_I_i * Sophus::SO3d::exp(w_i * dt).matrix() * a_i + R_G_I_i * a_i) * dt) -
//          Sophus::SO3d::hat(w_j) * R_G_I_j.transpose() * P_G_I_j +
//          R_G_I_j.transpose() *
//              (v_j + (R_G_I_j * Sophus::SO3d::exp(w_j * dt).matrix() * a_j + R_G_I_j * a_j) *
//              dt));
// }
}  // namespace tassel_core
