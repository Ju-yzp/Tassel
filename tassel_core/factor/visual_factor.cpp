#include "visual_factor.h"
#include "visual_reprojection.h"

#include <Eigen/Geometry>
#include <algorithm>
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

    // 深度权重
    double weight = 1.0 / (depth * depth + 1.0);
    double sqrt_weight = std::sqrt(weight);

    // 共享雅各比计算
    Eigen::Matrix<double, 2, 6> J_i, J_j;
    Eigen::Matrix<double, 2, 1> J_l;
    Eigen::Vector2d r_raw;
    computeVisualReprojection(
        uv_i, uv_j, R_h, P_h, R_t, P_t, depth, ric, tic, J_i, J_j, J_l, r_raw);

    Eigen::Map<Eigen::Vector2d> r(residuals);
    r = sqrt_weight * r_raw;

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_H(jacobians[0]);
            jacobian_H = sqrt_weight * J_i;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_T(jacobians[1]);
            jacobian_T = sqrt_weight * J_j;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_L(jacobians[2]);
            jacobian_L = sqrt_weight * J_l;
        }
    }
    return true;
}

}  // namespace tassel_core
