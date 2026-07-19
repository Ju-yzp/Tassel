#include "TwoViewFactor.h"

#include <sophus/so3.hpp>

#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
TwoViewFactor::TwoViewFactor(
    const Eigen::Vector3d& uv_i_, const Eigen::Vector3d& uv_j_, const Eigen::Matrix3d& ric_,
    const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
    const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
    const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_, const double* ba_i_lin_,
    const double* ba_j_lin_, const double sqrt_info_)
    : uv_i(uv_i_),
      uv_j(uv_j_),
      ric(ric_),
      tic(tic_),
      w_i(w_i_),
      w_j(w_j_),
      a_i(a_i_),
      a_j(a_j_),
      v_i(Eigen::Map<const Eigen::Vector3d>(v_i_)),
      v_j(Eigen::Map<const Eigen::Vector3d>(v_j_)),
      bg_i_lin(Eigen::Map<const Eigen::Vector3d>(bg_i_lin_)),
      bg_j_lin(Eigen::Map<const Eigen::Vector3d>(bg_j_lin_)),
      ba_i_lin(Eigen::Map<const Eigen::Vector3d>(ba_i_lin_)),
      ba_j_lin(Eigen::Map<const Eigen::Vector3d>(ba_j_lin_)),
      sqrt_info(sqrt_info_) {}

bool TwoViewFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    Eigen::Vector3d P_i(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d phi_i(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Matrix3d R_i = Sophus::SO3d::exp(phi_i).matrix();

    Eigen::Vector3d P_j(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d phi_j(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Matrix3d R_j = Sophus::SO3d::exp(phi_j).matrix();

    double offset_time = parameters[2][0];

    const Eigen::Vector3d& V_i = v_i;
    const Eigen::Vector3d& V_j = v_j;
    const Eigen::Vector3d& bg_i = bg_i_lin;
    const Eigen::Vector3d& bg_j = bg_j_lin;
    const Eigen::Vector3d& ba_i = ba_i_lin;
    const Eigen::Vector3d& ba_j = ba_j_lin;

    Eigen::Matrix3d Ri_I_in_G = R_i * Sophus::SO3d::exp((w_i - bg_i) * offset_time).matrix();
    Eigen::Matrix3d Rj_I_in_G = R_j * Sophus::SO3d::exp((w_j - bg_j) * offset_time).matrix();
    Eigen::Matrix3d Ri_C_in_G = Ri_I_in_G * ric;
    Eigen::Matrix3d Rj_C_in_G = Rj_I_in_G * ric;

    Eigen::Vector3d Pi_I_in_G =
        P_i + V_i * offset_time + 0.5 * R_i * (a_i - ba_i) * offset_time * offset_time;
    Eigen::Vector3d Pj_I_in_G =
        P_j + V_j * offset_time + 0.5 * R_j * (a_j - ba_j) * offset_time * offset_time;

    Eigen::Vector3d Pi_C_in_G = Pi_I_in_G + Ri_I_in_G * tic;
    Eigen::Vector3d Pj_C_in_G = Pj_I_in_G + Rj_I_in_G * tic;
    Eigen::Vector3d delta_p_ij = Pj_C_in_G - Pi_C_in_G;
    if (!delta_p_ij.allFinite() || delta_p_ij.norm() < 1e-12) {
        return false;
    }
    Eigen::Vector3d direction = delta_p_ij.normalized();
    *residuals = uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * direction) *
                 Ri_C_in_G.transpose() * Rj_C_in_G * uv_j;

    if (jacobians) {
        const Eigen::Vector3d q_i = Ri_C_in_G.transpose() * Rj_C_in_G * uv_j;
        const Eigen::Matrix3d direction_projector =
            (Eigen::Matrix3d::Identity() - direction * direction.transpose()) / delta_p_ij.norm();
        const Eigen::RowVector3d residual_direction_jacobian =
            -uv_i.transpose() * Sophus::SO3d::hat(q_i) * Ri_C_in_G.transpose() *
            direction_projector;
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.block<1, 3>(0, 0) =
                -uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * Rj_C_in_G * uv_j) *
                Ri_C_in_G.transpose() *
                (-1.0 / delta_p_ij.norm() *
                 (Eigen::Matrix3d::Identity() - direction * direction.transpose()));
            jacobian_pose_i.block<1, 3>(0, 3) =
                uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * direction) *
                    ric.transpose() * Sophus::SO3d::exp((bg_i - w_i) * offset_time).matrix() *
                    Sophus::SO3d::hat(R_i.transpose() * Rj_C_in_G * uv_j) -
                uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * Rj_C_in_G * uv_j) *
                    ric.transpose() * Sophus::SO3d::exp((bg_i - w_i) * offset_time).matrix() *
                    Sophus::SO3d::hat(R_i.transpose() * direction);
            const Eigen::Matrix3d delta_position_rotation_i =
                R_i *
                (Sophus::SO3d::hat(Sophus::SO3d::exp((w_i - bg_i) * offset_time).matrix() * tic) +
                 0.5 * Sophus::SO3d::hat(a_i - ba_i) * offset_time * offset_time);
            jacobian_pose_i.block<1, 3>(0, 3) +=
                residual_direction_jacobian * delta_position_rotation_i;

            double minus_data[36];
            SE3RightManifold manifold;
            manifold.MinusJacobian(parameters[0], minus_data);
            const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(minus_data);
            jacobian_pose_i = jacobian_pose_i.eval() * minus;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_pose_j(jacobians[1]);
            jacobian_pose_j.block<1, 3>(0, 0) =
                -uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * Rj_C_in_G * uv_j) *
                Ri_C_in_G.transpose() *
                (1.0 / delta_p_ij.norm() *
                 (Eigen::Matrix3d::Identity() - direction * direction.transpose()));
            jacobian_pose_j.block<1, 3>(0, 3) =
                -uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * direction) *
                Ri_C_in_G.transpose() * R_j *
                Sophus::SO3d::hat(
                    Sophus::SO3d::exp((w_j - bg_j) * offset_time).matrix() * ric * uv_j);
            const Eigen::Matrix3d delta_position_rotation_j =
                -R_j *
                (Sophus::SO3d::hat(Sophus::SO3d::exp((w_j - bg_j) * offset_time).matrix() * tic) +
                 0.5 * Sophus::SO3d::hat(a_j - ba_j) * offset_time * offset_time);
            jacobian_pose_j.block<1, 3>(0, 3) +=
                residual_direction_jacobian * delta_position_rotation_j;

            double minus_data[36];
            SE3RightManifold manifold;
            manifold.MinusJacobian(parameters[1], minus_data);
            const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(minus_data);
            jacobian_pose_j = jacobian_pose_j.eval() * minus;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 1, 1>> jacobian_dt(jacobians[2]);
            jacobian_dt =
                uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * direction) *
                    (ric.transpose() * Sophus::SO3d::hat((bg_i - w_i)) * Ri_I_in_G.transpose() *
                         Rj_C_in_G +
                     Ri_C_in_G.transpose() * R_j * Sophus::SO3d::hat(w_j - bg_j) *
                         Sophus::SO3d::exp((w_j - bg_j) * offset_time).matrix() * ric) *
                    uv_j -
                uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * Rj_C_in_G * uv_j) *
                    (Ri_C_in_G.transpose() *
                         (1.0 / delta_p_ij.norm() *
                          (Eigen::Matrix3d::Identity() - direction * direction.transpose())) *
                         (V_j - V_i + (R_j * (a_j - ba_j) - R_i * (a_i - ba_i)) * offset_time +
                          (Rj_I_in_G * Sophus::SO3d::hat(w_j - bg_j) -
                           Ri_I_in_G * Sophus::SO3d::hat(w_i - bg_i)) *
                              tic) +
                     ric.transpose() * Sophus::SO3d::hat((bg_i - w_i)) * Ri_I_in_G.transpose() *
                         direction);
        }

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[0]) *= sqrt_info;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[1]) *= sqrt_info;
        }
        if (jacobians[2]) {
            jacobians[2][0] *= sqrt_info;
        }
    }
    *residuals *= sqrt_info;
    return true;
}

}  // namespace tassel_core
