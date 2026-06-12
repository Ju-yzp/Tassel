#include "TwoViewFactor.h"

#include <sophus/so3.hpp>

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
      v_i(v_i_),
      v_j(v_j_),
      bg_i_lin(bg_i_lin_),
      bg_j_lin(bg_j_lin_),
      ba_i_lin(ba_i_lin_),
      ba_j_lin(ba_j_lin_),
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

    Eigen::Vector3d V_i(v_i[0], v_i[1], v_i[2]);
    Eigen::Vector3d V_j(v_j[0], v_j[1], v_j[2]);
    Eigen::Vector3d bg_i(bg_i_lin[0], bg_i_lin[1], bg_i_lin[2]);
    Eigen::Vector3d bg_j(bg_j_lin[0], bg_j_lin[1], bg_j_lin[2]);
    Eigen::Vector3d ba_i(ba_i_lin[0], ba_i_lin[1], ba_i_lin[2]);
    Eigen::Vector3d ba_j(ba_j_lin[0], ba_j_lin[1], ba_j_lin[2]);

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
    Eigen::Vector3d direction = delta_p_ij.normalized();
    *residuals = uv_i.transpose() * Sophus::SO3d::hat(Ri_C_in_G.transpose() * direction) *
                 Ri_C_in_G.transpose() * Rj_C_in_G * uv_j;

    if (jacobians) {
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
                         (V_j - V_i +
                          (Rj_I_in_G * Sophus::SO3d::hat(w_j - bg_j) -
                           Ri_I_in_G * Sophus::SO3d::hat(w_i - bg_i)) *
                              ric * tic) +
                     ric.transpose() * Sophus::SO3d::hat((bg_i - w_i)) * Ri_I_in_G.transpose() *
                         direction);
        }
    }
    return true;
}

}  // namespace tassel_core
