#include "visual_factor.h"
#include <Eigen/Geometry>
#include <cmath>
#include <sophus/so3.hpp>

namespace tassel_core {

VisualFactor::VisualFactor(
    const Eigen::Vector3d& uv_i_, const Eigen::Vector2d& pt_j_, double depth_,
    const Eigen::Matrix3d& ric_, const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_,
    const Eigen::Vector3d& w_j_, const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_,
    const double* v_i_, const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_,
    const Eigen::Matrix2d& sqrt_info_, const Eigen::Vector3d& G_, const CameraBase* camera_)
    : uv_i(uv_i_),
      pt_j(pt_j_),
      depth(depth_),
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
      sqrt_info(sqrt_info_),
      G(G_),
      camera(camera_) {}

bool VisualFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    Eigen::Vector3d P_i(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d phi_i(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Matrix3d R_i = Sophus::SO3d::exp(phi_i).matrix();

    Eigen::Vector3d P_j(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d phi_j(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Matrix3d R_j = Sophus::SO3d::exp(phi_j).matrix();

    double dt = parameters[2][0];
    double inv_depth = parameters[3][0];

    Eigen::Vector3d V_i(v_i[0], v_i[1], v_i[2]);
    Eigen::Vector3d V_j(v_j[0], v_j[1], v_j[2]);
    Eigen::Vector3d bg_i(bg_i_lin[0], bg_i_lin[1], bg_i_lin[2]);
    Eigen::Vector3d bg_j(bg_j_lin[0], bg_j_lin[1], bg_j_lin[2]);
    Eigen::Matrix3d A_i = Sophus::SO3d::exp((w_i - bg_i) * dt).matrix();
    Eigen::Matrix3d A_j = Sophus::SO3d::exp((bg_j - w_j) * dt).matrix();
    Eigen::Vector3d pi_in_C = uv_i / inv_depth;
    Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
    Eigen::Vector3d pi_in_G = R_i * A_i * pi_in_I + P_i + V_i * dt + 0.5 * R_i * a_i * dt * dt;
    Eigen::Vector3d pj_in_I =
        A_j * R_j.transpose() * (pi_in_G - (P_j + V_j * dt + 0.5 * R_j * a_j * dt * dt));
    Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);

    double inv_z = 1.0 / pj_in_C.z();
    Eigen::Vector2d uv_pred_norm(pj_in_C.x() * inv_z, pj_in_C.y() * inv_z);
    Eigen::Vector2d uv_pred_pixel = camera->distort(uv_pred_norm);

    Eigen::Map<Eigen::Vector2d> r(residuals);
    r = sqrt_info * (uv_pred_pixel - pt_j);

    if (jacobians) {
        Eigen::MatrixXd H_dz_dzn, H_dz_dzeta;
        camera->get_jacobian(uv_pred_norm, H_dz_dzn, H_dz_dzeta);

        Eigen::Matrix<double, 2, 3> duv_dP;
        duv_dP << inv_z, 0, -pj_in_C.x() * inv_z * inv_z, 0, inv_z, -pj_in_C.y() * inv_z * inv_z;
        Eigen::Matrix<double, 2, 3> reduce = sqrt_info * H_dz_dzn * duv_dP;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.block<2, 3>(0, 0) = reduce * ric.transpose() * A_j * R_j.transpose();
            jacobian_pose_i.block<2, 3>(0, 3) = reduce * ric.transpose() * A_j * R_j.transpose() *
                                                (-R_i * Sophus::SO3d::hat(A_i * pi_in_I) -
                                                 0.5 * R_i * Sophus::SO3d::hat(a_i * dt * dt));
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            jacobian_pose_j.block<2, 3>(0, 0) = reduce * (-ric.transpose() * A_j * R_j.transpose());
            jacobian_pose_j.block<2, 3>(0, 3) =
                reduce * ric.transpose() * A_j *
                Sophus::SO3d::hat(
                    R_j.transpose() * (pi_in_G - (P_j + V_j * dt + 0.5 * R_j * a_j * dt * dt)));
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_dt(jacobians[2]);
            jacobian_dt = reduce * ric.transpose() *
                          (Sophus::SO3d::hat(bg_j - w_j) * pj_in_I +
                           A_j * R_j.transpose() *
                               (R_i * Sophus::SO3d::hat(w_i - bg_i) * A_i * pi_in_I + V_i - V_j));
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_inv_depth(jacobians[3]);
            jacobian_inv_depth = reduce * ric.transpose() * A_j * R_j.transpose() * R_i * A_i *
                                 ric * (-pi_in_C / inv_depth);
        }
    }
    return true;
}

}  // namespace tassel_core
