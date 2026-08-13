// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#include "reprojection_factor.h"
#include <Eigen/Geometry>
#include <cmath>
#include <sophus/so3.hpp>
#include <stdexcept>

#include "state/state.h"
#include "tassel_utils/types.h"

namespace tassel_core {

ReprojectionFactor::ReprojectionFactor(
    const Eigen::Vector3d& uv_i_, const Eigen::Vector2d& pt_j_, const Eigen::Matrix3d& ric_,
    const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
    const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
    const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_, const double* ba_i_lin_,
    const double* ba_j_lin_, const Eigen::Matrix2d& sqrt_info_, const CameraBase* camera_,
    double sync_delay_i_, double sync_delay_j_, const State* state_, int host_frame_index_,
    int target_frame_index_)
    : uv_i(uv_i_),
      pt_j(pt_j_),
      ric(ric_),
      tic(tic_),
      w_i(w_i_),
      w_j(w_j_),
      a_i(a_i_),
      a_j(a_j_),
      v_i(v_i_[0], v_i_[1], v_i_[2]),
      v_j(v_j_[0], v_j_[1], v_j_[2]),
      bg_i(bg_i_lin_[0], bg_i_lin_[1], bg_i_lin_[2]),
      bg_j(bg_j_lin_[0], bg_j_lin_[1], bg_j_lin_[2]),
      ba_i(ba_i_lin_[0], ba_i_lin_[1], ba_i_lin_[2]),
      ba_j(ba_j_lin_[0], ba_j_lin_[1], ba_j_lin_[2]),
      sqrt_info(sqrt_info_),
      camera(camera_),
      sync_delay_i(sync_delay_i_),
      sync_delay_j(sync_delay_j_),
      state(state_),
      host_frame_index(host_frame_index_),
      target_frame_index(target_frame_index_) {
    if (state && (host_frame_index < 0 || target_frame_index < 0)) {
        throw std::invalid_argument("Cached reprojection factor requires valid frame indices");
    }
}

bool ReprojectionFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    if (state) {
        return evaluateCached(parameters[3][0], residuals, jacobians);
    }

    Eigen::Vector3d P_i(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d phi_i(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Matrix3d R_i = Sophus::SO3d::exp(phi_i).matrix();

    Eigen::Vector3d P_j(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d phi_j(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Matrix3d R_j = Sophus::SO3d::exp(phi_j).matrix();

    const double time_delay = parameters[2][0];
    const double dt_i = time_delay - sync_delay_i;
    const double dt_j = time_delay - sync_delay_j;
    double inv_depth = parameters[3][0];

    const Eigen::Vector3d omega_i = w_i - bg_i;
    const Eigen::Vector3d omega_j = w_j - bg_j;
    const Eigen::Vector3d acc_i = a_i - ba_i;
    const Eigen::Vector3d body_rot_acc_i = Sophus::SO3d::hat(omega_i) * acc_i;
    const Eigen::Vector3d body_rot_acc_j = Sophus::SO3d::hat(omega_j) * (a_j - ba_j);
    const Eigen::Matrix3d delta_rotation_i = Sophus::SO3d::exp(omega_i * dt_i).matrix();
    const Eigen::Matrix3d delta_rotation_j = Sophus::SO3d::exp(omega_j * dt_j).matrix();
    const Eigen::Matrix3d compensated_rotation_i = R_i * delta_rotation_i;
    const Eigen::Matrix3d inverse_compensated_rotation_j = (R_j * delta_rotation_j).transpose();
    const double dt_i2 = dt_i * dt_i;
    const double dt_j2 = dt_j * dt_j;
    const Eigen::Vector3d world_acc_i = R_i * acc_i - tassel_utils::G;
    const Eigen::Vector3d world_acc_j = R_j * (a_j - ba_j) - tassel_utils::G;
    const Eigen::Vector3d world_rot_acc_i = R_i * body_rot_acc_i;
    const Eigen::Vector3d world_rot_acc_j = R_j * body_rot_acc_j;
    const Eigen::Vector3d compensated_position_i =
        P_i + v_i * dt_i + 0.5 * world_acc_i * dt_i2 + (1.0 / 6.0) * world_rot_acc_i * dt_i2 * dt_i;
    const Eigen::Vector3d compensated_position_j =
        P_j + v_j * dt_j + 0.5 * world_acc_j * dt_j2 + (1.0 / 6.0) * world_rot_acc_j * dt_j2 * dt_j;
    const Eigen::Vector3d velocity_i = v_i + world_acc_i * dt_i + 0.5 * world_rot_acc_i * dt_i2;
    const Eigen::Vector3d velocity_j = v_j + world_acc_j * dt_j + 0.5 * world_rot_acc_j * dt_j2;

    // rho 为逆深度，dt_k = time_delay - sync_delay_k。两帧使用同一时间补偿模型：
    //   A_k       = Exp((gyro_k - Bg_k) * dt_k)
    //   P_bar_k   = P_k + V_k*dt_k + 1/2*(R_k*acc_k-G)*dt_k^2
    //               + 1/6*R_k*[omega_k]x*acc_k*dt_k^3
    //   p_C_i     = uv_i / rho
    //   p_G       = R_i*A_i*(ric*p_C_i+tic) + P_bar_i
    //   p_C_j     = ric^T*(A_j^T*R_j^T*(p_G-P_bar_j)-tic)
    // 代码中的目标帧 A_j 直接构造为 Exp(-omega_j*dt_j)，等价于上式的 A_j^T。
    Eigen::Vector3d pi_in_C = uv_i / inv_depth;
    Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
    Eigen::Vector3d pi_in_G = compensated_rotation_i * pi_in_I + compensated_position_i;
    Eigen::Vector3d pj_in_I = inverse_compensated_rotation_j * (pi_in_G - compensated_position_j);
    Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);

    double inv_z = 1.0 / pj_in_C.z();
    Eigen::Vector2d uv_pred_norm(pj_in_C.x() * inv_z, pj_in_C.y() * inv_z);
    Eigen::Vector2d predicted_pixel;
    Eigen::Matrix2d H_dz_dzn;
    if (jacobians) {
        camera->distortWithJacobian(uv_pred_norm, predicted_pixel, H_dz_dzn);
    } else {
        predicted_pixel = camera->distort(uv_pred_norm);
    }
    Eigen::Map<Eigen::Vector2d> r(residuals);
    r = sqrt_info * (predicted_pixel - pt_j);

    if (jacobians) {
        // 此处输出对优化参数中旋转向量的雅可比；写入边缘化系统时再转换到右扰动切空间。
        Eigen::Matrix<double, 2, 3> duv_dP;
        duv_dP << inv_z, 0, -pj_in_C.x() * inv_z * inv_z, 0, inv_z, -pj_in_C.y() * inv_z * inv_z;
        const Eigen::Matrix<double, 2, 3> reduce = sqrt_info * H_dz_dzn * duv_dP;
        const Eigen::Matrix3d camera_target_transform =
            ric.transpose() * delta_rotation_j.transpose() * R_j.transpose();
        const Eigen::Matrix<double, 2, 3> reduced_target_transform =
            reduce * camera_target_transform;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.block<2, 3>(0, 0) = reduced_target_transform;
            jacobian_pose_i.block<2, 3>(0, 3) =
                -reduced_target_transform * R_i *
                (Sophus::SO3d::hat(delta_rotation_i * pi_in_I) +
                 0.5 * Sophus::SO3d::hat(acc_i * dt_i * dt_i) +
                 (1.0 / 6.0) * Sophus::SO3d::hat(body_rot_acc_i * dt_i * dt_i * dt_i));
            jacobian_pose_i.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobian(-phi_i);
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            jacobian_pose_j.block<2, 3>(0, 0) = -reduced_target_transform;
            jacobian_pose_j.block<2, 3>(0, 3) =
                reduce * ric.transpose() * delta_rotation_j.transpose() *
                Sophus::SO3d::hat(
                    R_j.transpose() *
                    (pi_in_G - P_j - v_j * dt_j + 0.5 * tassel_utils::G * dt_j * dt_j));
            jacobian_pose_j.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobian(-phi_j);
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_dt(jacobians[2]);
            const Eigen::Vector3d& dP_i_dt = velocity_i;
            const Eigen::Vector3d& dP_j_dt = velocity_j;
            jacobian_dt = reduce * ric.transpose() *
                          (Sophus::SO3d::hat(bg_j - w_j) * pj_in_I +
                           delta_rotation_j.transpose() * R_j.transpose() *
                               (R_i * delta_rotation_i * Sophus::SO3d::hat(w_i - bg_i) * pi_in_I +
                                dP_i_dt - dP_j_dt));
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_inv_depth(jacobians[3]);
            jacobian_inv_depth = -reduce * ric.transpose() * delta_rotation_j.transpose() *
                                 R_j.transpose() * R_i * delta_rotation_i * ric *
                                 (pi_in_C / inv_depth);
        }
    }
    return true;
}

bool ReprojectionFactor::evaluateCached(
    double inv_depth, double* residuals, double** jacobians) const {
    const bool require_jacobian = jacobians != nullptr;
    if (!state->visual_values_valid || (require_jacobian && !state->visual_jacobians_valid)) {
        throw std::logic_error("Visual state is unavailable at the current evaluation point");
    }
    if (host_frame_index >= state->max_frame_count ||
        target_frame_index >= state->max_frame_count) {
        throw std::out_of_range("Cached reprojection frame index is outside the state");
    }
    const auto& host = state->frames[host_frame_index];
    const auto& target = state->frames[target_frame_index];
    const Eigen::Matrix3d relative_rotation = target.visual_inverse_rotation * host.visual_rotation;
    const Eigen::Vector3d relative_translation =
        target.visual_inverse_rotation *
        (host.visual_compensated_position - target.visual_compensated_position);
    const Eigen::Matrix3d camera_relative_rotation = ric.transpose() * relative_rotation * ric;
    const Eigen::Vector3d camera_relative_translation =
        ric.transpose() * (relative_rotation * tic + relative_translation - tic);

    const Eigen::Vector3d pi_in_C = uv_i / inv_depth;
    const Eigen::Vector3d pj_in_C =
        camera_relative_rotation * pi_in_C + camera_relative_translation;

    const double inv_z = 1.0 / pj_in_C.z();
    const Eigen::Vector2d uv_pred_norm(pj_in_C.x() * inv_z, pj_in_C.y() * inv_z);
    Eigen::Vector2d predicted_pixel;
    Eigen::Matrix2d distortion_jacobian;
    if (jacobians) {
        camera->distortWithJacobian(uv_pred_norm, predicted_pixel, distortion_jacobian);
    } else {
        predicted_pixel = camera->distort(uv_pred_norm);
    }
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = sqrt_info * (predicted_pixel - pt_j);

    if (!jacobians) {
        return true;
    }

    Eigen::Matrix<double, 2, 3> normalized_projection_jacobian;
    normalized_projection_jacobian << inv_z, 0, -pj_in_C.x() * inv_z * inv_z, 0, inv_z,
        -pj_in_C.y() * inv_z * inv_z;
    const Eigen::Matrix<double, 2, 3> reduce =
        sqrt_info * distortion_jacobian * normalized_projection_jacobian;
    const Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;

    if (jacobians[0] || jacobians[1]) {
        Eigen::Matrix<double, 2, 6> relative_pose_jacobian;
        relative_pose_jacobian.leftCols<3>() = reduce;
        relative_pose_jacobian.rightCols<3>() = -reduce * Sophus::SO3d::hat(pj_in_C);

        // 帧对缓存映射到 ambient 参数，Ceres 随后通过 Manifold 转回右扰动切空间。
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            Eigen::Matrix<double, 6, 6> host_pose_jacobian;
            const Eigen::Vector3d host_rotation_offset =
                0.5 * host.visual_acceleration * host.visual_dt * host.visual_dt +
                (1.0 / 6.0) * host.visual_body_rotational_acceleration * host.visual_dt *
                    host.visual_dt * host.visual_dt +
                host.visual_delta_rotation * tic;
            const Eigen::Matrix3d host_rotation =
                target.visual_camera_inverse_rotation * host.visual_base_rotation;
            host_pose_jacobian.setZero();
            host_pose_jacobian.topLeftCorner<3, 3>() = target.visual_camera_inverse_rotation;
            host_pose_jacobian.topRightCorner<3, 3>() =
                -host_rotation * Sophus::SO3d::hat(host_rotation_offset) +
                Sophus::SO3d::hat(camera_relative_translation) * host_rotation;
            host_pose_jacobian.bottomRightCorner<3, 3>() = host_rotation;
            host_pose_jacobian.rightCols<3>() *= host.visual_rotation_parameter_jacobian;
            jacobian_pose_i = relative_pose_jacobian * host_pose_jacobian;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            Eigen::Matrix<double, 6, 6> target_pose_jacobian;
            const Eigen::Vector3d target_rotation_offset =
                0.5 * target.visual_acceleration * target.visual_dt * target.visual_dt +
                (1.0 / 6.0) * target.visual_body_rotational_acceleration * target.visual_dt *
                    target.visual_dt * target.visual_dt +
                target.visual_delta_rotation * tic;
            const Eigen::Matrix3d target_rotation =
                target.visual_camera_inverse_rotation * target.visual_base_rotation;
            target_pose_jacobian.setZero();
            target_pose_jacobian.topLeftCorner<3, 3>() = -target.visual_camera_inverse_rotation;
            target_pose_jacobian.topRightCorner<3, 3>() =
                target_rotation * Sophus::SO3d::hat(target_rotation_offset);
            target_pose_jacobian.bottomRightCorner<3, 3>() = -target_rotation;
            target_pose_jacobian.rightCols<3>() *= target.visual_rotation_parameter_jacobian;
            jacobian_pose_j = relative_pose_jacobian * target_pose_jacobian;
        }
    }

    if (jacobians[2]) {
        const Eigen::Vector3d pj_in_I = relative_rotation * pi_in_I + relative_translation;
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_delay(jacobians[2]);
        jacobian_delay =
            reduce * ric.transpose() *
            (-Sophus::SO3d::hat(target.visual_omega) * pj_in_I +
             target.visual_inverse_rotation *
                 (host.visual_rotation * Sophus::SO3d::hat(host.visual_omega) * pi_in_I +
                  host.visual_position_delay_jacobian - target.visual_position_delay_jacobian));
    }

    if (jacobians[3]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_inv_depth(jacobians[3]);
        jacobian_inv_depth = -reduce * camera_relative_rotation * (pi_in_C / inv_depth);
    }
    return true;
}

}  // namespace tassel_core
