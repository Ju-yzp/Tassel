// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#include "reprojection_factor.h"
#include <Eigen/Geometry>
#include <cmath>
#include <sophus/so3.hpp>
#include <stdexcept>

#include "factor/visual_frame_cache.h"
#include "state/frame_kinematics.h"
#include "tassel_utils/types.h"

namespace tassel_core {

ReprojectionFactor::ReprojectionFactor(
    const Eigen::Vector3d& uv_i_, const Eigen::Vector2d& pt_j_, const Eigen::Matrix3d& ric_,
    const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
    const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
    const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_, const double* ba_i_lin_,
    const double* ba_j_lin_, const Eigen::Matrix2d& sqrt_info_, const CameraBase* camera_,
    double sync_delay_i_, double sync_delay_j_, const VisualFrameCache* frame_cache_,
    int host_frame_index_, int target_frame_index_)
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
      frame_cache(frame_cache_),
      host_frame_index(host_frame_index_),
      target_frame_index(target_frame_index_) {
    if (frame_cache && (host_frame_index < 0 || target_frame_index < 0)) {
        throw std::invalid_argument("Cached reprojection factor requires valid frame indices");
    }
}

bool ReprojectionFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    if (frame_cache) {
        return evaluateCached(parameters[3][0], residuals, jacobians);
    }

    Eigen::Vector3d P_i(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d phi_i(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Matrix3d R_i = Sophus::SO3d::exp(phi_i).matrix();

    Eigen::Vector3d P_j(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d phi_j(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Matrix3d R_j = Sophus::SO3d::exp(phi_j).matrix();

    const double delay_time = parameters[2][0];
    const double dt_i = delay_time - sync_delay_i;
    const double dt_j = delay_time - sync_delay_j;
    double inv_depth = parameters[3][0];

    const FrameKinematics host =
        propagateFrameKinematics(R_i, P_i, v_i, w_i, a_i, bg_i, ba_i, dt_i);
    const FrameKinematics target =
        propagateFrameKinematics(R_j, P_j, v_j, w_j, a_j, bg_j, ba_j, dt_j);
    const Eigen::Matrix3d& A_i = host.delta_rotation;
    const Eigen::Matrix3d A_j = target.delta_rotation.transpose();
    const Eigen::Vector3d& acc_i = host.acceleration;
    const Eigen::Vector3d& body_rot_acc_i = host.body_rotational_acceleration;

    // rho 为逆深度，dt_k = delay_time - sync_delay_k。两帧使用同一时间补偿模型：
    //   A_k       = Exp((gyro_k - Bg_k) * dt_k)
    //   P_bar_k   = P_k + V_k*dt_k + 1/2*(R_k*acc_k-G)*dt_k^2
    //               + 1/6*R_k*[omega_k]x*acc_k*dt_k^3
    //   p_C_i     = uv_i / rho
    //   p_G       = R_i*A_i*(ric*p_C_i+tic) + P_bar_i
    //   p_C_j     = ric^T*(A_j^T*R_j^T*(p_G-P_bar_j)-tic)
    // 代码中的目标帧 A_j 直接构造为 Exp(-omega_j*dt_j)，等价于上式的 A_j^T。
    Eigen::Vector3d pi_in_C = uv_i / inv_depth;
    Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
    Eigen::Vector3d pi_in_G = host.rotation * pi_in_I + host.position;
    Eigen::Vector3d pj_in_I = target.inverse_rotation * (pi_in_G - target.position);
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
        const Eigen::Matrix3d camera_target_transform = ric.transpose() * A_j * R_j.transpose();
        const Eigen::Matrix<double, 2, 3> reduced_target_transform =
            reduce * camera_target_transform;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.block<2, 3>(0, 0) = reduced_target_transform;
            jacobian_pose_i.block<2, 3>(0, 3) =
                -reduced_target_transform * R_i *
                (Sophus::SO3d::hat(A_i * pi_in_I) + 0.5 * Sophus::SO3d::hat(acc_i * dt_i * dt_i) +
                 (1.0 / 6.0) * Sophus::SO3d::hat(body_rot_acc_i * dt_i * dt_i * dt_i));
            jacobian_pose_i.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobian(-phi_i);
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            jacobian_pose_j.block<2, 3>(0, 0) = -reduced_target_transform;
            jacobian_pose_j.block<2, 3>(0, 3) =
                reduce * ric.transpose() * A_j *
                Sophus::SO3d::hat(
                    R_j.transpose() *
                    (pi_in_G - P_j - v_j * dt_j + 0.5 * tassel_utils::G * dt_j * dt_j));
            jacobian_pose_j.block<2, 3>(0, 3) *= Sophus::SO3d::leftJacobian(-phi_j);
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_dt(jacobians[2]);
            const Eigen::Vector3d& dP_i_dt = host.velocity;
            const Eigen::Vector3d& dP_j_dt = target.velocity;
            jacobian_dt =
                reduce * ric.transpose() *
                (Sophus::SO3d::hat(bg_j - w_j) * pj_in_I +
                 A_j * R_j.transpose() *
                     (R_i * A_i * Sophus::SO3d::hat(w_i - bg_i) * pi_in_I + dP_i_dt - dP_j_dt));
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_inv_depth(jacobians[3]);
            jacobian_inv_depth = -reduce * ric.transpose() * A_j * R_j.transpose() * R_i * A_i *
                                 ric * (pi_in_C / inv_depth);
        }
    }
    return true;
}

bool ReprojectionFactor::evaluateCached(
    double inv_depth, double* residuals, double** jacobians) const {
    const bool require_jacobian = jacobians != nullptr;
    const auto& host = frame_cache->frame(host_frame_index, require_jacobian);
    const auto& target = frame_cache->frame(target_frame_index, require_jacobian);
    const auto& pair = frame_cache->pair(host_frame_index, target_frame_index, require_jacobian);

    const Eigen::Vector3d pi_in_C = uv_i / inv_depth;
    const Eigen::Vector3d pj_in_C =
        pair.camera_relative_rotation * pi_in_C + pair.camera_relative_translation;

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
            jacobian_pose_i = relative_pose_jacobian * pair.host_pose_jacobian;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            jacobian_pose_j = relative_pose_jacobian * pair.target_pose_jacobian;
        }
    }

    if (jacobians[2]) {
        const Eigen::Vector3d pj_in_I =
            pair.relative_rotation * pi_in_I + pair.relative_translation;
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_delay(jacobians[2]);
        jacobian_delay = reduce * ric.transpose() *
                         (-Sophus::SO3d::hat(target.omega) * pj_in_I +
                          target.inverse_compensated_rotation *
                              (host.world_rotation * Sophus::SO3d::hat(host.omega) * pi_in_I +
                               host.position_delay_jacobian - target.position_delay_jacobian));
    }

    if (jacobians[3]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_inv_depth(jacobians[3]);
        jacobian_inv_depth = -reduce * pair.camera_relative_rotation * (pi_in_C / inv_depth);
    }
    return true;
}

}  // namespace tassel_core
