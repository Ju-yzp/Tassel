#ifndef TASSEL_CORE_FACTOR_IMU_FACTOR_H_
#define TASSEL_CORE_FACTOR_IMU_FACTOR_H_

#include <ceres/sized_cost_function.h>
#include <memory>
#include <sophus/so3.hpp>

#include "factor/integrator_base.h"
#include "tassel_utils/types.h"

namespace tassel_core {

template <typename Derived>
class IMUFactor : public ceres::SizedCostFunction<15, 6, 9, 6, 9> {
public:
    IMUFactor(std::shared_ptr<IntegratorBase<Derived>> integrator_) : integrator(integrator_) {}

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        const Eigen::Vector3d& G = tassel_utils::G;
        Eigen::Vector3d P_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d phi_i(parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix3d R_i = Sophus::SO3d::exp(phi_i).matrix();

        Eigen::Vector3d V_i(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Ba_i(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bg_i(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d P_j(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d phi_j(parameters[2][3], parameters[2][4], parameters[2][5]);
        Eigen::Matrix3d R_j = Sophus::SO3d::exp(phi_j).matrix();

        Eigen::Vector3d V_j(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Ba_j(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bg_j(parameters[3][6], parameters[3][7], parameters[3][8]);

        Eigen::Map<Eigen::Matrix<double, 15, 1, Eigen::ColMajor>> res(residuals);
        Eigen::Matrix3d dp_dba = integrator->get_dp_dba();
        Eigen::Matrix3d dp_dbg = integrator->get_dp_dbg();

        Eigen::Matrix3d dq_dbg = integrator->get_dq_dbg();

        Eigen::Matrix3d dv_dba = integrator->get_dv_dba();
        Eigen::Matrix3d dv_dbg = integrator->get_dv_dbg();

        Eigen::Vector3d dba = Ba_i - integrator->ba_linearized;
        Eigen::Vector3d dbg = Bg_i - integrator->bg_linearized;

        Eigen::Matrix3d corrected_delta_q =
            integrator->final_delta_q * Sophus::SO3d::exp(dq_dbg * dbg).matrix();
        Eigen::Vector3d corrected_delta_v = integrator->final_delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = integrator->final_delta_p + dp_dba * dba + dp_dbg * dbg;

        double sum_dt = integrator->sum_dt;

        // 残差: [dp, dq, dv, dba, dbg]
        res.block<3, 1>(0, 0) =
            R_i.transpose() * (0.5 * G * sum_dt * sum_dt + P_j - P_i - V_i * sum_dt) -
            corrected_delta_p;
        res.block<3, 1>(3, 0) =
            Sophus::SO3d(corrected_delta_q.inverse() * (R_i.inverse() * R_j)).log();
        res.template block<3, 1>(6, 0) =
            R_i.transpose() * (G * sum_dt + V_j - V_i) - corrected_delta_v;
        res.template block<3, 1>(9, 0) = Ba_j - Ba_i;
        res.template block<3, 1>(12, 0) = Bg_j - Bg_i;

        // 信息矩阵
        Eigen::Matrix<double, 15, 15> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(integrator->covariance.inverse())
                .matrixL()
                .transpose();

        Eigen::Matrix3d Jr_inv =
            Sophus::SO3d::leftJacobianInverse(res.block<3, 1>(3, 0)).transpose();
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_i(
                    jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.template block<3, 3>(0, 3) = Sophus::SO3d::hat(
                    R_i.transpose() * (0.5 * G * sum_dt * sum_dt + P_j - P_i - V_i * sum_dt));
                jacobian_pose_i.template block<3, 3>(0, 0) = -R_i.transpose();
                jacobian_pose_i.template block<3, 3>(3, 3) = Jr_inv * R_j.transpose() * R_i;
                jacobian_pose_i.template block<3, 3>(6, 3) =
                    Sophus::SO3d::hat(R_i.transpose() * (G * sum_dt + V_j - V_i));
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(
                    jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(0, 0) = -R_i.transpose() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(0, 3) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(0, 6) = -dp_dbg;

                jacobian_speedbias_i.block<3, 3>(3, 6) = -Jr_inv * dq_dbg;

                jacobian_speedbias_i.block<3, 3>(6, 0) = -R_i.transpose();
                jacobian_speedbias_i.block<3, 3>(6, 3) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(6, 6) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
                jacobian_speedbias_i.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_j(
                    jacobians[2]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0, 0) = R_i.transpose();
                jacobian_pose_j.block<3, 3>(3, 3) = Jr_inv;
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }
            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(
                    jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(6, 0) = R_i.transpose();
                jacobian_speedbias_j.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
            }
        }

        res = sqrt_info * res;

        return true;
    }
    std::shared_ptr<IntegratorBase<Derived>> integrator;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FACTOR_IMU_FACTOR_H_
