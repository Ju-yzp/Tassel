#ifndef TASSEL_CORE_FACTOR_IMU_FACTOR_H_
#define TASSEL_CORE_FACTOR_IMU_FACTOR_H_

#include <ceres/sized_cost_function.h>
#include <memory>
#include "factor/integrator_base.h"

namespace tassel_core {

template <typename Derived>
class IMUFactor : ceres::SizedCostFunction<15, 6, 9, 6, 9> {
    IMUFactor(std::shared_ptr<IntegratorBase<Derived>> integrator_) : integrator(integrator_) {}

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Matrix3d R_i, R_j;
        Eigen::Vector3d P_i, P_j, V_i, V_j, Ba_i, Ba_j, Bg_i, Bg_j;

        Eigen::Map<Eigen::Matrix<double, 15, 1, Eigen::ColMajor>> res(residuals);
        Eigen::Matrix3d dp_dba = integrator->jacobian.template block<3, 3>(0, 9);
        Eigen::Matrix3d dp_dbg = integrator->jacobian.template block<3, 3>(0, 12);

        Eigen::Matrix3d dq_dbg = integrator->jacobian.template block<3, 3>(3, 12);

        Eigen::Matrix3d dv_dba = integrator->jacobian.template block<3, 3>(6, 9);
        Eigen::Matrix3d dv_dbg = integrator->jacobian.template block<3, 3>(6, 12);

        // 优化点和线性化点之间变化量
        Eigen::Vector3d dba = Ba_i - integrator->ba_linearized;
        Eigen::Vector3d dbg = Bg_i - integrator->bg_linearized;

        Eigen::Matrix3d corrected_delta_q =
            integrator->final_delta_q * Sophus::SO3d::exp(dq_dbg * dbg).matrix();
        Eigen::Vector3d corrected_delta_v = integrator->final_delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = integrator->delta_p + dp_dba * dba + dp_dbg * dbg;

        double sum_dt = integrator->sum_dt;

        // 残差
        res.template block<3, 1>(0, 0) =
            R_i.transpose() * (0.5 * tassel_utils::G * sum_dt * sum_dt + P_j - P_i - V_i * sum_dt) -
            corrected_delta_p;
        // res.block<3, 1>(3, 0) =
        //     2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        res.template block<3, 1>(6, 0) =
            R_i.transpose() * (tassel_utils::G * sum_dt + V_j - V_i) - corrected_delta_v;
        res.template block<3, 1>(9, 0) = Ba_j - Ba_i;
        res.template block<3, 1>(12, 0) = Bg_j - Bg_i;

        // 信息矩阵
        Eigen::Matrix<double, 15, 15> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(integrator->covariance.inverse())
                .matrixL()
                .transpose();

        // 雅各比矩阵
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_i(
                    jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.template block<3, 3>(0, 0) = -R_i.transpose() * P_i;
                jacobian_pose_i.template block<3, 3>(0, 3) = -Sophus::SO3d::hat(
                    R_i.transpose() *
                    (0.5 * tassel_utils::G * sum_dt * sum_dt + P_j - P_i - V_i * sum_dt));
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(
                    jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.template block<3, 3>(0, 0) = -R_i.transpose() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(0, 3) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(0, 6) = -dp_dbg;

                jacobian_speedbias_i.block<3, 3>(6, 0) = -R_i.transpose();
                jacobian_speedbias_i.block<3, 3>(6, 3) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(6, 6) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
                jacobian_speedbias_i.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_j(
                    jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(0, 0) = R_i.transpose();
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }
            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(
                    jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(6, 0) = -R_i.transpose();
                jacobian_speedbias_j.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
            }
        }

        return true;
    }

    void get_dense_Jp_b(Eigen::MatrixXd& Jp, Eigen::VectorXd& b, int start_row) {}

    std::shared_ptr<IntegratorBase<Derived>> integrator;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FACTOR_IMU_FACTOR_H_
