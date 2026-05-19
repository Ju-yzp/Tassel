#include "preintegrator.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

namespace tassel_core {
PreIntegrator::PreIntegrator(Eigen::Vector3d ba, Eigen::Vector3d bg)
    : convariance_(Eigen::Matrix<double, 15, 15>::Zero()),
      jacobian_(Eigen::Matrix<double, 15, 18>::Zero()),
      linearized_ba_(ba),
      linearized_bg_(bg),
      final_delta_q_(Eigen::Matrix3d::Identity()),
      final_delta_v_(Eigen::Vector3d::Zero()),
      final_delta_p_(Eigen::Vector3d::Zero()) {}

void PreIntegrator::add_measurement(const tassel_utils::IMUMeasurement& measurement) {
    buffer_.push_back(measurement);
    integrate();
}

void PreIntegrator::integrate() {
    if (buffer_.size() < 2) {
        return;
    }
    tassel_utils::IMUMeasurement& end = buffer_.back();
    tassel_utils::IMUMeasurement& start = buffer_[buffer_.size() - 2];
    double dt = end.get_timestamp() - start.get_timestamp();

    Eigen::Vector3d gyro = 0.5 * (end.gyro + start.gyro) - linearized_bg_;
    Eigen::Matrix3d temp_q = final_delta_q_ * Sophus::SO3d::exp(dt * gyro).matrix();
    Eigen::Vector3d acc_0 = final_delta_q_ * (start.acc - linearized_ba_);
    Eigen::Vector3d acc_1 = temp_q * (end.acc - linearized_ba_);
    Eigen::Vector3d acc = 0.5 * (acc_0 + acc_1);
    Eigen::Vector3d temp_v = final_delta_v_ + acc * dt;
    Eigen::Vector3d temp_p = final_delta_p_ + final_delta_v_ * dt + 0.5 * acc * dt * dt;

    // 更新协方差，状态转移矩阵
    Eigen::Vector3d w_x = gyro;
    Eigen::Vector3d a_0_x = start.acc - linearized_ba_;
    Eigen::Vector3d a_1_x = end.acc - linearized_ba_;
    Eigen::Matrix3d R_w_x = Sophus::SO3d::hat(w_x), R_a_0_x = Sophus::SO3d::hat(a_0_x),
                    R_a_1_x = Sophus::SO3d::hat(a_1_x);

    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(0, 3) =
        -0.25 * final_delta_q_ * R_a_0_x * dt * dt +
        -0.25 * temp_q * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt * dt;
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(0, 9) = -0.25 * (final_delta_q_ + temp_q) * dt * dt;
    F.block<3, 3>(0, 12) = -0.25 * temp_q * R_a_1_x * dt * dt * -dt;
    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
    F.block<3, 3>(3, 12) = -1.0 * Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(6, 3) = -0.5 * final_delta_q_ * R_a_0_x * dt +
                          -0.5 * temp_q * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 9) = -0.5 * (final_delta_q_ + temp_q) * dt;
    F.block<3, 3>(6, 12) = -0.5 * temp_q * R_a_1_x * dt * -dt;
    F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 15, 18> V = Eigen::Matrix<double, 15, 18>::Zero();
    V.block<3, 3>(0, 0) = 0.25 * final_delta_q_ * dt * dt;
    V.block<3, 3>(0, 3) = 0.25 * -temp_q * R_a_1_x * dt * dt * 0.5 * dt;
    V.block<3, 3>(0, 6) = 0.25 * temp_q * dt * dt;
    V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3) = 0.5 * Eigen::Matrix3d::Identity() * dt;
    V.block<3, 3>(3, 9) = 0.5 * Eigen::Matrix3d::Identity() * dt;
    V.block<3, 3>(6, 0) = 0.5 * final_delta_q_ * dt;
    V.block<3, 3>(6, 3) = 0.5 * -temp_q * R_a_1_x * dt * 0.5 * dt;
    V.block<3, 3>(6, 6) = 0.5 * temp_q * dt;
    V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12) = Eigen::Matrix3d::Identity() * dt;
    V.block<3, 3>(12, 15) = Eigen::Matrix3d::Identity() * dt;

    jacobian_ = F * jacobian_;
    convariance_ = F * convariance_ * F.transpose() + V * noise_ * V.transpose();

    final_delta_q_ = temp_q;
    final_delta_v_ = temp_v;
    final_delta_p_ = temp_p;
    sum_dt_ += dt;

    // 需要把final_delta_q_做正交化
    Eigen::Quaterniond q(final_delta_q_);
    q.normalize();
    final_delta_q_ = q.toRotationMatrix();
}

Eigen::Matrix<double, 15, 1> PreIntegrator::evaluate(
    const Eigen::Vector3d& Pi, const Eigen::Matrix3d& Qi, const Eigen::Vector3d& Vi,
    const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi, const Eigen::Vector3d& Pj,
    const Eigen::Matrix3d& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj,
    const Eigen::Vector3d& Bgj) {
    Eigen::Matrix<double, 15, 1> residuals;

    Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(0, 9);
    Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 12);
    Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(3, 12);
    Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(6, 9);
    Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(6, 12);

    Eigen::Vector3d dba = Bai - linearized_ba_;
    Eigen::Vector3d dbg = Bgi - linearized_bg_;

    Eigen::Matrix3d corrected_delta_q = final_delta_q_ * Sophus::SO3d(dq_dbg * dbg).matrix();
    Eigen::Vector3d corrected_delta_v = final_delta_v_ + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = final_delta_p_ + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(tassel_utils::O_P, 0) =
        Qi.inverse() * (0.5 * tassel_utils::G * sum_dt_ * sum_dt_ + Pj - Pi - Vi * sum_dt_) -
        corrected_delta_p;
    residuals.block<3, 1>(tassel_utils::O_R, 0) =
        2.0 * Sophus::SO3d(corrected_delta_q.inverse() * (Qi.inverse() * Qj)).log();
    residuals.block<3, 1>(tassel_utils::O_V, 0) =
        Qi.inverse() * (tassel_utils::G * sum_dt_ + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(tassel_utils::O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(tassel_utils::O_BG, 0) = Bgj - Bgi;
    return residuals;
}
}  // namespace tassel_core
