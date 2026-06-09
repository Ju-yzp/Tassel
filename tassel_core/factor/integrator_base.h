#ifndef TASSEL_CORE_INTEGRATOR_BASE_H_
#define TASSEL_CORE_INTEGRATOR_BASE_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <utility>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_core {

// 用户可以自行实现积分方法:欧拉法、中值法、龙格-库塔法等，同时设计可以避免频繁调用积分方法时的虚函数调用开销问题
template <typename Derived>
class IntegratorBase {
public:
    IntegratorBase(
        Eigen::Vector3d ba_lin, Eigen::Vector3d bg_lin, Eigen::Matrix<double, 18, 18> init_noise) {
        reset(ba_lin, bg_lin, init_noise);
    }

    void propagate(const tassel_utils::IMUMeasurement& measurement) {
        buffer.push_back(measurement);
        if (buffer.size() > 1) {
            const auto& prev = buffer[buffer.size() - 2];
            const auto& cur = buffer[buffer.size() - 1];
            static_cast<Derived*>(this)->integrate(prev, cur);
        }
    }

    void reset(
        const Eigen::Vector3d ba_lin, const Eigen::Vector3d bg_lin,
        const Eigen::Matrix<double, 18, 18> init_noise) {
        buffer.clear();
        sum_dt = 0.0;
        ba_linearized = ba_lin;
        bg_linearized = bg_lin;
        final_delta_p = Eigen::Vector3d::Zero();
        final_delta_v = Eigen::Vector3d::Zero();
        final_delta_q = Eigen::Matrix3d::Identity();
        noise = init_noise;
        jacobian = Eigen::Matrix<double, 15, 18>::Identity();
        covariance = Eigen::Matrix<double, 15, 15>::Zero();
    }

    void repropagate(
        const Eigen::Vector3d ba_lin, const Eigen::Vector3d bg_lin,
        const Eigen::Matrix<double, 18, 18> init_noise) {
        std::vector<tassel_utils::IMUMeasurement> tmp_buffer;
        std::swap(tmp_buffer, buffer);
        reset(ba_lin, bg_lin, init_noise);

        for (auto& measurement : tmp_buffer) {
            propagate(measurement);
        }
    }

    inline Eigen::Matrix3d get_dq_dbg() const { return jacobian.template block<3, 3>(3, 12); }

    inline Eigen::Matrix3d get_dp_dbg() const { return jacobian.template block<3, 3>(0, 12); }

    inline Eigen::Matrix3d get_dp_dba() const { return jacobian.template block<3, 3>(0, 9); }

    inline Eigen::Matrix3d get_dv_dbg() const { return jacobian.template block<3, 3>(6, 12); }

    inline Eigen::Matrix3d get_dv_dba() const { return jacobian.template block<3, 3>(6, 9); }

    std::vector<tassel_utils::IMUMeasurement> buffer;

    double sum_dt;

    // 偏置线性化点
    Eigen::Vector3d ba_linearized;
    Eigen::Vector3d bg_linearized;

    Eigen::Matrix3d final_delta_q;
    Eigen::Vector3d final_delta_p;
    Eigen::Vector3d final_delta_v;

    Eigen::Matrix<double, 15, 15> covariance;
    Eigen::Matrix<double, 15, 18> jacobian;
    Eigen::Matrix<double, 18, 18> noise;
};

class MidPointIntegrator : public IntegratorBase<MidPointIntegrator> {
public:
    using IntegratorBase::IntegratorBase;

    void integrate(
        const tassel_utils::IMUMeasurement& prev, const tassel_utils::IMUMeasurement& cur) {
        double dt = cur.timestamp - prev.timestamp;
        Eigen::Vector3d gyro = 0.5 * (prev.gyro + cur.gyro) - bg_linearized;
        Eigen::Matrix3d temp_delta_q = final_delta_q * Sophus::SO3d::exp(gyro * dt).matrix();
        Eigen::Vector3d prev_acc = final_delta_q * (prev.acc - ba_linearized);
        Eigen::Vector3d cur_acc = temp_delta_q * (cur.acc - ba_linearized);
        Eigen::Vector3d acc = 0.5 * (prev_acc + cur_acc);
        Eigen::Vector3d temp_delta_p = final_delta_p + final_delta_v * dt + 0.5 * acc * dt * dt;
        Eigen::Vector3d temp_delta_v = final_delta_v + acc * dt;

        Eigen::Vector3d a_0_x = prev.acc - ba_linearized;
        Eigen::Vector3d a_1_x = cur.acc - ba_linearized;
        Eigen::Matrix3d R_w_x = Sophus::SO3d::hat(gyro), R_a_0_x = Sophus::SO3d::hat(a_0_x),
                        R_a_1_x = Sophus::SO3d::hat(a_1_x);

        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
        F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(0, 3) =
            -0.25 * final_delta_q * R_a_0_x * dt * dt +
            -0.25 * temp_delta_q * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt * dt;
        F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * dt;
        F.block<3, 3>(0, 9) = -0.25 * (final_delta_q + temp_delta_q) * dt * dt;
        F.block<3, 3>(0, 12) = -0.25 * temp_delta_q * R_a_1_x * dt * dt * -dt;
        F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
        F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3, 3) * dt;
        F.block<3, 3>(6, 3) =
            -0.5 * final_delta_q * R_a_0_x * dt +
            -0.5 * temp_delta_q * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
        F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (final_delta_q + temp_delta_q) * dt;
        F.block<3, 3>(6, 12) = -0.5 * temp_delta_q * R_a_1_x * dt * -dt;
        F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * final_delta_q * dt * dt;
        V.block<3, 3>(0, 3) = 0.25 * -temp_delta_q * R_a_1_x * dt * dt * 0.5 * dt;
        V.block<3, 3>(0, 6) = 0.25 * temp_delta_q * dt * dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt;
        V.block<3, 3>(3, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt;
        V.block<3, 3>(6, 0) = 0.5 * final_delta_q * dt;
        V.block<3, 3>(6, 3) = 0.5 * -temp_delta_q * R_a_1_x * dt * 0.5 * dt;
        V.block<3, 3>(6, 6) = 0.5 * temp_delta_q * dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3, 3) * dt;
        V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3, 3) * dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();

        Eigen::Quaterniond q(temp_delta_q);
        q.normalize();
        final_delta_q = q.toRotationMatrix();
        final_delta_p = temp_delta_p;
        final_delta_v = temp_delta_v;
        sum_dt += dt;
    }
};

}  // namespace tassel_core

#endif /* TASSEL_CORE_INTEGRATOR_BASE_H_ */
