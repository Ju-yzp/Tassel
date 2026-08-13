#ifndef TASSEL_CORE_TESTS_IMU_TEST_UTILS_H_
#define TASSEL_CORE_TESTS_IMU_TEST_UTILS_H_

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

#include <sophus/so3.hpp>

#include "tassel_utils/types.h"

namespace tassel_core::test {

struct ImuSample {
    double ts = 0.0;
    Eigen::Vector3d pos_w_i = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel_w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot_w_i = Eigen::Matrix3d::Identity();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
};

struct ImuTimeline {
    double imu_dt = 0.0;
    std::vector<ImuSample> states;

    const ImuSample& at_index(int k) const { return states[k]; }
};

struct CameraState {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
};

inline ImuTimeline generateConstantMotionTimeline(
    double duration, double imu_dt, const Eigen::Vector3d& a_body, const Eigen::Vector3d& w_body,
    const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
    const Eigen::Vector3d& gravity = tassel_utils::G) {
    const int total_steps = static_cast<int>(duration / imu_dt) + 1;
    ImuTimeline timeline;
    timeline.imu_dt = imu_dt;
    timeline.states.reserve(total_steps);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d V = Eigen::Vector3d::Zero();

    for (int k = 0; k < total_steps; ++k) {
        ImuSample sample;
        sample.ts = k * imu_dt;
        sample.rot_w_i = R;
        sample.pos_w_i = P;
        sample.vel_w = V;
        sample.accel_bias = ba;
        sample.gyro_bias = bg;
        sample.gyro = w_body + bg;
        sample.acc = a_body + R.transpose() * gravity + ba;
        timeline.states.push_back(sample);

        const double half_dt = 0.5 * imu_dt;
        const Eigen::Matrix3d R_mid = R * Sophus::SO3d::exp(w_body * half_dt).matrix();
        const Eigen::Vector3d V_next = V + R_mid * a_body * imu_dt;
        const Eigen::Vector3d P_next = P + 0.5 * (V + V_next) * imu_dt;
        R = R * Sophus::SO3d::exp(w_body * imu_dt).matrix();
        P = P_next;
        V = V_next;
    }

    return timeline;
}

inline ImuSample interpolateSample(const ImuTimeline& timeline, double t) {
    const auto& states = timeline.states;
    const double imu_dt = timeline.imu_dt;
    int i0 = static_cast<int>(std::floor(t / imu_dt));
    int i1 = i0 + 1;
    if (i0 < 0) {
        i0 = 0;
        i1 = 0;
    }
    if (i1 >= static_cast<int>(states.size())) {
        i0 = static_cast<int>(states.size()) - 1;
        i1 = i0;
    }

    const auto& s0 = states[i0];
    const auto& s1 = states[i1];
    const double alpha = (i1 > i0) ? (t - s0.ts) / (s1.ts - s0.ts) : 0.0;
    const double a = std::clamp(alpha, 0.0, 1.0);

    ImuSample sample;
    sample.ts = t;
    sample.pos_w_i = (1.0 - a) * s0.pos_w_i + a * s1.pos_w_i;
    sample.vel_w = (1.0 - a) * s0.vel_w + a * s1.vel_w;
    sample.gyro = (1.0 - a) * s0.gyro + a * s1.gyro;
    sample.acc = (1.0 - a) * s0.acc + a * s1.acc;
    sample.accel_bias = (1.0 - a) * s0.accel_bias + a * s1.accel_bias;
    sample.gyro_bias = (1.0 - a) * s0.gyro_bias + a * s1.gyro_bias;
    const Eigen::Quaterniond q0(s0.rot_w_i);
    const Eigen::Quaterniond q1(s1.rot_w_i);
    sample.rot_w_i = q0.slerp(a, q1).toRotationMatrix();
    return sample;
}

inline CameraState integrateImu(
    const ImuTimeline& timeline, double t_param, double t_cam, const Eigen::Vector3d& bg,
    const Eigen::Vector3d& ba) {
    ImuSample start = interpolateSample(timeline, t_param);
    Eigen::Vector3d P = start.pos_w_i;
    Eigen::Vector3d V = start.vel_w;
    Eigen::Matrix3d R = start.rot_w_i;
    double t = t_param;

    while (t < t_cam - 1e-9) {
        const double next_imu = std::ceil((t + 1e-9) / timeline.imu_dt) * timeline.imu_dt;
        const double t_next = std::min(next_imu, t_cam);
        const double dt = t_next - t;
        if (dt < 1e-9) {
            break;
        }

        const ImuSample prev = interpolateSample(timeline, t);
        const ImuSample next = interpolateSample(timeline, t_next);

        const Eigen::Vector3d gyro = 0.5 * (prev.gyro + next.gyro) - bg;
        const Eigen::Matrix3d R_next = R * Sophus::SO3d::exp(gyro * dt).matrix();
        const Eigen::Vector3d prev_acc = R * (prev.acc - ba);
        const Eigen::Vector3d next_acc = R_next * (next.acc - ba);
        const Eigen::Vector3d acc_mid = 0.5 * (prev_acc + next_acc);

        P += V * dt + 0.5 * acc_mid * dt * dt;
        V += acc_mid * dt;
        R = R_next;
        t = t_next;
    }

    return {R, P};
}

}  // namespace tassel_core::test

#endif  // TASSEL_CORE_TESTS_IMU_TEST_UTILS_H_
