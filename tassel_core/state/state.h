#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <sophus/se3.hpp>
#include <vector>

namespace tassel_core {
struct State {
    State(
        int max_frame_count_ = 10, bool use_imu_ = false,
        Eigen::Matrix3d ric_ = Eigen::Matrix3d::Identity(),
        Eigen::Vector3d tic_ = Eigen::Vector3d::Zero())
        : max_frame_count(max_frame_count_),
          cur_frame_count(0),
          ric(ric_),
          tic(tic_),
          use_imu(use_imu_) {
        Rs.resize(max_frame_count, Eigen::Matrix3d::Identity());
        Ps.resize(max_frame_count, Eigen::Vector3d::Zero());
        Vs.resize(max_frame_count, Eigen::Vector3d::Zero());
        Bas.resize(max_frame_count, Eigen::Vector3d::Zero());
        Bgs.resize(max_frame_count, Eigen::Vector3d::Zero());
        param_poses.resize(max_frame_count, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        param_speed_bias.resize(max_frame_count, std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, 0});
        delay_time = 0.0;
        param_delay_time = 0.0;
        if (max_frame_count < 1) {
            throw std::runtime_error("max_frame_count must be greater than 0");
        }
    }

    void stateToParam(int idx) {
        static constexpr double kSinThetaEps = 1e-6;
        static constexpr double kPiMinusThetaEps = 1e-4;

        const Eigen::Matrix3d& R = Rs[idx];
        double cos_theta = (R.trace() - 1.0) / 2.0;
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double theta = std::acos(cos_theta);
        double sin_theta = std::sin(theta);

        Eigen::Vector3d phi;
        if (sin_theta < kSinThetaEps) {
            phi = Sophus::SO3d::vee(R - R.transpose()) * (0.5 - theta * theta / 12.0);
        } else if (M_PI - theta < kPiMinusThetaEps) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(R);
            int idx_max = es.eigenvalues().maxCoeff();
            phi = es.eigenvectors().col(idx_max).normalized() * theta;
            if ((Sophus::SO3d::vee(R - R.transpose())).dot(phi) < 0) phi = -phi;
        } else {
            phi = Sophus::SO3d::vee(R - R.transpose()) * (theta / (2.0 * sin_theta));
        }

        param_poses[idx][0] = phi.x();
        param_poses[idx][1] = phi.y();
        param_poses[idx][2] = phi.z();
        param_poses[idx][3] = Ps[idx].x();
        param_poses[idx][4] = Ps[idx].y();
        param_poses[idx][5] = Ps[idx].z();

        for (int d = 0; d < 3; ++d) {
            param_speed_bias[idx][d] = Vs[idx][d];
            param_speed_bias[idx][d + 3] = Bas[idx][d];
            param_speed_bias[idx][d + 6] = Bgs[idx][d];
        }
    }

    void paramToState(int idx) {
        Eigen::Vector3d phi(param_poses[idx][0], param_poses[idx][1], param_poses[idx][2]);
        Sophus::SO3d R_so3 = Sophus::SO3d::exp(phi);
        Rs[idx] = R_so3.matrix();
        Ps[idx] = Eigen::Vector3d(param_poses[idx][3], param_poses[idx][4], param_poses[idx][5]);

        for (int d = 0; d < 3; ++d) {
            Vs[idx][d] = param_speed_bias[idx][d];
            Bas[idx][d] = param_speed_bias[idx][d + 3];
            Bgs[idx][d] = param_speed_bias[idx][d + 6];
        }
    }

    void stateToParams() {
        for (int i = 0; i < max_frame_count; ++i) {
            stateToParam(i);
        }
        param_delay_time = delay_time;
    }

    void paramsToState() {
        for (int i = 0; i < max_frame_count; ++i) {
            paramToState(i);
        }
        delay_time = param_delay_time;
    }

    int max_frame_count;
    int cur_frame_count;
    bool use_imu;

    // 外参
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;

    // 位姿 / 速度 / 偏置 / 时间延迟
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> Ps;
    std::vector<Eigen::Vector3d> Vs;
    std::vector<Eigen::Vector3d> Bas;
    std::vector<Eigen::Vector3d> Bgs;
    double delay_time;

    std::vector<std::array<double, 6>> param_poses;
    std::vector<std::array<double, 9>> param_speed_bias;
    double param_delay_time;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
