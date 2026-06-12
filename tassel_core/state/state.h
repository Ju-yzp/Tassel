#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <sophus/se3.hpp>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_core {
class CameraBase;
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
        params_pose.resize(max_frame_count, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        params_speed_bias.resize(max_frame_count, std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, 0});
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

        params_pose[idx][0] = Ps[idx].x();
        params_pose[idx][1] = Ps[idx].y();
        params_pose[idx][2] = Ps[idx].z();
        params_pose[idx][3] = phi.x();
        params_pose[idx][4] = phi.y();
        params_pose[idx][5] = phi.z();

        for (int d = 0; d < 3; ++d) {
            params_speed_bias[idx][d] = Vs[idx][d];
            params_speed_bias[idx][d + 3] = Bas[idx][d];
            params_speed_bias[idx][d + 6] = Bgs[idx][d];
        }
    }

    void paramToState(int idx) {
        Ps[idx] = Eigen::Vector3d(params_pose[idx][0], params_pose[idx][1], params_pose[idx][2]);
        Eigen::Vector3d phi(params_pose[idx][3], params_pose[idx][4], params_pose[idx][5]);
        Sophus::SO3d R_so3 = Sophus::SO3d::exp(phi);
        Rs[idx] = R_so3.matrix();

        for (int d = 0; d < 3; ++d) {
            Vs[idx][d] = params_speed_bias[idx][d];
            Bas[idx][d] = params_speed_bias[idx][d + 3];
            Bgs[idx][d] = params_speed_bias[idx][d + 6];
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

    State get_compensated_state() const {
        State compensated = *this;
        for (int i = 0; i < cur_frame_count; ++i) {
            compensated.Rs[i] =
                Rs[i] * Sophus::SO3d::exp((gyro_vec[i] - Bgs[i]) * delay_time).matrix();
            compensated.Ps[i] =
                Ps[i] + Vs[i] * delay_time +
                0.5 * (Rs[i] * (acc_vec[i] - Bas[i]) - tassel_utils::G) * delay_time * delay_time;
        }
        return compensated;
    }

    void reset() {
        cur_frame_count = 0;
        acc_vec.clear();
        gyro_vec.clear();
        std::fill(Rs.begin(), Rs.end(), Eigen::Matrix3d::Identity());
        std::fill(Ps.begin(), Ps.end(), Eigen::Vector3d::Zero());
        std::fill(Vs.begin(), Vs.end(), Eigen::Vector3d::Zero());
        std::fill(Bas.begin(), Bas.end(), Eigen::Vector3d::Zero());
        std::fill(Bgs.begin(), Bgs.end(), Eigen::Vector3d::Zero());
        std::fill(params_pose.begin(), params_pose.end(), std::array<double, 6>{0, 0, 0, 0, 0, 0});
        std::fill(
            params_speed_bias.begin(), params_speed_bias.end(),
            std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, 0});
        delay_time = 0.0;
        param_delay_time = 0.0;
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

    // 保存imu(t)时刻采样的imu体坐标系下的加速度和角速度
    std::vector<Eigen::Vector3d> acc_vec;
    std::vector<Eigen::Vector3d> gyro_vec;

    // 状态变量转换后的ceres优化变量
    std::vector<std::array<double, 6>> params_pose;
    std::vector<std::array<double, 9>> params_speed_bias;  // 线速度 / 加速度偏置 / 角速度偏置
    double param_delay_time;

    //  相机模型（用于 VisualFactor 畸变与雅可比）
    const CameraBase* camera = nullptr;

    //  视觉因子信息矩阵
    Eigen::Matrix2d visual_sqrt_info;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
