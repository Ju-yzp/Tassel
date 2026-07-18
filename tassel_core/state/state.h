#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <sophus/se3.hpp>
#include <stdexcept>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_core {
class CameraBase;
struct State {
    State(int max_frame_count_ = 10) : max_frame_count(max_frame_count_), cur_frame_count(0) {
        if (max_frame_count < 1) {
            throw std::runtime_error("max_frame_count must be greater than 0");
        }
        Rs.resize(max_frame_count, Eigen::Matrix3d::Identity());
        Ps.resize(max_frame_count, Eigen::Vector3d::Zero());
        Vs.resize(max_frame_count, Eigen::Vector3d::Zero());
        Bas.resize(max_frame_count, Eigen::Vector3d::Zero());
        Bgs.resize(max_frame_count, Eigen::Vector3d::Zero());
        frame_delays.resize(max_frame_count, 0.0);
        frame_ids.resize(max_frame_count, tassel_utils::kInvalidFrameId);
        params_pose.resize(max_frame_count, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        params_speed_bias.resize(max_frame_count, std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, 0});
        delay_time = 0.0;
        param_delay_time = 0.0;
    }

    void stateToParam(int idx) {
        const Eigen::Matrix3d& R = Rs[idx];
        const Eigen::Vector3d phi = Sophus::SO3d(R).log();

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

    int findFrameSlot(tassel_utils::FrameId frame_id) const {
        if (frame_id == tassel_utils::kInvalidFrameId) return -1;
        for (int i = 0; i <= cur_frame_count; ++i) {
            if (frame_ids[i] == frame_id) return i;
        }
        return -1;
    }

    int firstActiveImuSlot() const { return has_retained_host ? 1 : 0; }

    void copyFrameSlot(int source, int destination) {
        Rs[destination] = Rs[source];
        Ps[destination] = Ps[source];
        Vs[destination] = Vs[source];
        Bas[destination] = Bas[source];
        Bgs[destination] = Bgs[source];
        frame_delays[destination] = frame_delays[source];
        frame_ids[destination] = frame_ids[source];
    }

    State get_compensated_state() const {
        State compensated = *this;
        for (int i = 0; i < cur_frame_count; ++i) {
            const double dt = delay_time - frame_delays[i];
            compensated.Rs[i] = Rs[i] * Sophus::SO3d::exp((gyro_vec[i] - Bgs[i]) * dt).matrix();
            compensated.Ps[i] = Ps[i] + Vs[i] * dt +
                                0.5 * (Rs[i] * (acc_vec[i] - Bas[i]) - tassel_utils::G) * dt * dt;
        }
        return compensated;
    }

    void reset() {
        cur_frame_count = 0;
        has_retained_host = false;
        acc_vec.clear();
        gyro_vec.clear();
        std::fill(Rs.begin(), Rs.end(), Eigen::Matrix3d::Identity());
        std::fill(Ps.begin(), Ps.end(), Eigen::Vector3d::Zero());
        std::fill(Vs.begin(), Vs.end(), Eigen::Vector3d::Zero());
        std::fill(Bas.begin(), Bas.end(), Eigen::Vector3d::Zero());
        std::fill(Bgs.begin(), Bgs.end(), Eigen::Vector3d::Zero());
        std::fill(frame_delays.begin(), frame_delays.end(), 0.0);
        std::fill(frame_ids.begin(), frame_ids.end(), tassel_utils::kInvalidFrameId);
        std::fill(params_pose.begin(), params_pose.end(), std::array<double, 6>{0, 0, 0, 0, 0, 0});
        std::fill(
            params_speed_bias.begin(), params_speed_bias.end(),
            std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, 0});
        delay_time = 0.0;
        param_delay_time = 0.0;
    }

    int max_frame_count;
    // Index of the newest populated slot; during a full window it is max_frame_count - 1.
    int cur_frame_count;
    bool has_retained_host = false;
    // 位姿 / 速度 / 偏置 / 时间延迟
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> Ps;
    std::vector<Eigen::Vector3d> Vs;
    std::vector<Eigen::Vector3d> Bas;
    std::vector<Eigen::Vector3d> Bgs;
    std::vector<double> frame_delays;
    std::vector<tassel_utils::FrameId> frame_ids;
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
