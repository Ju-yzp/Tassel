#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#include "tassel_utils/macros.h"

namespace tassel_core {
using Pose = Sophus::SE3d;

inline void increasePose(const Eigen::Vector<double, 6>& delta, Pose& pose) {
    pose = pose * Sophus::SE3d::exp(delta);
}

struct PoseVelBiasState {
    PoseVelBiasState()
        : pose_linearized(Sophus::SE3d()),
          v_linearized(Eigen::Vector3d::Zero()),
          ba_linearized(Eigen::Vector3d::Zero()),
          bg_linearized(Eigen::Vector3d::Zero()),
          linearized(false),
          delta(Eigen::Vector<double, tassel_utils::TOTAL_SIZE>::Zero()),
          T_wc_current(Sophus::SE3d()),
          v_current(Eigen::Vector3d::Zero()),
          ba_current(Eigen::Vector3d::Zero()),
          bg_current(Eigen::Vector3d::Zero()) {}

    PoseVelBiasState(const Sophus::SE3d& T_wc, bool linearized = false)
        : pose_linearized(T_wc),
          v_linearized(Eigen::Vector3d::Zero()),
          ba_linearized(Eigen::Vector3d::Zero()),
          bg_linearized(Eigen::Vector3d::Zero()),
          linearized(linearized),
          delta(Eigen::Vector<double, tassel_utils::TOTAL_SIZE>::Zero()),
          T_wc_current(T_wc),
          v_current(Eigen::Vector3d::Zero()),
          ba_current(Eigen::Vector3d::Zero()),
          bg_current(Eigen::Vector3d::Zero()) {}

    inline void setLinearized() {
        linearized = true;
        if (delta.segment<tassel_utils::POSE_SIZE>(tassel_utils::POSE_IDX).norm() > 1e-12) {
            throw std::runtime_error("delta is not zero");
        }
        T_wc_current = pose_linearized;
    }

    inline Pose get_pose() const {
        if (!linearized) {
            return T_wc_current;
        } else {
            return pose_linearized;
        }
    }

    inline Pose get_optimized_pose() const { return T_wc_current; }

    inline void applyDelta(const Eigen::VectorXd& inc, bool use_imu = false) {
        if (!linearized) {
            increasePose(
                inc.segment<tassel_utils::POSE_SIZE>(tassel_utils::POSE_IDX), pose_linearized);
            if (use_imu) {
                v_linearized += inc.segment<tassel_utils::SPEED_SIZE>(tassel_utils::SPEED_IDX);
                ba_linearized +=
                    inc.segment<tassel_utils::BIAS_ACC_SIZE>(tassel_utils::BIAS_ACC_IDX);
                bg_linearized +=
                    inc.segment<tassel_utils::BIAS_GYRO_SIZE>(tassel_utils::BIAS_GYRO_IDX);
                v_current = v_linearized;
                ba_current = ba_linearized;
                bg_current = bg_linearized;
            }
            T_wc_current = pose_linearized;
        } else {
            if (use_imu) {
                delta += inc;
                v_current =
                    v_linearized + delta.segment<tassel_utils::SPEED_SIZE>(tassel_utils::SPEED_IDX);
                ba_current = ba_linearized +
                             delta.segment<tassel_utils::BIAS_ACC_SIZE>(tassel_utils::BIAS_ACC_IDX);
                bg_current = bg_linearized + delta.segment<tassel_utils::BIAS_GYRO_SIZE>(
                                                 tassel_utils::BIAS_GYRO_IDX);
            } else {
                delta.segment<tassel_utils::POSE_SIZE>(tassel_utils::POSE_IDX) +=
                    inc.segment<tassel_utils::POSE_SIZE>(tassel_utils::POSE_IDX);
            }
            T_wc_current =
                pose_linearized *
                Sophus::SE3d::exp(delta.segment<tassel_utils::POSE_SIZE>(tassel_utils::POSE_IDX));
        }
    }

    Eigen::VectorXd get_delta(bool use_imu = false) const {
        if (use_imu) {
            return delta;
        } else {
            return delta.segment<tassel_utils::POSE_SIZE>(tassel_utils::POSE_IDX);
        }
    }

    inline bool isLinearized() const { return linearized; }

    inline void updateLinearizationPoint() {
        if (linearized) {
            pose_linearized = T_wc_current;
            ba_linearized = ba_current;
            bg_linearized = bg_current;
            v_linearized = v_current;
            delta.setZero();
        }
    }

    inline void restore() {
        delta = storage_delta;
        T_wc_current = storage_T_wc_current;
        ba_current = storage_ba_current;
        bg_current = storage_bg_current;
        v_current = storage_v_current;
        pose_linearized = storage_pose_linearized;
        ba_linearized = storage_ba_linearized;
        bg_linearized = storage_bg_linearized;
        v_linearized = storage_v_linearized;
    }

    inline void save() {
        storage_delta = delta;
        storage_T_wc_current = T_wc_current;
        storage_ba_current = ba_current;
        storage_bg_current = bg_current;
        storage_v_current = v_current;
        storage_pose_linearized = pose_linearized;
        storage_ba_linearized = ba_linearized;
        storage_bg_linearized = bg_linearized;
        storage_v_linearized = v_linearized;
    }

    inline void reset() {
        linearized = false;
        delta = Eigen::Vector<double, tassel_utils::TOTAL_SIZE>::Zero();
        ba_linearized = Eigen::Vector3d::Zero();
        bg_linearized = Eigen::Vector3d::Zero();
        v_linearized = Eigen::Vector3d::Zero();
        ba_current = Eigen::Vector3d::Zero();
        bg_current = Eigen::Vector3d::Zero();
        v_current = Eigen::Vector3d::Zero();
        T_wc_current = Pose();
        pose_linearized = Pose();
    }

    void init_pose(Pose init_pose) {
        T_wc_current = init_pose;
        pose_linearized = init_pose;
        storage_pose_linearized = init_pose;
        storage_T_wc_current = init_pose;
    }

private:
    // 线性化点
    Pose pose_linearized;
    Eigen::Vector3d v_linearized;
    Eigen::Vector3d ba_linearized;
    Eigen::Vector3d bg_linearized;
    bool linearized;
    Eigen::VectorXd delta;

    // 优化后的状态
    Pose T_wc_current;
    Eigen::Vector3d v_current;
    Eigen::Vector3d ba_current;
    Eigen::Vector3d bg_current;

    // 保存的状态，用于步长不合适的时候恢复状态
    Pose storage_pose_linearized;
    Eigen::Vector3d storage_v_linearized;
    Eigen::Vector3d storage_ba_linearized;
    Eigen::Vector3d storage_bg_linearized;
    Eigen::VectorXd storage_delta;
    Pose storage_T_wc_current;
    Eigen::Vector3d storage_v_current;
    Eigen::Vector3d storage_ba_current;
    Eigen::Vector3d storage_bg_current;
};

struct State {
    explicit State(int max_frame_count_ = 10, bool use_imu = false)
        : max_frame_count(max_frame_count_), cur_frame_count(0), use_imu(use_imu) {
        poses.resize(max_frame_count);
        if (max_frame_count < 1) {
            throw std::runtime_error("max_frame_count must be greater than 0");
        }
    };
    int max_frame_count;
    int cur_frame_count;
    std::vector<PoseVelBiasState> poses;
    bool use_imu;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
