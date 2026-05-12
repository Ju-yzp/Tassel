#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace tassel_core {
using Pose = Sophus::SE3d;

inline void increasePose(const Eigen::Vector<double, 6> delta, Pose& pose) {
    pose = pose * Sophus::SE3d::exp(delta);
}
// vo系统（只带位姿信息）
struct PoseStateWithLin {
    PoseStateWithLin() : linearized(false), delta(Eigen::Vector<double, 6>::Zero()) {}

    PoseStateWithLin(const Sophus::SE3d& T_wc, bool linearized = false)
        : pose_linearized(T_wc),
          linearized(linearized),
          delta(Eigen::Vector<double, 6>::Zero()),
          T_wc_current(T_wc) {}

    inline void setLinearized() {
        linearized = true;
        if (!delta.isApproxToConstant(0)) {
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
    };

    inline Pose get_pose_linearized() const { return pose_linearized; }

    inline void applyDelta(const Eigen::Vector<double, 6>& delta) {
        if (!linearized) {
            increasePose(delta, pose_linearized);
            T_wc_current = pose_linearized;
        } else {
            this->delta += delta;
            T_wc_current = pose_linearized;
            increasePose(delta, T_wc_current);
        }
    }

    Eigen::Vector<double, 6> get_delta() const { return delta; }

    inline bool isLinearized() const { return linearized; }

    inline void restore() {
        delta = storage_delta;
        T_wc_current = storage_T_wc_current;
        pose_linearized = storage_pose_linearized;
    }

    inline void save() {
        storage_delta = delta;
        storage_T_wc_current = T_wc_current;
        storage_pose_linearized = pose_linearized;
    }

    Pose get_optimized_pose() const { return optimized_pose; }

    void set_optimized_pose(const Pose& optimized_pose) { this->optimized_pose = optimized_pose; }

    inline void applyDeltaToOptimizedPose(const Eigen::Vector<double, 6>& delta) {
        increasePose(delta, optimized_pose);
    }

private:
    Pose pose_linearized;
    bool linearized;  // 线性化标志
    Eigen::Vector<double, 6> delta;
    Pose T_wc_current;

    // 旧状态，用于优化失败后恢复原本状态
    Pose storage_pose_linearized;
    Eigen::Vector<double, 6> storage_delta;
    Pose storage_T_wc_current;

    Pose optimized_pose;
};

struct State {
    explicit State(int max_frame_count_ = 0) : max_frame_count(max_frame_count_) {
        poses.resize(max_frame_count);
    };
    int max_frame_count;  // 滑动窗口最大帧数
    int cur_frame_count;  // 当前帧数
    std::vector<PoseStateWithLin> poses;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
