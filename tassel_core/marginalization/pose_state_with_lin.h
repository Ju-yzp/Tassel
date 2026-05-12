#ifndef TASSEL_CORE_MARGINALIZATION_POSE_STATE_WITH_LIN_H_
#define TASSEL_CORE_MARGINALIZATION_POSE_STATE_WITH_LIN_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.hpp>

namespace tassel_core {

using VecN = Eigen::Matrix<double, 6, 1>;

struct PoseState {
    int64_t ts;  // 使用纳秒为单位的时间戳
    Sophus::SE3d T_wc;
    static void increasePose(const VecN& delta, Sophus::SE3d& T) {  // 默认使用右乘
        T = T * Sophus::SE3d::exp(delta);
    }
};

// 适用于VO系统，不带速度，偏置约束,只含位姿信息
class PoseStateWithLin {
public:
    PoseStateWithLin() : linearized(false), delta(VecN::Zero()) {}

    PoseStateWithLin(int64_t ts, const Sophus::SE3d& T_wc, bool linearized = false)
        : pose_linearized(ts, T_wc), linearized(linearized), delta(VecN::Zero()) {}

    void setLinearized() {
        linearized = true;
        if (!delta.isApproxToConstant(0)) {
            throw std::runtime_error("delta is not zero");
        }
        T_wc_current = pose_linearized.T_wc;
    }

    Sophus::SE3d get_pose() const {
        if (!linearized) {
            return T_wc_current;
        } else {
            return pose_linearized.T_wc;
        }
    };

    Sophus::SE3d get_pose_linearized() const { return pose_linearized.T_wc; }

    void applyDelta(const VecN& delta) {
        if (!linearized) {
            PoseState::increasePose(delta, pose_linearized.T_wc);

        } else {
            this->delta += delta;
            T_wc_current = pose_linearized.T_wc;
            PoseState::increasePose(delta, T_wc_current);
        }
    }

    VecN get_delta() const { return delta; }

    inline bool isLinearized() const { return linearized; }

    inline int64_t get_ts() const { return pose_linearized.ts; }

    inline void restore() {
        delta = storage_delta;
        T_wc_current = storage_T_wc_current;
        pose_linearized = storage_pose_linearized;
    }

    void save() {
        storage_delta = delta;
        storage_T_wc_current = T_wc_current;
        storage_pose_linearized = pose_linearized;
    }

private:
    PoseState pose_linearized;
    bool linearized;  // 线性化标志
    VecN delta;
    Sophus::SE3d T_wc_current;

    // 旧状态，用于优化失败后恢复原本状态
    PoseState storage_pose_linearized;
    VecN storage_delta;
    Sophus::SE3d storage_T_wc_current;
};
}  // namespace tassel_core
#endif
