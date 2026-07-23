#ifndef TASSEL_CORE_REPROJECTION_H_
#define TASSEL_CORE_REPROJECTION_H_

#include <Eigen/Core>

namespace tassel_core {

struct FrameState;

// 以下几何辅助函数与 ReprojectionFactor 使用同一时间延迟运动模型，供三角化、
// 离群点剔除和宿主转移复用，避免这些路径采用不同的路标传播公式。
bool hostPointToWorld(
    const FrameState& host, const Eigen::Vector3d& host_uv, double host_depth,
    double host_sync_delay, double delay_time, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, Eigen::Vector3d& world_point);

bool worldPointToTargetCamera(
    const FrameState& target, const Eigen::Vector3d& world_point, double target_sync_delay,
    double delay_time, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    Eigen::Vector3d& target_point);

// 将特征点从宿主相机的观测时刻传播到目标相机的观测时刻。
// 几何关系无效时返回 false。
bool reprojectToTargetCamera(
    const FrameState& host, const FrameState& target, const Eigen::Vector3d& host_uv,
    double host_depth, double host_sync_delay, double target_sync_delay, double delay_time,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, Eigen::Vector3d& target_point);

}  // namespace tassel_core

#endif  // TASSEL_CORE_REPROJECTION_H_
