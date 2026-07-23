#include "reprojection.h"

#include <cmath>
#include <sophus/so3.hpp>

#include "state/state.h"

namespace tassel_core {

bool hostPointToWorld(
    const FrameState& host, const Eigen::Vector3d& host_uv, double host_depth,
    double host_sync_delay, double delay_time, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, Eigen::Vector3d& world_point) {
    if (!std::isfinite(host_depth) || host_depth <= 0.0 || !host_uv.allFinite()) {
        return false;
    }

    const double dt_i = delay_time - host_sync_delay;
    if (!std::isfinite(dt_i)) {
        return false;
    }

    const Eigen::Matrix3d A_i = Sophus::SO3d::exp((host.gyro - host.Bg) * dt_i).matrix();
    const Eigen::Vector3d host_omega = host.gyro - host.Bg;
    const Eigen::Vector3d host_acc = host.acc - host.Ba;
    const Eigen::Vector3d host_rot_acc = host.R * Sophus::SO3d::hat(host_omega) * host_acc;
    const Eigen::Vector3d point_i = ric * (host_uv * host_depth) + tic;
    world_point = host.R * A_i * point_i + host.P + host.V * dt_i +
                  0.5 * (host.R * host_acc - tassel_utils::G) * dt_i * dt_i +
                  (1.0 / 6.0) * host_rot_acc * dt_i * dt_i * dt_i;
    return world_point.allFinite();
}

bool worldPointToTargetCamera(
    const FrameState& target, const Eigen::Vector3d& world_point, double target_sync_delay,
    double delay_time, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    Eigen::Vector3d& target_point) {
    const double dt_j = delay_time - target_sync_delay;
    if (!world_point.allFinite() || !std::isfinite(dt_j)) {
        return false;
    }

    const Eigen::Matrix3d A_j = Sophus::SO3d::exp((target.Bg - target.gyro) * dt_j).matrix();
    const Eigen::Vector3d target_omega = target.gyro - target.Bg;
    const Eigen::Vector3d target_acc = target.acc - target.Ba;
    const Eigen::Vector3d target_rot_acc = target.R * Sophus::SO3d::hat(target_omega) * target_acc;
    const Eigen::Vector3d point_j =
        A_j * target.R.transpose() *
        (world_point - (target.P + target.V * dt_j +
                        0.5 * (target.R * target_acc - tassel_utils::G) * dt_j * dt_j +
                        (1.0 / 6.0) * target_rot_acc * dt_j * dt_j * dt_j));
    target_point = ric.transpose() * (point_j - tic);
    return target_point.allFinite() && target_point.z() > 1e-12;
}

bool reprojectToTargetCamera(
    const FrameState& host, const FrameState& target, const Eigen::Vector3d& host_uv,
    double host_depth, double host_sync_delay, double target_sync_delay, double delay_time,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, Eigen::Vector3d& target_point) {
    Eigen::Vector3d world_point;
    return hostPointToWorld(
               host, host_uv, host_depth, host_sync_delay, delay_time, ric, tic, world_point) &&
           worldPointToTargetCamera(
               target, world_point, target_sync_delay, delay_time, ric, tic, target_point);
}

}  // namespace tassel_core
