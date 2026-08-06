#include "reprojection.h"

#include <cmath>

#include "state/frame_kinematics.h"
#include "state/state.h"

namespace tassel_core {

bool compensatedCameraPose(
    const FrameState& frame, double sync_delay, double delay_time, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, Eigen::Matrix3d& rotation, Eigen::Vector3d& position) {
    const double dt = delay_time - sync_delay;
    if (!std::isfinite(dt)) {
        return false;
    }
    const FrameKinematics kinematics = propagateFrameKinematics(
        frame.R, frame.P, frame.V, frame.gyro, frame.acc, frame.Bg, frame.Ba, dt);
    rotation = kinematics.rotation * ric;
    position = kinematics.rotation * tic + kinematics.position;
    return rotation.allFinite() && position.allFinite();
}

bool hostPointToWorld(
    const FrameState& host, const Eigen::Vector3d& host_uv, double host_depth,
    double host_sync_delay, double delay_time, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, Eigen::Vector3d& world_point) {
    if (!std::isfinite(host_depth) || host_depth <= 0.0 || !host_uv.allFinite()) {
        return false;
    }

    Eigen::Matrix3d camera_rotation;
    Eigen::Vector3d camera_position;
    if (!compensatedCameraPose(
            host, host_sync_delay, delay_time, ric, tic, camera_rotation, camera_position)) {
        return false;
    }
    world_point = camera_rotation * (host_uv * host_depth) + camera_position;
    return world_point.allFinite();
}

bool worldPointToTargetCamera(
    const FrameState& target, const Eigen::Vector3d& world_point, double target_sync_delay,
    double delay_time, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    Eigen::Vector3d& target_point) {
    if (!world_point.allFinite()) {
        return false;
    }
    Eigen::Matrix3d camera_rotation;
    Eigen::Vector3d camera_position;
    if (!compensatedCameraPose(
            target, target_sync_delay, delay_time, ric, tic, camera_rotation, camera_position)) {
        return false;
    }
    target_point = camera_rotation.transpose() * (world_point - camera_position);
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
