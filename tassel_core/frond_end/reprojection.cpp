#include "reprojection.h"

#include <cmath>

#include "state/state.h"

namespace tassel_core {

bool compensatedCameraPose(
    const FrameState& frame, double sync_delay, double time_delay, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, Eigen::Matrix3d& rotation, Eigen::Vector3d& position) {
    const double dt = time_delay - sync_delay;
    if (!std::isfinite(dt)) {
        return false;
    }
    const Eigen::Vector3d omega = frame.imu_gyro - frame.gyro_bias;
    const Eigen::Vector3d acceleration = frame.imu_acc - frame.accel_bias;
    const Eigen::Matrix3d delta_rotation = Sophus::SO3d::exp(omega * dt).matrix();
    const Eigen::Matrix3d compensated_rotation = frame.rot_w_i * delta_rotation;
    const Eigen::Vector3d body_rotational_acceleration = Sophus::SO3d::hat(omega) * acceleration;
    const double dt2 = dt * dt;
    const Eigen::Vector3d world_acceleration = frame.rot_w_i * acceleration - tassel_utils::G;
    const Eigen::Vector3d world_rotational_acceleration =
        frame.rot_w_i * body_rotational_acceleration;
    const Eigen::Vector3d compensated_position =
        frame.pos_w_i + frame.vel_w * dt + 0.5 * world_acceleration * dt2 +
        (1.0 / 6.0) * world_rotational_acceleration * dt2 * dt;
    rotation = compensated_rotation * ric;
    position = compensated_rotation * tic + compensated_position;
    return rotation.allFinite() && position.allFinite();
}

bool hostPointToWorld(
    const FrameState& host, const Eigen::Vector3d& host_uv, double host_depth,
    double host_sync_delay, double time_delay, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, Eigen::Vector3d& world_point) {
    if (!std::isfinite(host_depth) || host_depth <= 0.0 || !host_uv.allFinite()) {
        return false;
    }

    Eigen::Matrix3d camera_rotation;
    Eigen::Vector3d camera_position;
    if (!compensatedCameraPose(
            host, host_sync_delay, time_delay, ric, tic, camera_rotation, camera_position)) {
        return false;
    }
    world_point = camera_rotation * (host_uv * host_depth) + camera_position;
    return world_point.allFinite();
}

bool worldPointToTargetCamera(
    const FrameState& target, const Eigen::Vector3d& world_point, double target_sync_delay,
    double time_delay, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    Eigen::Vector3d& target_point) {
    if (!world_point.allFinite()) {
        return false;
    }
    Eigen::Matrix3d camera_rotation;
    Eigen::Vector3d camera_position;
    if (!compensatedCameraPose(
            target, target_sync_delay, time_delay, ric, tic, camera_rotation, camera_position)) {
        return false;
    }
    target_point = camera_rotation.transpose() * (world_point - camera_position);
    return target_point.allFinite() && target_point.z() > 1e-12;
}

bool reprojectToTargetCamera(
    const FrameState& host, const FrameState& target, const Eigen::Vector3d& host_uv,
    double host_depth, double host_sync_delay, double target_sync_delay, double time_delay,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, Eigen::Vector3d& target_point) {
    Eigen::Vector3d world_point;
    return hostPointToWorld(
               host, host_uv, host_depth, host_sync_delay, time_delay, ric, tic, world_point) &&
           worldPointToTargetCamera(
               target, world_point, target_sync_delay, time_delay, ric, tic, target_point);
}

}  // namespace tassel_core
