#include "state/state.h"

#include "feature_manager/feature_manager.h"

#include <Eigen/Geometry>

namespace tassel_core {
void State::paramsToState(bool use_imu) {
    if (max_frame_count == 0) {
        throw std::runtime_error("State::paramsToState: max_frame_count is 0");
    }
    if (param_pose.size() != max_frame_count + 1 || Rs.size() != max_frame_count + 1 ||
        Ps.size() != max_frame_count + 1) {
        throw std::runtime_error("State::paramsToState: pose vector size mismatch");
    }
    if (use_imu) {
        if (param_speed.size() != max_frame_count + 1 || Vs.size() != max_frame_count + 1 ||
            Bas.size() != max_frame_count + 1 || Bgs.size() != max_frame_count + 1) {
            throw std::runtime_error("State::paramsToState: IMU vector size mismatch");
        }
    }

    for (size_t i = 0; i < max_frame_count + 1; ++i) {
        Eigen::Quaterniond q(
            param_pose[i][6], param_pose[i][3], param_pose[i][4], param_pose[i][5]);
        Rs[i] = q.toRotationMatrix();
        Ps[i] = Eigen::Vector3d(param_pose[i][0], param_pose[i][1], param_pose[i][2]);
    }
    if (use_imu) {
        for (size_t i = 0; i < max_frame_count + 1; ++i) {
            Vs[i] = Eigen::Vector3d(param_speed[i][0], param_speed[i][1], param_speed[i][2]);
            Bas[i] = Eigen::Vector3d(param_speed[i][3], param_speed[i][4], param_speed[i][5]);
            Bgs[i] = Eigen::Vector3d(param_speed[i][6], param_speed[i][7], param_speed[i][8]);
        }
    }

    if (depth_ptrs.size() != param_inv_depth.size()) {
        throw std::runtime_error("State::paramsToState: depth_ptrs/param_inv_depth size mismatch");
    }

    for (size_t i = 0; i < param_inv_depth.size(); ++i) {
        (*depth_ptrs[i]) = 1.0 / param_inv_depth[i];
    }

    param_inv_depth.clear();
    depth_ptrs.clear();
}

void State::stateToParams(bool use_imu) {
    if (max_frame_count == 0) {
        throw std::runtime_error("State::stateToParams: max_frame_count is 0");
    }
    if (Rs.size() != max_frame_count + 1 || Ps.size() != max_frame_count + 1 ||
        param_pose.size() != max_frame_count + 1) {
        throw std::runtime_error("State::stateToParams: pose vector size mismatch");
    }
    if (use_imu) {
        if (Vs.size() != max_frame_count + 1 || Bas.size() != max_frame_count + 1 ||
            Bgs.size() != max_frame_count + 1 || param_speed.size() != max_frame_count + 1) {
            throw std::runtime_error("State::stateToParams: IMU vector size mismatch");
        }
    }

    for (size_t i = 0; i < max_frame_count + 1; ++i) {
        Eigen::Quaterniond q(Rs[i]);
        param_pose[i][0] = Ps[i].x();
        param_pose[i][1] = Ps[i].y();
        param_pose[i][2] = Ps[i].z();
        param_pose[i][3] = q.x();
        param_pose[i][4] = q.y();
        param_pose[i][5] = q.z();
        param_pose[i][6] = q.w();
    }

    if (use_imu) {
        for (size_t i = 0; i < max_frame_count + 1; ++i) {
            param_speed[i][0] = Vs[i].x();
            param_speed[i][1] = Vs[i].y();
            param_speed[i][2] = Vs[i].z();
            param_speed[i][3] = Bas[i].x();
            param_speed[i][4] = Bas[i].y();
            param_speed[i][5] = Bas[i].z();
            param_speed[i][6] = Bgs[i].x();
            param_speed[i][7] = Bgs[i].y();
            param_speed[i][8] = Bgs[i].z();
        }
    }
}

void State::forwardSlide() {
    if (cur_frame_count != max_frame_count) {
        return;
    }
    for (int i = 1; i < max_frame_count + 1; ++i) {
        Rs[i - 1] = Rs[i];
        Ps[i - 1] = Ps[i];
        Vs[i - 1] = Vs[i];
        Bas[i - 1] = Bas[i];
        Bgs[i - 1] = Bgs[i];
        param_pose[i - 1] = param_pose[i];
        param_speed[i - 1] = param_speed[i];
    }
}
}  // namespace tassel_core
