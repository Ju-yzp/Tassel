#include "feature.h"

#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "reprojection.h"
#include "state/state.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
Feature::Feature(int host_frame_index, size_t max_capacity)
    : host_frame_index(host_frame_index), estimated_depth(InvalidDepth) {
    observations.reserve(max_capacity);
}

void Feature::monoTriangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, double min_depth) {
    if (estimated_depth != InvalidDepth || observations.size() <= 1) {
        return;
    }

    const int host_index = host_frame_index;
    if (host_index < 0 || host_index > state.latest_frame_index) {
        return;
    }
    Eigen::Matrix3d reference_r;
    Eigen::Vector3d reference_t;
    if (!compensatedCameraPose(
            state.frames[host_index], observations.front().sync_delay, state.delay_time, ric, tic,
            reference_r, reference_t)) {
        throw std::runtime_error("Feature host camera pose is not finite");
    }
    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    std::vector<Eigen::Vector2d> uvs;
    poses.reserve(observations.size());
    uvs.reserve(observations.size());

    Eigen::Matrix<double, 3, 4> reference_pose = Eigen::Matrix<double, 3, 4>::Identity();
    poses.push_back(reference_pose);
    uvs.push_back(observations[0].uv.head<2>());

    for (size_t obs_idx = 1; obs_idx < observations.size(); ++obs_idx) {
        const auto& observation = observations[obs_idx];
        const int current_frame_index = observationFrameIndex(obs_idx);
        if (current_frame_index > state.latest_frame_index) {
            throw std::logic_error("Feature observation index is outside the active window");
        }
        Eigen::Matrix3d cur_r;
        Eigen::Vector3d cur_t;
        if (!compensatedCameraPose(
                state.frames[current_frame_index], observation.sync_delay, state.delay_time, ric,
                tic, cur_r, cur_t)) {
            throw std::runtime_error("Feature target camera pose is not finite");
        }
        Eigen::Matrix3d dr = cur_r.transpose() * reference_r;
        Eigen::Vector3d dt = cur_r.transpose() * (reference_t - cur_t);
        Eigen::Matrix<double, 3, 4> pose;
        pose.block<3, 3>(0, 0) = dr;
        pose.block<3, 1>(0, 3) = dt;
        poses.push_back(pose);
        uvs.push_back(observation.uv.head<2>());
    }

    double cond;
    Eigen::Vector4d h = tassel_utils::triangulateMultiView(poses, uvs, &cond);
    if (std::isfinite(cond) && cond < 1e6 && std::abs(h(3)) > 1e-12) {
        const double depth = tassel_utils::dehomogenize(h).z();
        if (std::isfinite(depth) && depth > min_depth) {
            estimated_depth = depth;
        }
    }
}

void Feature::removeFrame(
    int frame_index, const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    const int observation_index = frame_index - host_frame_index;
    if (observation_index < 0 || observation_index >= static_cast<int>(observations.size())) {
        return;
    }
    auto removed_it = observations.begin() + observation_index;

    if (host_frame_index == frame_index && observations.size() > 1) {
        const int new_host_index = frame_index + 1;
        if (transferHost(new_host_index, state, ric, tic)) {
            observations.erase(observations.begin() + 1);
        } else {
            observations.erase(removed_it);
            host_frame_index = new_host_index;
            estimated_depth = InvalidDepth;
        }
        return;
    }
    removeFrameObservation(frame_index);
}

bool Feature::transferHost(
    int new_host_index, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    const int old_frame_index = host_frame_index;
    const int new_observation_index = new_host_index - host_frame_index;
    if (old_frame_index < 0 || new_host_index > state.latest_frame_index ||
        new_observation_index <= 0 ||
        new_observation_index >= static_cast<int>(observations.size()) ||
        estimated_depth == InvalidDepth) {
        return false;
    }
    const int new_frame_index = new_host_index;
    auto old_host_it = observations.begin();
    auto new_host_it = observations.begin() + new_observation_index;

    Eigen::Vector3d pj_in_C;
    if (!reprojectToTargetCamera(
            state.frames[old_frame_index], state.frames[new_frame_index], old_host_it->uv,
            estimated_depth, old_host_it->sync_delay, new_host_it->sync_delay, state.delay_time,
            ric, tic, pj_in_C)) {
        return false;
    }

    estimated_depth = pj_in_C.z();
    host_frame_index = new_host_index;
    std::iter_swap(observations.begin(), new_host_it);
    return true;
}

void Feature::removeFrameObservation(int frame_index) {
    const int observation_index = frame_index - host_frame_index;
    if (observation_index >= 0 && observation_index < static_cast<int>(observations.size())) {
        observations.erase(observations.begin() + observation_index);
    }
}
}  // namespace tassel_core
