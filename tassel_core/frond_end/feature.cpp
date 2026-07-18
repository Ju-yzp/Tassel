#include "feature.h"

#include <Eigen/SVD>
#include <cmath>
#include <vector>

#include "reprojection.h"
#include "state/state.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
Feature::Feature(int start_slot_, size_t max_capacity)
    : start_slot(start_slot_), estimated_depth(INVALID_DEPTH) {
    observations.reserve(max_capacity);
}

void Feature::monoTriangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double min_translation, double min_depth, double max_depth) {
    if (estimated_depth != INVALID_DEPTH || observations.size() <= 1) {
        return;
    }

    const int host_slot = start_slot;
    if (host_slot < 0 || host_slot > state.newest_slot) {
        return;
    }
    Eigen::Matrix3d reference_r = state.frames[host_slot].R * ric;
    Eigen::Vector3d reference_t = state.frames[host_slot].R * tic + state.frames[host_slot].P;
    Eigen::Vector3d reference_ray = observations[0].uv.normalized();

    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    std::vector<Eigen::Vector2d> uvs;
    poses.reserve(observations.size());
    uvs.reserve(observations.size());

    Eigen::Matrix<double, 3, 4> reference_pose = Eigen::Matrix<double, 3, 4>::Identity();
    poses.push_back(reference_pose);
    uvs.push_back(observations[0].uv.head<2>());

    for (size_t obs_idx = 1; obs_idx < observations.size(); ++obs_idx) {
        const auto& observation = observations[obs_idx];
        const int cur_slot = observationSlot(obs_idx);
        if (cur_slot > state.newest_slot) {
            continue;
        }
        Eigen::Matrix3d cur_r = state.frames[cur_slot].R * ric;
        Eigen::Vector3d cur_t = state.frames[cur_slot].R * tic + state.frames[cur_slot].P;
        Eigen::Matrix3d dr = cur_r.transpose() * reference_r;
        Eigen::Vector3d dt = cur_r.transpose() * (reference_t - cur_t);
        Eigen::Vector3d t_ref_cur = reference_r.transpose() * (cur_t - reference_t);
        Eigen::Vector3d transverse_t = t_ref_cur - reference_ray * reference_ray.dot(t_ref_cur);

        if (transverse_t.norm() > min_translation) {
            Eigen::Matrix<double, 3, 4> pose;
            pose.block<3, 3>(0, 0) = dr;
            pose.block<3, 1>(0, 3) = dt;
            poses.push_back(pose);
            uvs.push_back(observation.uv.head<2>());
        }
    }

    if (poses.size() < 2) {
        return;
    }

    double cond;
    Eigen::Vector4d h = tassel_utils::triangulateMultiView(poses, uvs, &cond);
    if (std::isfinite(cond) && cond < 1e6 && std::abs(h(3)) > 1e-12) {
        Eigen::Vector3d p_ref = tassel_utils::dehomogenize(h);
        bool positive_depth =
            std::isfinite(p_ref.z()) && p_ref.z() > min_depth && p_ref.z() < max_depth;
        for (const auto& pose : poses) {
            Eigen::Vector3d p_cur = pose.block<3, 3>(0, 0) * p_ref + pose.block<3, 1>(0, 3);
            positive_depth = positive_depth && std::isfinite(p_cur.z()) && p_cur.z() > min_depth;
        }
        if (positive_depth) {
            estimated_depth = p_ref.z();
        }
    }
}

void Feature::removeFrame(
    int frame_slot, const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    const int observation_index = frame_slot - start_slot;
    if (observation_index < 0 || observation_index >= static_cast<int>(observations.size())) {
        return;
    }
    auto removed_it = observations.begin() + observation_index;

    if (start_slot == frame_slot && observations.size() > 1) {
        const int new_host_slot = frame_slot + 1;
        if (transferHost(new_host_slot, state, ric, tic)) {
            observations.erase(observations.begin() + 1);
        } else {
            observations.erase(removed_it);
            start_slot = new_host_slot;
            estimated_depth = INVALID_DEPTH;
        }
        return;
    }
    removeFrameObservation(frame_slot);
}

bool Feature::transferHost(
    int new_host_slot, const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    const int old_slot = start_slot;
    const int new_observation_index = new_host_slot - start_slot;
    if (old_slot < 0 || new_host_slot > state.newest_slot || new_observation_index <= 0 ||
        new_observation_index >= static_cast<int>(observations.size()) ||
        estimated_depth == INVALID_DEPTH) {
        return false;
    }
    const int new_slot = new_host_slot;
    auto old_host_it = observations.begin();
    auto new_host_it = observations.begin() + new_observation_index;

    Eigen::Vector3d pj_in_C;
    if (!reprojectToTargetCamera(
            state.frames[old_slot], state.frames[new_slot], old_host_it->uv, estimated_depth,
            old_host_it->sync_delay, new_host_it->sync_delay, state.delay_time, ric, tic,
            pj_in_C)) {
        return false;
    }

    estimated_depth = pj_in_C.z();
    start_slot = new_host_slot;
    std::iter_swap(observations.begin(), new_host_it);
    return true;
}

void Feature::removeFrameObservation(int frame_slot) {
    const int observation_index = frame_slot - start_slot;
    if (observation_index >= 0 && observation_index < static_cast<int>(observations.size())) {
        observations.erase(observations.begin() + observation_index);
    }
}
}  // namespace tassel_core
