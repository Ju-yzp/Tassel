#include "feature.h"

#include <Eigen/SVD>
#include <cmath>
#include <sophus/so3.hpp>
#include <vector>

#include "state/state.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
Feature::Feature(tassel_utils::FrameId host_frame_id, size_t max_capacity)
    : host_frame_id(host_frame_id), estimated_depth(INVALID_DEPTH) {
    observations.reserve(max_capacity);
}

void Feature::monoTriangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double min_translation, double min_depth, double max_depth) {
    if (estimated_depth != INVALID_DEPTH || observations.size() <= 1) {
        return;
    }

    const int host_slot = state.findFrameSlot(host_frame_id);
    if (host_slot < 0) return;
    Eigen::Matrix3d reference_r = state.Rs[host_slot] * ric;
    Eigen::Vector3d reference_t = state.Rs[host_slot] * tic + state.Ps[host_slot];
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
        const int cur_slot = state.findFrameSlot(observation.frame_id);
        if (cur_slot < 0) continue;
        Eigen::Matrix3d cur_r = state.Rs[cur_slot] * ric;
        Eigen::Vector3d cur_t = state.Rs[cur_slot] * tic + state.Ps[cur_slot];
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

    if (poses.size() < 2) return;

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
    tassel_utils::FrameId frame_id, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    auto removed_it = std::find_if(observations.begin(), observations.end(), [&](const auto& obs) {
        return obs.frame_id == frame_id;
    });
    if (removed_it == observations.end()) return;

    if (host_frame_id == frame_id && observations.size() > 1) {
        const auto new_host_id = std::next(removed_it)->frame_id;
        if (!transferHost(new_host_id, state, ric, tic)) {
            host_frame_id = new_host_id;
            estimated_depth = INVALID_DEPTH;
            std::iter_swap(observations.begin(), std::next(removed_it));
        }
    }
    removeFrameObservation(frame_id);
}

bool Feature::transferHost(
    tassel_utils::FrameId new_host_frame_id, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    auto old_host_it = std::find_if(observations.begin(), observations.end(), [&](const auto& obs) {
        return obs.frame_id == host_frame_id;
    });
    auto new_host_it = std::find_if(observations.begin(), observations.end(), [&](const auto& obs) {
        return obs.frame_id == new_host_frame_id;
    });
    if (old_host_it == observations.end() || new_host_it == observations.end()) return false;

    const int old_slot = state.findFrameSlot(host_frame_id);
    const int new_slot = state.findFrameSlot(new_host_frame_id);
    if (old_slot < 0 || new_slot < 0 || estimated_depth == INVALID_DEPTH) return false;

    const double dt_i = state.delay_time - old_host_it->applied_delay;
    const double dt_j = state.delay_time - new_host_it->applied_delay;
    const Eigen::Matrix3d A_i =
        Sophus::SO3d::exp((state.gyro_vec[old_slot] - state.Bgs[old_slot]) * dt_i).matrix();
    const Eigen::Matrix3d A_j =
        Sophus::SO3d::exp((state.Bgs[new_slot] - state.gyro_vec[new_slot]) * dt_j).matrix();
    const Eigen::Vector3d pi_in_I = ric * (estimated_depth * old_host_it->uv) + tic;
    const Eigen::Vector3d pi_in_W =
        state.Rs[old_slot] * A_i * pi_in_I + state.Ps[old_slot] + state.Vs[old_slot] * dt_i +
        0.5 *
            (state.Rs[old_slot] * (state.acc_vec[old_slot] - state.Bas[old_slot]) -
             tassel_utils::G) *
            dt_i * dt_i;
    const Eigen::Vector3d pj_in_I =
        A_j * state.Rs[new_slot].transpose() *
        (pi_in_W - (state.Ps[new_slot] + state.Vs[new_slot] * dt_j +
                    0.5 *
                        (state.Rs[new_slot] * (state.acc_vec[new_slot] - state.Bas[new_slot]) -
                         tassel_utils::G) *
                        dt_j * dt_j));
    const Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);
    if (!pj_in_C.allFinite() || pj_in_C.z() <= 0) return false;

    estimated_depth = pj_in_C.z();
    host_frame_id = new_host_frame_id;
    std::iter_swap(observations.begin(), new_host_it);
    return true;
}

void Feature::removeFrameObservation(tassel_utils::FrameId frame_id) {
    std::erase_if(observations, [&](const auto& obs) { return obs.frame_id == frame_id; });
}
}  // namespace tassel_core
