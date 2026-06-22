#include "feature.h"

#include <spdlog/spdlog.h>

#include <Eigen/SVD>
#include <vector>

#include "state/state.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
Feature::Feature(size_t max_capacity) : start_frame_id(0), estimated_depth(INVALID_DEPTH) {
    observations.reserve(max_capacity);
}

Feature::Feature(size_t start_frame_id, size_t max_capacity)
    : start_frame_id(start_frame_id), estimated_depth(INVALID_DEPTH) {
    observations.reserve(max_capacity);
}

Feature::Feature() : start_frame_id(0) { observations.reserve(15); }

void Feature::stereoTriangulate(
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1,
    const Eigen::Vector3d& tic1, double min_depth, double max_depth) {
    if (estimated_depth != INVALID_DEPTH || !observations[0].is_stereo) {
        return;
    }

    Eigen::Matrix<double, 3, 4> pose0 = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> pose1;
    pose1.block<3, 3>(0, 0) = ric1.transpose() * ric;
    pose1.block<3, 1>(0, 3) = ric1.transpose() * (tic - tic1);

    double cond;
    Eigen::Vector4d h = tassel_utils::triangulateTwoView(
        pose0, observations[0].uv.head<2>(), pose1, observations[0].uv_r.head<2>(), &cond);
    if (cond < 1e6) {
        Eigen::Vector3d p = tassel_utils::dehomogenize(h);
        if (p.z() > min_depth && p.z() < max_depth) {
            estimated_depth = p.z();
        }
    }
}

void Feature::monoTriangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double min_translation, double min_depth, double max_depth) {
    if (estimated_depth != INVALID_DEPTH || observations.size() <= 1) {
        return;
    }

    Eigen::Matrix3d reference_r = state.Rs[start_frame_id];
    Eigen::Vector3d reference_t = reference_r * tic + state.Ps[start_frame_id];

    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    std::vector<Eigen::Vector2d> uvs;

    int cur_frame_id = start_frame_id;
    for (auto& observation : observations) {
        if (cur_frame_id >= state.cur_frame_count) break;

        Eigen::Matrix3d cur_r = state.Rs[cur_frame_id] * ric;
        Eigen::Vector3d cur_t = state.Rs[cur_frame_id] * tic + state.Ps[cur_frame_id];
        Eigen::Matrix3d dr = cur_r.transpose() * reference_r;
        Eigen::Vector3d dt = cur_r.transpose() * (reference_t - cur_t);

        if (dt.norm() > min_translation) {
            Eigen::Matrix<double, 3, 4> pose;
            pose.block<3, 3>(0, 0) = dr;
            pose.block<3, 1>(0, 3) = dt;
            poses.push_back(pose);
            uvs.push_back(observation.uv.head<2>());
        }
        ++cur_frame_id;
    }

    if (poses.size() < 2) return;

    double cond;
    Eigen::Vector4d h = tassel_utils::triangulateMultiView(poses, uvs, &cond);
    if (cond < 1e6) {
        Eigen::Vector3d p = tassel_utils::dehomogenize(h);
        if (p.z() > min_depth && p.z() < max_depth) {
            estimated_depth = p.z();
        }
    }
}

void Feature::removeOldest(
    const Eigen::Matrix3d& prev_r, const Eigen::Vector3d& prev_t, const Eigen::Matrix3d& cur_r,
    const Eigen::Vector3d& cur_t, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (start_frame_id == 0) {
        if (estimated_depth != INVALID_DEPTH && observations.size() > 1) {
            Eigen::Vector3d pi_in_C = estimated_depth * observations[0].uv;
            Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
            Eigen::Vector3d pi_in_W = prev_r * pi_in_I + prev_t;
            Eigen::Vector3d pj_in_I = cur_r.transpose() * (pi_in_W - cur_t);
            Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);
            if (pj_in_C.z() > 0) {
                estimated_depth = pj_in_C.z();
            } else {
                estimated_depth = INVALID_DEPTH;
            }
        }
        observations.erase(observations.begin());
    } else {
        --start_frame_id;
    }
}

void Feature::removeNewest(size_t frame_count) {
    if (start_frame_id + observations.size() == frame_count + 1) {
        observations.pop_back();
    }
}
}  // namespace tassel_core
