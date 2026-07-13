#include "feature.h"

#include <Eigen/SVD>
#include <cmath>
#include <sophus/so3.hpp>
#include <vector>

#include "state/state.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
Feature::Feature(size_t start_frame_id, size_t max_capacity)
    : start_frame_id(start_frame_id), estimated_depth(INVALID_DEPTH) {
    observations.reserve(max_capacity);
}

void Feature::monoTriangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double min_translation, double min_depth, double max_depth) {
    if (estimated_depth != INVALID_DEPTH || observations.size() <= 1) {
        return;
    }

    Eigen::Matrix3d reference_r = state.Rs[start_frame_id] * ric;
    Eigen::Vector3d reference_t = state.Rs[start_frame_id] * tic + state.Ps[start_frame_id];
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
        int cur_frame_id = static_cast<int>(start_frame_id + obs_idx);
        Eigen::Matrix3d cur_r = state.Rs[cur_frame_id] * ric;
        Eigen::Vector3d cur_t = state.Rs[cur_frame_id] * tic + state.Ps[cur_frame_id];
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

    if (poses.size() < 1) return;

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

void Feature::removeOldest(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (start_frame_id == 0) {
        if (estimated_depth != INVALID_DEPTH && observations.size() > 1) {
            const double dt_i = state.delay_time - observations[0].applied_delay;
            const double dt_j = state.delay_time - observations[1].applied_delay;
            const Eigen::Matrix3d A_i =
                Sophus::SO3d::exp((state.gyro_vec[0] - state.Bgs[0]) * dt_i).matrix();
            const Eigen::Matrix3d A_j =
                Sophus::SO3d::exp((state.Bgs[1] - state.gyro_vec[1]) * dt_j).matrix();

            const Eigen::Vector3d pi_in_C = estimated_depth * observations[0].uv;
            const Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
            const Eigen::Vector3d pi_in_W =
                state.Rs[0] * A_i * pi_in_I + state.Ps[0] + state.Vs[0] * dt_i +
                0.5 * state.Rs[0] * (state.acc_vec[0] - state.Bas[0]) * dt_i * dt_i;
            const Eigen::Vector3d pj_in_I =
                A_j * state.Rs[1].transpose() *
                (pi_in_W - (state.Ps[1] + state.Vs[1] * dt_j +
                            0.5 * state.Rs[1] * (state.acc_vec[1] - state.Bas[1]) * dt_j * dt_j));
            const Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);
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
