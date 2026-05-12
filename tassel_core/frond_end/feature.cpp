#include "feature.h"

#include <spdlog/spdlog.h>

#include <Eigen/SVD>
#include "state/state.h"

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

    Eigen::Vector3d uv = observations[0].uv;
    Eigen::Vector3d uv_r = observations[0].uv_r;

    Eigen::Matrix<double, 3, 4> pose0 = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> pose1;
    pose1.block<3, 3>(0, 0) = ric1.transpose() * ric;
    pose1.block<3, 1>(0, 3) = ric1.transpose() * (tic - tic1);
    Eigen::Matrix4d A;
    A.row(0) = uv.x() * pose0.row(2) - pose0.row(0);
    A.row(1) = uv.y() * pose0.row(2) - pose0.row(1);
    A.row(2) = uv_r.x() * pose1.row(2) - pose1.row(0);
    A.row(3) = uv_r.y() * pose1.row(2) - pose1.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);
    Eigen::Vector3d point_3d = homogeneous_point.head<3>() / homogeneous_point(3);
    Eigen::Vector4d singular_values = svd.singularValues();
    double condition_number = singular_values(0) / singular_values(3);
    if (condition_number < 1e6 && point_3d.z() > min_depth && point_3d.z() < max_depth) {
        estimated_depth = point_3d.z();
        return;
    }
}

void Feature::monoTriangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, double min_depth,
    double max_depth) {
    if (estimated_depth != INVALID_DEPTH) {
        return;
    }
    if (observations.size() > 2) {
        Eigen::MatrixXd A(2 * static_cast<int>(observations.size()), 4);
        Pose reference_pose = state.poses[start_frame_id].get_optimized_pose();
        Eigen::Matrix3d reference_r = reference_pose.rotationMatrix();
        Eigen::Vector3d reference_t = reference_r * tic + reference_pose.translation();

        int cur_frame_id = start_frame_id;

        int row = 0;
        for (auto& observation : observations) {
            Pose cur_pose = state.poses[cur_frame_id].get_optimized_pose();
            Eigen::Matrix3d cur_r = cur_pose.rotationMatrix() * ric;
            Eigen::Vector3d cur_t = cur_pose.rotationMatrix() * tic + cur_pose.translation();

            Eigen::Matrix3d dr = cur_r.transpose() * reference_r;
            Eigen::Vector3d dt = cur_r.transpose() * (reference_t - cur_t);
            Eigen::Matrix<double, 3, 4> pose;
            pose.block(0, 0, 3, 3) = dr;
            pose.block(0, 3, 3, 1) = dt;
            Eigen::Vector3d direction = observation.uv;

            A.row(row++) = direction.x() * pose.row(2) - pose.row(0);
            A.row(row++) = direction.y() * pose.row(2) - pose.row(1);
            ++cur_frame_id;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = homogeneous_point.head<3>() / homogeneous_point(3);

        Eigen::Vector4d singular_values = svd.singularValues();
        double condition_number = singular_values(0) / singular_values(3);
        if (condition_number < 1e6 && point_3d.z() > min_depth && point_3d.z() < max_depth) {
            estimated_depth = point_3d.z();
        }
    }
}

void Feature::removeOldest(
    const Eigen::Matrix3d& prev_r, const Eigen::Vector3d& prev_t, const Eigen::Matrix3d& cur_r,
    const Eigen::Vector3d& cur_t, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (start_frame_id == 0 && estimated_depth != INVALID_DEPTH) {
        if (observations.size() > 1) {
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
