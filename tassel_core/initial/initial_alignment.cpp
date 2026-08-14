#include "initial/initial_alignment.h"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sophus/so3.hpp>
#include <vector>

#include "tassel_utils/rotation.h"

namespace tassel_core {
namespace {

constexpr double kMaxAlignmentCondition = 1e8;

bool solveObservableSystem(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& rhs, const char* stage,
    Eigen::VectorXd& solution) {
    if (jacobian.rows() < jacobian.cols() || rhs.size() != jacobian.rows() ||
        !jacobian.allFinite() || !rhs.allFinite()) {
        spdlog::warn(
            "{} solve failed: rows={}, cols={}, rhs={}, finite={}", stage, jacobian.rows(),
            jacobian.cols(), rhs.size(), jacobian.allFinite() && rhs.allFinite());
        return false;
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& singular_values = svd.singularValues();
    const double largest = singular_values[0];
    const double smallest = singular_values[singular_values.size() - 1];
    const double condition = largest / smallest;
    if (!std::isfinite(condition) || smallest <= 0.0 || condition > kMaxAlignmentCondition) {
        spdlog::warn(
            "{} solve failed: rank-deficient or ill-conditioned system, condition={:.3e}, "
            "limit={:.3e}",
            stage, condition, kMaxAlignmentCondition);
        return false;
    }

    solution = svd.solve(rhs);
    if (!solution.allFinite()) {
        spdlog::warn("{} solve failed: non-finite solution", stage);
        return false;
    }
    spdlog::debug("{} condition={:.3e}", stage, condition);
    return true;
}

}  // namespace

bool linearAlignment(
    const std::vector<Eigen::Matrix3d>& rotations, const std::vector<Eigen::Vector3d>& positions,
    std::vector<Eigen::Vector3d>& velocities, const std::vector<Eigen::Vector3d>& delta_velocities,
    const std::vector<Eigen::Vector3d>& delta_positions, const std::vector<double>& dts,
    Eigen::Vector3d& gravity, double& scale, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double gravity_norm_tolerance, double target_gravity_norm) {
    const int frame_count = static_cast<int>(rotations.size());
    if (frame_count < 2 || positions.size() != rotations.size() ||
        velocities.size() != rotations.size() || delta_velocities.size() != rotations.size() - 1 ||
        delta_positions.size() != rotations.size() - 1 || dts.size() != rotations.size() - 1) {
        spdlog::error(
            "LinearAlignment input failed: R={}, P={}, V={}, dv={}, dp={}, dt={}", rotations.size(),
            positions.size(), velocities.size(), delta_velocities.size(), delta_positions.size(),
            dts.size());
        return false;
    }
    for (size_t i = 0; i < dts.size(); ++i) {
        const double dt = dts[i];
        if (!std::isfinite(dt) || dt <= 0.0) {
            spdlog::error("LinearAlignment input failed: interval={}, dt={}", i, dt);
            return false;
        }
    }
    const int state_size = frame_count * 3 + 4;

    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6 * (frame_count - 1), state_size);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(6 * (frame_count - 1));
    for (int i = 0; i < frame_count - 1; ++i) {
        const int j = i + 1;
        const double dt = dts[i];
        const int velocity_column = i * 3;

        Eigen::Matrix<double, 6, 10> interval_jacobian;
        interval_jacobian.setZero();
        Eigen::Matrix<double, 6, 1> interval_rhs;
        interval_rhs.setZero();

        interval_jacobian.block<3, 3>(0, 0) =
            -dt * ric * rotations[i].transpose() * ric.transpose();
        interval_jacobian.block<3, 3>(0, 6) =
            0.5 * dt * dt * ric * rotations[i].transpose() * ric.transpose();
        interval_jacobian.block<3, 1>(0, 9) =
            ric * rotations[i].transpose() * (positions[j] - positions[i]) / 100.0;
        interval_rhs.block<3, 1>(0, 0) =
            ric * rotations[i].transpose() * rotations[j] * ric.transpose() * tic - tic +
            delta_positions[i];

        interval_jacobian.block<3, 3>(3, 0) = -ric * rotations[i].transpose() * ric.transpose();
        interval_jacobian.block<3, 3>(3, 3) = ric * rotations[i].transpose() * ric.transpose();
        interval_jacobian.block<3, 3>(3, 6) = ric * rotations[i].transpose() * ric.transpose() * dt;
        interval_rhs.block<3, 1>(3, 0) = delta_velocities[i];

        const int row = 6 * i;
        jacobian.block<6, 3>(row, velocity_column) = interval_jacobian.leftCols<3>();
        jacobian.block<6, 3>(row, velocity_column + 3) = interval_jacobian.middleCols<3>(3);
        jacobian.block<6, 4>(row, state_size - 4) = interval_jacobian.rightCols<4>();
        rhs.segment<6>(row) = interval_rhs;
    }

    Eigen::VectorXd solution;
    if (!solveObservableSystem(jacobian, rhs, "LinearAlignment", solution)) {
        return false;
    }

    scale = solution(state_size - 1) / 100.0;
    gravity = solution.segment<3>(state_size - 4);

    spdlog::info(
        "LinearAlignment: |g|={:.4f} g=({:.3f},{:.3f},{:.3f}) s={:.4f}", gravity.norm(),
        gravity.x(), gravity.y(), gravity.z(), scale);
    if (!std::isfinite(scale) || !gravity.allFinite() || scale <= 0 ||
        std::abs(gravity.norm() - target_gravity_norm) > gravity_norm_tolerance) {
        spdlog::warn(
            "LinearAlignment validity failed: scale={}, gravity_norm={}, target={}, tolerance={}",
            scale, gravity.norm(), target_gravity_norm, gravity_norm_tolerance);
        return false;
    }

    for (int i = 0; i < frame_count; ++i) {
        velocities[i] = solution.segment<3>(i * 3);
    }

    return true;
}

bool refineGravitySpeeds(
    std::vector<Eigen::Vector3d>& velocities, const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Vector3d>& delta_velocities,
    const std::vector<Eigen::Vector3d>& delta_positions, const std::vector<double>& dts,
    Eigen::Vector3d& gravity, double& scale, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    double gravity_norm) {
    const int frame_count = static_cast<int>(velocities.size());
    if (frame_count < 2 || rotations.size() != velocities.size() ||
        positions.size() != velocities.size() || delta_velocities.size() != velocities.size() - 1 ||
        delta_positions.size() != velocities.size() - 1 || dts.size() != velocities.size() - 1 ||
        !gravity.allFinite() || gravity.norm() < 1e-12) {
        spdlog::error(
            "GravityRefinement input failed: R={}, P={}, V={}, dv={}, dp={}, dt={}, "
            "gravity_norm={}",
            rotations.size(), positions.size(), velocities.size(), delta_velocities.size(),
            delta_positions.size(), dts.size(), gravity.norm());
        return false;
    }
    const int state_size = frame_count * 3 + 3;
    const int gravity_column = frame_count * 3;
    const int scale_column = frame_count * 3 + 2;

    Eigen::Vector3d gravity_direction = gravity.normalized();

    for (int iter = 0; iter < 4; ++iter) {
        const Eigen::Matrix<double, 2, 3> tangent_basis =
            tassel_utils::tangentBasis(gravity_direction);
        const Eigen::Matrix<double, 3, 2> gravity_jacobian =
            gravity_norm * tangent_basis.transpose();
        const Eigen::Vector3d linearized_gravity = gravity_norm * gravity_direction;

        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6 * (frame_count - 1), state_size);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(6 * (frame_count - 1));

        for (int i = 0; i < frame_count - 1; ++i) {
            const int j = i + 1;
            const double dt = dts[i];
            if (!std::isfinite(dt) || dt <= 0.0) {
                spdlog::error(
                    "GravityRefinement input failed: iteration={}, interval={}, dt={}", iter, i,
                    dt);
                return false;
            }
            const double half_dt_squared = 0.5 * dt * dt;

            const Eigen::Matrix3d rotation_i = ric * rotations[i].transpose() * ric.transpose();
            const Eigen::Matrix3d camera_to_body_i = ric * rotations[i].transpose();
            const Eigen::Matrix<double, 3, 2> projected_gravity_jacobian =
                rotation_i * gravity_jacobian;

            const int velocity_i_column = i * 3;
            const int velocity_j_column = j * 3;

            Eigen::Matrix<double, 6, 9> interval_jacobian;
            interval_jacobian.setZero();

            interval_jacobian.block<3, 3>(0, 0) = -dt * rotation_i;
            interval_jacobian.block<3, 2>(0, 6) = half_dt_squared * projected_gravity_jacobian;
            interval_jacobian.block<3, 1>(0, 8) =
                camera_to_body_i * (positions[j] - positions[i]) / 100.0;

            interval_jacobian.block<3, 3>(3, 0) = -rotation_i;
            interval_jacobian.block<3, 3>(3, 3) = rotation_i;
            interval_jacobian.block<3, 2>(3, 6) = dt * projected_gravity_jacobian;

            Eigen::Matrix<double, 6, 1> interval_rhs;
            interval_rhs.block<3, 1>(0, 0) =
                delta_positions[i] + camera_to_body_i * rotations[j] * ric.transpose() * tic - tic -
                half_dt_squared * rotation_i * linearized_gravity;
            interval_rhs.block<3, 1>(3, 0) =
                delta_velocities[i] - dt * rotation_i * linearized_gravity;

            const int row = 6 * i;
            jacobian.block<6, 3>(row, velocity_i_column) = interval_jacobian.leftCols<3>();
            jacobian.block<6, 3>(row, velocity_j_column) = interval_jacobian.middleCols<3>(3);
            jacobian.block<6, 2>(row, gravity_column) = interval_jacobian.middleCols<2>(6);
            jacobian.block<6, 1>(row, scale_column) = interval_jacobian.rightCols<1>();
            rhs.segment<6>(row) = interval_rhs;
        }

        Eigen::VectorXd solution;
        if (!solveObservableSystem(jacobian, rhs, "GravityRefinement", solution)) {
            return false;
        }

        const Eigen::Vector2d tangent_update = solution.segment<2>(gravity_column);
        gravity_direction =
            (gravity_direction + tangent_basis.transpose() * tangent_update).normalized();
        if (!gravity_direction.allFinite()) {
            spdlog::warn(
                "GravityRefinement validity failed: iteration={}, non-finite gravity direction",
                iter);
            return false;
        }

        for (int i = 0; i < frame_count; ++i) {
            velocities[i] = solution.segment<3>(i * 3);
        }
        scale = solution(scale_column) / 100.0;
    }

    gravity = gravity_norm * gravity_direction;
    if (!gravity.allFinite() || !std::isfinite(scale)) {
        spdlog::warn(
            "GravityRefinement validity failed: gravity_norm={}, scale={}", gravity.norm(), scale);
        return false;
    }
    return true;
}

Eigen::Vector3d solveGyroBiasCorrection(
    const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Matrix3d>& rotation_bias_jacobians,
    const std::vector<Eigen::Matrix3d>& delta_rotations, const Eigen::Matrix3d& ric) {
    const size_t interval_count = rotation_bias_jacobians.size();
    if (interval_count == 0 || rotations.size() != interval_count + 1 ||
        delta_rotations.size() != interval_count) {
        spdlog::error(
            "GyroBias input failed: R={}, dq_dbg={}, delta_q={}, expected_R={}", rotations.size(),
            rotation_bias_jacobians.size(), delta_rotations.size(), interval_count + 1);
        return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }

    Eigen::MatrixXd jacobian(3 * interval_count, 3);
    Eigen::VectorXd residual(3 * interval_count);

    for (size_t i = 0; i < interval_count; ++i) {
        Eigen::Matrix3d visual_rotation =
            ric * rotations[i].transpose() * rotations[i + 1] * ric.transpose();
        Eigen::Quaterniond q_ij(visual_rotation);
        q_ij.normalize();
        visual_rotation = q_ij.toRotationMatrix();

        Eigen::Quaterniond q_delta(delta_rotations[i]);
        q_delta.normalize();
        const Eigen::Matrix3d rotation_error =
            q_delta.toRotationMatrix().transpose() * visual_rotation;
        jacobian.block<3, 3>(3 * i, 0) = rotation_bias_jacobians[i];
        residual.segment<3>(3 * i) = Sophus::SO3d(rotation_error).log();
    }

    Eigen::VectorXd correction;
    if (!solveObservableSystem(jacobian, residual, "GyroBiasCorrection", correction)) {
        return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    spdlog::info(
        "Gyro bias correction: ({:.6f}, {:.6f}, {:.6f})", correction.x(), correction.y(),
        correction.z());
    return correction;
}

}  // namespace tassel_core
