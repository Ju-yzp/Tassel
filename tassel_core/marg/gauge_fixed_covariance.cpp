#include "gauge_fixed_covariance.h"

#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <stdexcept>

#include "marg/schmidt/schmidt_update.h"

namespace tassel_core {

GaugeFixedBasis buildPositionYawGaugeBasis(
    int state_size, int anchor_pose_column, const Eigen::Matrix3d& anchor_rotation) {
    constexpr double rotation_tolerance = 1e-8;
    if (state_size < 6 || anchor_pose_column < 0 || anchor_pose_column + 6 > state_size) {
        throw std::invalid_argument("Gauge anchor pose is outside the state layout");
    }
    if (!anchor_rotation.allFinite() ||
        !(anchor_rotation.transpose() * anchor_rotation)
             .isApprox(Eigen::Matrix3d::Identity(), rotation_tolerance) ||
        std::abs(anchor_rotation.determinant() - 1.0) > rotation_tolerance) {
        throw std::invalid_argument("Gauge anchor rotation is not in SO(3)");
    }

    const Eigen::Vector3d yaw_direction = anchor_rotation.transpose() * Eigen::Vector3d::UnitZ();
    std::array<Eigen::Vector3d, 3> axes{
        Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()};
    int least_aligned_axis = 0;
    for (int i = 1; i < 3; ++i) {
        if (std::abs(yaw_direction.dot(axes[static_cast<size_t>(i)])) <
            std::abs(yaw_direction.dot(axes[static_cast<size_t>(least_aligned_axis)]))) {
            least_aligned_axis = i;
        }
    }
    Eigen::Vector3d rotation_basis_x =
        axes[static_cast<size_t>(least_aligned_axis)] -
        yaw_direction * yaw_direction.dot(axes[static_cast<size_t>(least_aligned_axis)]);
    rotation_basis_x.normalize();
    const Eigen::Vector3d rotation_basis_y = yaw_direction.cross(rotation_basis_x).normalized();

    GaugeFixedBasis basis;
    basis.tangent = Eigen::MatrixXd::Zero(state_size, state_size - 4);
    basis.full_to_reduced.assign(static_cast<size_t>(state_size), -1);
    const int position_column = anchor_pose_column;
    const int rotation_column = anchor_pose_column + 3;
    int reduced_column = 0;
    for (int full_column = 0; full_column < state_size; ++full_column) {
        if (full_column >= position_column && full_column < position_column + 3) {
            continue;
        }
        if (full_column == rotation_column) {
            basis.tangent.block<3, 1>(rotation_column, reduced_column++) = rotation_basis_x;
            basis.tangent.block<3, 1>(rotation_column, reduced_column++) = rotation_basis_y;
            full_column += 2;
            continue;
        }
        basis.tangent(full_column, reduced_column) = 1.0;
        basis.full_to_reduced[static_cast<size_t>(full_column)] = reduced_column;
        ++reduced_column;
    }
    if (reduced_column != state_size - 4 ||
        !(basis.tangent.transpose() * basis.tangent)
             .isApprox(Eigen::MatrixXd::Identity(state_size - 4, state_size - 4), 1e-12)) {
        throw std::logic_error("Gauge-fixed tangent basis is inconsistent");
    }
    return basis;
}

GaugeFixedCovariance covarianceInFixedGauge(
    const Eigen::MatrixXd& sqrt_information, const GaugeFixedBasis& basis) {
    if (sqrt_information.cols() != basis.tangent.rows() ||
        basis.full_to_reduced.size() != static_cast<size_t>(basis.tangent.rows()) ||
        !sqrt_information.allFinite() || !basis.tangent.allFinite()) {
        throw std::invalid_argument("Gauge-fixed covariance dimensions are inconsistent");
    }
    const Eigen::MatrixXd reduced_sqrt_information = sqrt_information * basis.tangent;
    GaugeFixedCovariance covariance;
    covariance.reduced = covarianceFromFullRankSqrtInformation(reduced_sqrt_information);
    covariance.lifted = basis.tangent * covariance.reduced * basis.tangent.transpose();
    return covariance;
}

}  // namespace tassel_core
