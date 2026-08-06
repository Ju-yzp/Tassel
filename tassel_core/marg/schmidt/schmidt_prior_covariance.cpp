#include "marg/schmidt/schmidt_prior_covariance.h"

#include <Eigen/Eigenvalues>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "marg/gauge_fixed_covariance.h"
#include "marg/schmidt/schmidt_update.h"
#include "marg/state_layout.h"

namespace tassel_core {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRotationTolerance = 1e-8;

PriorStateLayout requireMatchingLayout(
    const SchmidtPriorCovariance& prior_covariance, const MargLinData* prior = nullptr) {
    const PriorStateLayout layout(
        prior_covariance.state_count, static_cast<int>(prior_covariance.covariance.cols()));
    if (prior_covariance.covariance.rows() != layout.columnCount() ||
        !prior_covariance.covariance.allFinite() ||
        !prior_covariance.covariance.isApprox(prior_covariance.covariance.transpose(), 1e-10)) {
        throw std::invalid_argument("Schmidt prior covariance layout is inconsistent");
    }
    if (prior && (prior->linearization_poses.size() != static_cast<size_t>(layout.stateCount()) ||
                  prior->H.cols() != layout.columnCount())) {
        throw std::invalid_argument("Schmidt covariance does not match its marginalization prior");
    }
    return layout;
}

void requireRotation(const Eigen::Matrix3d& rotation) {
    if (!rotation.allFinite() ||
        !(rotation.transpose() * rotation)
             .isApprox(Eigen::Matrix3d::Identity(), kRotationTolerance) ||
        std::abs(rotation.determinant() - 1.0) > kRotationTolerance) {
        throw std::invalid_argument("Covariance gauge rotation is not in SO(3)");
    }
}

}  // namespace

SchmidtPriorCovariance buildGaugeFixedPriorCovariance(
    const MargLinData& prior, const Eigen::Matrix3d& anchor_rotation) {
    const int state_count = static_cast<int>(prior.linearization_poses.size());
    const PriorStateLayout layout(state_count, static_cast<int>(prior.H.cols()));
    if (layout.kind() == PriorLayoutKind::PoseOnly) {
        throw std::invalid_argument("Pose-only prior cannot propagate an IMU covariance state");
    }
    const GaugeFixedBasis gauge =
        buildPositionYawGaugeBasis(layout.columnCount(), layout.poseColumn(0), anchor_rotation);
    SchmidtPriorCovariance result;
    result.covariance = covarianceInFixedGauge(prior.H, gauge).lifted;
    result.state_count = state_count;
    return result;
}

bool priorHasFiniteGaugeFixedCovariance(
    const MargLinData& prior, const Eigen::Matrix3d& anchor_rotation) {
    const int state_count = static_cast<int>(prior.linearization_poses.size());
    const PriorStateLayout layout(state_count, static_cast<int>(prior.H.cols()));
    if (layout.kind() == PriorLayoutKind::PoseOnly || !prior.H.allFinite()) {
        throw std::invalid_argument("Prior cannot define a finite propagated IMU covariance");
    }
    const GaugeFixedBasis gauge =
        buildPositionYawGaugeBasis(layout.columnCount(), layout.poseColumn(0), anchor_rotation);
    const Eigen::MatrixXd reduced_information =
        gauge.tangent.transpose() * prior.H.transpose() * prior.H * gauge.tangent;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (reduced_information + reduced_information.transpose()));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Gauge-fixed prior rank decomposition failed");
    }
    const double scale = std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
    return solver.eigenvalues().minCoeff() > 1e-12 * scale;
}

void recenterPriorCovariance(
    SchmidtPriorCovariance& prior_covariance, const MargLinData& prior,
    const std::vector<std::array<double, 6>>& new_poses) {
    const PriorStateLayout layout = requireMatchingLayout(prior_covariance, &prior);
    if (new_poses.size() != static_cast<size_t>(layout.stateCount())) {
        throw std::invalid_argument("Covariance recenter state count is inconsistent");
    }

    // MargHelper 使用 delta_old = delta_center + T * delta_new 更新 H；协方差需用 T^{-1}。
    Eigen::MatrixXd inverse_map =
        Eigen::MatrixXd::Identity(layout.columnCount(), layout.columnCount());
    for (int frame_index = 0; frame_index < layout.stateCount(); ++frame_index) {
        const Eigen::Vector3d old_phi(
            prior.linearization_poses[frame_index][3], prior.linearization_poses[frame_index][4],
            prior.linearization_poses[frame_index][5]);
        const Eigen::Vector3d new_phi(
            new_poses[frame_index][3], new_poses[frame_index][4], new_poses[frame_index][5]);
        const Eigen::Vector3d rotation_delta =
            (Sophus::SO3d::exp(old_phi).inverse() * Sophus::SO3d::exp(new_phi)).log();
        if (!old_phi.allFinite() || !new_phi.allFinite() || !rotation_delta.allFinite() ||
            rotation_delta.norm() >= kPi - 1e-6) {
            throw std::invalid_argument("Covariance recenter rotation is invalid");
        }
        const Eigen::Matrix3d tangent_map = Sophus::SO3d::leftJacobianInverse(-rotation_delta);
        inverse_map.block<3, 3>(
            layout.poseColumn(frame_index) + 3, layout.poseColumn(frame_index) + 3) =
            tangent_map.inverse();
    }
    prior_covariance.covariance =
        inverse_map * prior_covariance.covariance * inverse_map.transpose();
    prior_covariance.covariance =
        0.5 * (prior_covariance.covariance + prior_covariance.covariance.transpose()).eval();
    requireMatchingLayout(prior_covariance);
}

void transformPriorCovarianceGauge(
    SchmidtPriorCovariance& prior_covariance, const Eigen::Matrix3d& rotation) {
    const PriorStateLayout layout = requireMatchingLayout(prior_covariance);
    requireRotation(rotation);
    // 右扰动姿态及体坐标 ba/bg 不变；仅世界坐标位置和速度随 gauge 左乘旋转。
    Eigen::MatrixXd gauge_map =
        Eigen::MatrixXd::Identity(layout.columnCount(), layout.columnCount());
    for (int frame_index = 0; frame_index < layout.stateCount(); ++frame_index) {
        gauge_map.block<3, 3>(layout.poseColumn(frame_index), layout.poseColumn(frame_index)) =
            rotation;
        if (layout.hasSpeedBias(frame_index)) {
            gauge_map.block<3, 3>(
                layout.speedBiasColumn(frame_index), layout.speedBiasColumn(frame_index)) =
                rotation;
        }
    }
    prior_covariance.covariance = gauge_map * prior_covariance.covariance * gauge_map.transpose();
    prior_covariance.covariance =
        0.5 * (prior_covariance.covariance + prior_covariance.covariance.transpose()).eval();
    requireMatchingLayout(prior_covariance);
}

SchmidtPriorCovariance propagatePriorCovarianceWithImu(
    const SchmidtPriorCovariance& prior_covariance, int parent_frame_index,
    const ImuCovariancePropagation& propagation) {
    const PriorStateLayout old_layout(
        prior_covariance.state_count, static_cast<int>(prior_covariance.covariance.cols()));
    if (prior_covariance.covariance.rows() != old_layout.columnCount() ||
        !old_layout.hasSpeedBias(parent_frame_index)) {
        throw std::invalid_argument("IMU parent is not a full state in the prior covariance");
    }
    const int parent_column = old_layout.poseColumn(parent_frame_index);
    const int child_column =
        old_layout.hasDelay() ? old_layout.delayColumn() : old_layout.columnCount();
    SchmidtPriorCovariance result;
    result.covariance = insertImuPropagatedCovariance(
        prior_covariance.covariance, parent_column, child_column, propagation);
    result.state_count = prior_covariance.state_count + 1;
    const PriorStateLayout new_layout(
        result.state_count, static_cast<int>(result.covariance.cols()));
    if (new_layout.kind() != old_layout.kind() || new_layout.hasDelay() != old_layout.hasDelay()) {
        throw std::logic_error("IMU propagation changed the prior layout kind");
    }
    return result;
}

SchmidtPriorCovariance retainMarginalizedPriorCovariance(
    const SchmidtPriorCovariance& current_covariance, RetainedHostAction action) {
    const PriorStateLayout current_layout = requireMatchingLayout(current_covariance);
    if (current_layout.kind() != PriorLayoutKind::PoseOnlyHost || !current_layout.hasDelay() ||
        current_layout.stateCount() < 2) {
        throw std::invalid_argument("Covariance retention requires a pose-only-host window");
    }

    std::vector<int> retained_columns;
    retained_columns.reserve(
        static_cast<size_t>(current_layout.columnCount() - PriorStateLayout::kFullStateSize));
    const auto append = [&retained_columns](int first, int count) {
        for (int d = 0; d < count; ++d) {
            retained_columns.push_back(first + d);
        }
    };
    const int retained_host = action == RetainedHostAction::MarginalizeOldestFrame ? 0 : 1;
    append(current_layout.poseColumn(retained_host), PriorStateLayout::kPoseSize);
    for (int frame_index = 2; frame_index < current_layout.stateCount(); ++frame_index) {
        append(current_layout.poseColumn(frame_index), PriorStateLayout::kFullStateSize);
    }
    append(current_layout.delayColumn(), 1);

    SchmidtPriorCovariance result;
    result.state_count = current_layout.stateCount() - 1;
    const PriorStateLayout result_layout(
        result.state_count, static_cast<int>(retained_columns.size()));
    if (result_layout.kind() != PriorLayoutKind::PoseOnlyHost || !result_layout.hasDelay()) {
        throw std::logic_error("Retained Schmidt covariance layout is inconsistent");
    }
    result.covariance.resize(retained_columns.size(), retained_columns.size());
    for (int row = 0; row < result.covariance.rows(); ++row) {
        for (int column = 0; column < result.covariance.cols(); ++column) {
            result.covariance(row, column) = current_covariance.covariance(
                retained_columns[static_cast<size_t>(row)],
                retained_columns[static_cast<size_t>(column)]);
        }
    }
    return result;
}

std::vector<std::array<int, 3>> priorBaColumns(const SchmidtPriorCovariance& prior_covariance) {
    const PriorStateLayout layout(
        prior_covariance.state_count, static_cast<int>(prior_covariance.covariance.cols()));
    std::vector<std::array<int, 3>> columns;
    for (int frame_index = 0; frame_index < layout.stateCount(); ++frame_index) {
        if (layout.hasSpeedBias(frame_index)) {
            columns.push_back(layout.baColumns(frame_index));
        }
    }
    return columns;
}

SchmidtPriorCovariance propagateAndUpdateSchmidtPrior(
    const SchmidtPriorCovariance& prior_covariance, const ImuCovariancePropagation& propagation,
    const Eigen::MatrixXd& visual_jacobian_in_window, const Eigen::VectorXd& visual_residual,
    int window_state_count, RetainedHostAction action) {
    const PriorStateLayout old_layout(
        prior_covariance.state_count, static_cast<int>(prior_covariance.covariance.cols()));
    if (old_layout.kind() != PriorLayoutKind::PoseOnlyHost || !old_layout.hasDelay() ||
        action == RetainedHostAction::InitializeRetainedSlot ||
        old_layout.stateCount() + 1 != window_state_count ||
        visual_jacobian_in_window.rows() != visual_residual.size() ||
        visual_jacobian_in_window.cols() !=
            window_state_count * PriorStateLayout::kFullStateSize + 1) {
        throw std::invalid_argument("Schmidt prior update layout is inconsistent");
    }
    SchmidtPriorCovariance propagated =
        propagatePriorCovarianceWithImu(prior_covariance, old_layout.stateCount() - 1, propagation);
    const PriorStateLayout current_layout(
        propagated.state_count, static_cast<int>(propagated.covariance.cols()));
    const std::vector<int> compact_to_window =
        current_layout.compactToWindowColumns(window_state_count);
    Eigen::MatrixXd compact_visual_jacobian(
        visual_jacobian_in_window.rows(), current_layout.columnCount());
    for (int compact_column = 0; compact_column < current_layout.columnCount(); ++compact_column) {
        compact_visual_jacobian.col(compact_column) =
            visual_jacobian_in_window.col(compact_to_window[static_cast<size_t>(compact_column)]);
    }

    if (visual_residual.size() > 0) {
        std::vector<int> schmidt_indices;
        for (const std::array<int, 3>& ba_columns : priorBaColumns(propagated)) {
            schmidt_indices.insert(schmidt_indices.end(), ba_columns.begin(), ba_columns.end());
        }
        const SchmidtJointUpdateResult update = applySchmidtUpdateToJointCovariance(
            propagated.covariance, compact_visual_jacobian,
            Eigen::MatrixXd::Identity(visual_residual.size(), visual_residual.size()),
            -visual_residual, schmidt_indices);
        propagated.covariance = update.covariance;
    }

    return retainMarginalizedPriorCovariance(propagated, action);
}

}  // namespace tassel_core
