#ifndef TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_WINDOW_UPDATE_H_
#define TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_WINDOW_UPDATE_H_

#include <Eigen/Core>

#include <ceres/loss_function.h>
#include <sophus/so3.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "factor/integrator_base.h"
#include "frond_end/feature.h"
#include "marg/imu_block.h"
#include "marg/marginalization_sqrt.h"
#include "marg/schmidt/schmidt_prior_covariance.h"
#include "marg/state_layout.h"
#include "state/state.h"

namespace tassel_core {

template <typename Integrator>
ImuCovariancePropagation linearizeImuCovariancePropagation(
    IntegratorBase<Integrator>* preintegrator, const FrameState& parent, const FrameState& child) {
    if (!preintegrator || preintegrator->buffer.size() < 2) {
        throw std::logic_error("Schmidt IMU interval has fewer than two measurements");
    }
    IMUBlock<Integrator> imu_block;
    imu_block.allocate(preintegrator);
    imu_block.linearize(
        parent.V, child.V, parent.P, child.P, Sophus::SO3d(parent.R).log(),
        Sophus::SO3d(child.R).log(), parent.Ba, child.Ba, parent.Bg, child.Bg);
    return buildImuCovariancePropagation(
        imu_block.jacobian().template leftCols<15>(), imu_block.jacobian().template rightCols<15>(),
        imu_block.residual());
}

template <typename Integrator>
std::optional<SchmidtPriorCovariance> tryBuildGaugeFixedWindowPosteriorCovariance(
    const std::vector<Feature*>& features, const std::shared_ptr<State>& state,
    std::vector<Integrator>& preintegrators, int first_imu_factor_index, const MargLinData* prior,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, double visual_huber_delta) {
    if (!state || state->max_frame_count < 2 || first_imu_factor_index < 0 ||
        first_imu_factor_index >= state->max_frame_count - 1) {
        throw std::invalid_argument("Schmidt window posterior layout is invalid");
    }
    std::vector<IntegratorBase<Integrator>*> imu_factors;
    imu_factors.reserve(static_cast<size_t>(state->max_frame_count - 1 - first_imu_factor_index));
    for (int index = first_imu_factor_index; index < state->max_frame_count - 1; ++index) {
        if (index >= static_cast<int>(preintegrators.size()) ||
            preintegrators[index].buffer.size() < 2) {
            throw std::logic_error("Schmidt window IMU interval has fewer than two measurements");
        }
        imu_factors.push_back(&preintegrators[index]);
    }

    auto linearizer = MarginalizationSqrt<Integrator>(
        features, -1, std::make_unique<ceres::HuberLoss>(visual_huber_delta), state, imu_factors,
        ric, tic, prior, first_imu_factor_index);
    linearizer.allocate();
    linearizer.linearize();
    linearizer.marginalizeLandmarks();
    Eigen::MatrixXd full_jacobian;
    Eigen::VectorXd full_residual;
    linearizer.buildReducedSystem(full_jacobian, full_residual);

    const int window_state_count = state->max_frame_count;
    const int compact_columns = PriorStateLayout::kPoseSize +
                                (window_state_count - 1) * PriorStateLayout::kFullStateSize + 1;
    const PriorStateLayout layout(window_state_count, compact_columns);
    const std::vector<int> compact_to_window = layout.compactToWindowColumns(window_state_count);
    MargLinData posterior;
    posterior.H.resize(full_jacobian.rows(), compact_columns);
    for (int compact_column = 0; compact_column < compact_columns; ++compact_column) {
        posterior.H.col(compact_column) =
            full_jacobian.col(compact_to_window[static_cast<size_t>(compact_column)]);
    }
    posterior.b = std::move(full_residual);
    posterior.linearization_poses.resize(window_state_count);
    posterior.linearization_speed_bias.resize(window_state_count);
    for (int frame_index = 0; frame_index < window_state_count; ++frame_index) {
        posterior.linearization_poses[frame_index] = state->frames[frame_index].pose;
        posterior.linearization_speed_bias[frame_index] = state->frames[frame_index].speed_bias;
    }
    posterior.linearization_delay_time = state->param_delay_time;

    // frame0 是仅保留位姿的宿主；其世界位置和 yaw 构成显式固定 gauge。
    const Eigen::Matrix3d anchor_rotation = state->frames[0].R;
    if (!priorHasFiniteGaugeFixedCovariance(posterior, anchor_rotation)) {
        return std::nullopt;
    }
    return buildGaugeFixedPriorCovariance(posterior, anchor_rotation);
}

template <typename Integrator>
SchmidtPriorCovariance propagateAndUpdateSchmidtWindow(
    const SchmidtPriorCovariance& prior_covariance,
    IntegratorBase<Integrator>* latest_preintegrator, const FrameState& parent,
    const FrameState& child, const Eigen::MatrixXd& visual_jacobian,
    const Eigen::VectorXd& visual_residual, int window_state_count, RetainedHostAction action) {
    const ImuCovariancePropagation propagation =
        linearizeImuCovariancePropagation(latest_preintegrator, parent, child);
    return propagateAndUpdateSchmidtPrior(
        prior_covariance, propagation, visual_jacobian, visual_residual, window_state_count,
        action);
}

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_WINDOW_UPDATE_H_
