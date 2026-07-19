// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#include "estimator.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <vector>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/reprojection_factor.h"
#include "marg/marg_helper.h"
#include "marg/marginalization_sqrt.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

#include "initial/initial_alignment.h"
#include "initial/initial_sfm.h"

namespace tassel_core {

namespace {

template <typename Integrator>
std::vector<Integrator> makePreintegrators(
    size_t count, const Eigen::Matrix<double, 18, 18>& noise) {
    return std::vector<Integrator>(
        count, Integrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise));
}

template <typename Range>
bool allFinite(const Range& values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

Eigen::Vector3d interpolateImuVector(
    const std::vector<tassel_utils::IMUMeasurement>& measurements, double timestamp, bool gyro) {
    if (measurements.empty()) {
        return Eigen::Vector3d::Zero();
    }
    const size_t count = measurements.size();
    if (count < 3) {
        const auto& sample = measurements.back();
        return gyro ? sample.gyro : sample.acc;
    }

    size_t right = 0;
    while (right < count && measurements[right].timestamp < timestamp) {
        ++right;
    }
    size_t first = right > 1 ? right - 2 : 0;
    if (first + 3 > count) {
        first = count - 3;
    }

    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < 3; ++i) {
        const double ti = measurements[first + i].timestamp;
        double weight = 1.0;
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                continue;
            }
            const double tj = measurements[first + j].timestamp;
            const double denominator = ti - tj;
            if (std::abs(denominator) < 1e-12) {
                return gyro ? measurements.back().gyro : measurements.back().acc;
            }
            weight *= (timestamp - tj) / denominator;
        }
        result += weight * (gyro ? measurements[first + i].gyro : measurements[first + i].acc);
    }
    return result;
}

}  // namespace

Estimator::Estimator(
    const tassel_tools::Parameters& params, std::shared_ptr<State> state,
    std::shared_ptr<FeatureManager> fm)
    : params_(params), state_(std::move(state)), feature_manager_(std::move(fm)) {
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    cv::setNumThreads(params_.num_threads);
    noise_ = initNoise();
    state_->visual_sqrt_info = Eigen::Matrix2d::Identity() * params_.visual_factor_weight;
    reset();
}

void Estimator::reset() {
    initialized_ = false;
    last_ts_ = -1;
    last_imu_acc_ = Eigen::Vector3d::Zero();
    last_imu_gyro_ = Eigen::Vector3d::Zero();
    switch (params_.integrator_type) {
        case tassel_utils::IntegratorType::kMidPoint:
            preintegrators_ =
                makePreintegrators<MidPointIntegrator>(state_->max_frame_count - 1, noise_);
            break;
        case tassel_utils::IntegratorType::kEuler:
            preintegrators_ =
                makePreintegrators<EulerIntegrator>(state_->max_frame_count - 1, noise_);
            break;
    }
    Rs_.resize(state_->max_frame_count, Eigen::Matrix3d::Identity());
    Ps_.resize(state_->max_frame_count, Eigen::Vector3d::Zero());
    Vs_.resize(state_->max_frame_count, Eigen::Vector3d::Zero());
    marginalization_prior_.reset();
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    state_->reset();
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay) {
    int& frame_count = state_->newest_slot;
    state_->frames[frame_count].timestamp_ns = frame_id;
    const double ts = tassel_utils::frameIdToSeconds(frame_id);
    state_->frames[frame_count].sync_delay = sync_delay;
    if (last_ts_ < 0 && !imu_measurements.empty()) {
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc - params_.acc_bias;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    const bool initialization_keyframe =
        feature_manager_->checkParallax(frame_count, feature_frame);
    const auto& input_stats = feature_manager_->lastInputStats();
    const bool weak_keyframe_connection =
        !feature_manager_->hasLatestKeyframe() ||
        input_stats.current_keyframe_connection_ratio <= (1.0 - params_.keyframe_new_feature_ratio);
    const bool is_keyframe = initialization_keyframe || weak_keyframe_connection;
    if (is_keyframe) {
        feature_manager_->acceptKeyframe(feature_frame);
    }
    feature_manager_->logInputStats(initialized_ ? is_keyframe : initialization_keyframe);

    predictFrameState(frame_count, imu_measurements);

    const double imu_query_timestamp = ts + sync_delay;
    state_->frames[frame_count].acc =
        interpolateImuVector(imu_measurements, imu_query_timestamp, false);
    state_->frames[frame_count].gyro =
        interpolateImuVector(imu_measurements, imu_query_timestamp, true);
    state_->frames[frame_count].is_keyframe = is_keyframe;
    if (!initialized_) {
        // 初始化期间将每个相机帧保留为独立的 VIO 状态。
        // SFM 前端从有限窗口中选择条件良好的种子帧对，关键帧分类不能将
        // 多帧 IMU 样本合并为过长的预积分区间。
        feature_manager_->triangulate(*state_, params_.ric, params_.tic);
        if (frame_count < state_->max_frame_count - 1) {
            ++frame_count;
            state_->copyFrameSlot(frame_count - 1, frame_count);
            state_->frames[frame_count].timestamp_ns = tassel_utils::kInvalidFrameId;
            int next_idx = frame_count - 1;
            visitPreintegrators([&](auto& preintegrators) {
                if (next_idx < static_cast<int>(preintegrators.size())) {
                    preintegrators[next_idx].reset(
                        state_->frames[frame_count - 1].Ba, state_->frames[frame_count - 1].Bg,
                        noise_);
                    preintegrators[next_idx].clearFrameInterval();
                }
            });
            return;
        }

        if (!tryInitialize()) {
            spdlog::info("VI initialization not ready; sliding initialization window");
            feature_manager_->removeOldestFrameObservations(*state_, params_.ric, params_.tic);
            slideInitializationWindow();
            return;
        }
        feature_manager_->resetLandmarkDepths();
    }

    feature_manager_->triangulate(*state_, params_.ric, params_.tic);
    optimize(ts);
    const Sophus::SE3d optimized_pose(state_->frames[frame_count].R, state_->frames[frame_count].P);
    if (pose_callback_) {
        pose_callback_(ts, optimized_pose);
    }
    if (realtime_pose_callback_) {
        realtime_pose_callback_(ts, optimized_pose);
    }
    feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
    RetainedHostAction host_action = RetainedHostAction::kKeep;
    if (!state_->has_retained_host) {
        host_action = RetainedHostAction::kCreate;
    } else if (state_->frames[1].is_keyframe) {
        host_action = RetainedHostAction::kReplace;
    }
    const MarginalizationLayout marginalization_layout(host_action);
    updateMarginalizationPrior(marginalization_layout);
    if (marginalization_layout.replacesRetainedHost()) {
        feature_manager_->replaceRetainedHost(0, 1, *state_, params_.ric, params_.tic);
    } else {
        feature_manager_->removeFrameObservations(1, *state_, params_.ric, params_.tic);
    }
    shiftWindowAfterMarginalization(marginalization_layout);

    if (cloud_callback_ && initialized_) {
        auto pts = feature_manager_->getPointCloud(*state_, params_.ric, params_.tic);
        cloud_callback_(ts, pts);
    }
}

void Estimator::predictFrameState(
    int frame_slot, const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    if (frame_slot == 0) {
        return;
    }

    TASSEL_ASSERT(frame_slot > 0 && frame_slot <= state_->newest_slot);
    const FrameState& previous_frame = state_->frames[frame_slot - 1];
    FrameState& predicted_frame = state_->frames[frame_slot];
    TASSEL_ASSERT(previous_frame.timestamp_ns != tassel_utils::kInvalidFrameId);
    TASSEL_ASSERT(predicted_frame.timestamp_ns > previous_frame.timestamp_ns);

    Eigen::Matrix3d rotation = predicted_frame.R;
    Eigen::Vector3d position = predicted_frame.P;
    Eigen::Vector3d velocity = predicted_frame.V;
    const Eigen::Vector3d acc_bias = predicted_frame.Ba;
    const Eigen::Vector3d gyro_bias = predicted_frame.Bg;

    visitPreintegrators([&](auto& preintegrators) {
        auto& preintegrator = preintegrators[frame_slot - 1];
        preintegrator.setFrameInterval(previous_frame.timestamp_ns, predicted_frame.timestamp_ns);

        for (const auto& imu : imu_measurements) {
            tassel_utils::IMUMeasurement calibrated_imu = imu;
            calibrated_imu.acc -= params_.acc_bias;
            if (!preintegrator.propagate(calibrated_imu)) {
                throw std::runtime_error("Invalid or non-monotonic IMU measurement");
            }

            const double dt = calibrated_imu.timestamp - last_ts_;
            const Eigen::Vector3d previous_acceleration =
                rotation * (last_imu_acc_ - acc_bias) - tassel_utils::G;
            const Eigen::Vector3d angular_velocity =
                0.5 * (last_imu_gyro_ + calibrated_imu.gyro) - gyro_bias;
            rotation *= Sophus::SO3d::exp(angular_velocity * dt).matrix();
            const Eigen::Vector3d current_acceleration =
                rotation * (calibrated_imu.acc - acc_bias) - tassel_utils::G;
            const Eigen::Vector3d average_acceleration =
                0.5 * (previous_acceleration + current_acceleration);
            position += velocity * dt + 0.5 * average_acceleration * dt * dt;
            velocity += average_acceleration * dt;

            last_ts_ = calibrated_imu.timestamp;
            last_imu_gyro_ = calibrated_imu.gyro;
            last_imu_acc_ = calibrated_imu.acc;
        }
    });

    predicted_frame.R = Eigen::Quaterniond(rotation).normalized().toRotationMatrix();
    predicted_frame.P = position;
    predicted_frame.V = velocity;
}

void Estimator::optimize(double timestamp) {
    const int latest_id = state_->newest_slot;
    state_->stateToParams();
    auto features = feature_manager_->collectLandmarks();

    ceres::Problem problem;
    std::vector<ceres::ResidualBlockId> prior_residuals;
    std::vector<ceres::ResidualBlockId> visual_residuals;
    std::vector<ceres::ResidualBlockId> imu_residuals;
    std::vector<int> visual_factors_per_frame(state_->newest_slot + 1, 0);

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->frames[i].pose.data(), 6, se3_manifold);
        problem.AddParameterBlock(state_->frames[i].speed_bias.data(), 9);
    }
    if (state_->has_retained_host) {
        problem.SetParameterBlockConstant(state_->frames[0].speed_bias.data());
    }
    if (!marginalization_prior_) {
        problem.SetParameterBlockConstant(state_->frames[0].pose.data());
    }

    problem.AddParameterBlock(&state_->param_delay_time, 1);
    int delay_observable_frames = 0;
    for (int i = 0; i <= state_->newest_slot; ++i) {
        const bool angular_motion_sufficient =
            (state_->frames[i].gyro - state_->frames[i].Bg).norm() >
            params_.delay_obs_gyro_threshold;
        const bool linear_motion_sufficient =
            state_->frames[i].V.norm() > params_.delay_obs_speed_threshold;
        if (angular_motion_sufficient || linear_motion_sufficient) {
            ++delay_observable_frames;
        }
    }

    if (delay_observable_frames < params_.delay_obs_min_frames) {
        problem.SetParameterBlockConstant(&state_->param_delay_time);
    }

    if (marginalization_prior_) {
        auto* prior_cost = new MarginalizationPriorFactor(*marginalization_prior_);
        std::vector<double*> prior_blocks;
        int num_kept = static_cast<int>(marginalization_prior_->linearization_poses.size());
        for (int i = 0; i < num_kept; ++i) {
            prior_blocks.push_back(state_->frames[i].pose.data());
            prior_blocks.push_back(state_->frames[i].speed_bias.data());
        }
        prior_blocks.push_back(&state_->param_delay_time);
        prior_residuals.push_back(problem.AddResidualBlock(prior_cost, nullptr, prior_blocks));
    }

    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;
    ceres::LossFunction* loss = new ceres::HuberLoss(visual_huber_delta);
    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < params_.max_depth) ? (1.0 / d) : 1.0;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
        // problem.SetParameterLowerBound(&inv_depth_params[k], 0, 1.0 / params_.max_depth);
        // problem.SetParameterUpperBound(&inv_depth_params[k], 0, 1.0 / params_.min_depth);
    }

    Eigen::Matrix2d sqrt_info = state_->visual_sqrt_info;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        const int host_id = f->start_slot;
        if (host_id < 0 || host_id > latest_id) {
            continue;
        }
        for (size_t obs_idx = 0; obs_idx < f->observations.size(); ++obs_idx) {
            const int observation_slot = f->observationSlot(obs_idx);
            if (observation_slot <= latest_id) {
                ++visual_factors_per_frame[observation_slot];
            }
        }
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            const int target_id = f->observationSlot(obs_idx);
            if (target_id > latest_id) {
                continue;
            }
            Eigen::Vector2d pt_j(f->observations[obs_idx].pt.x, f->observations[obs_idx].pt.y);
            auto* cost = new ReprojectionFactor(
                f->observations[0].uv, pt_j, params_.ric, params_.tic, state_->frames[host_id].gyro,
                state_->frames[target_id].gyro, state_->frames[host_id].acc,
                state_->frames[target_id].acc, state_->frames[host_id].speed_bias.data(),
                state_->frames[target_id].speed_bias.data(),
                state_->frames[host_id].speed_bias.data() + 6,
                state_->frames[target_id].speed_bias.data() + 6,
                state_->frames[host_id].speed_bias.data() + 3,
                state_->frames[target_id].speed_bias.data() + 3, sqrt_info, state_->camera,
                f->observations[0].sync_delay, f->observations[obs_idx].sync_delay);
            const ceres::ResidualBlockId residual = problem.AddResidualBlock(
                cost, loss, state_->frames[host_id].pose.data(),
                state_->frames[target_id].pose.data(), &state_->param_delay_time,
                &inv_depth_params[k]);
            visual_residuals.push_back(residual);
        }
    }

    visitPreintegrators([&](auto& preintegrators) {
        using Integrator = typename std::decay_t<decltype(preintegrators)>::value_type;
        const int imu_start = state_->firstImuFactorSlot();
        for (int i = imu_start; i < state_->newest_slot; ++i) {
            if (preintegrators[i].buffer.size() < 2) {
                continue;
            }
            TASSEL_ASSERT(preintegrators[i].start_frame_id == state_->frames[i].timestamp_ns);
            TASSEL_ASSERT(preintegrators[i].end_frame_id == state_->frames[i + 1].timestamp_ns);
            auto pint_ptr = std::shared_ptr<Integrator>(&preintegrators[i], [](Integrator*) {});
            auto* imu_cost = new IMUFactor<Integrator>(pint_ptr);
            imu_residuals.push_back(problem.AddResidualBlock(
                imu_cost, nullptr, state_->frames[i].pose.data(),
                state_->frames[i].speed_bias.data(), state_->frames[i + 1].pose.data(),
                state_->frames[i + 1].speed_bias.data()));
        }
    });

    auto evaluate_cost = [&](const std::vector<ceres::ResidualBlockId>& residuals) {
        if (residuals.empty()) {
            return 0.0;
        }
        ceres::Problem::EvaluateOptions options;
        options.apply_loss_function = true;
        options.residual_blocks = residuals;
        double cost = 0.0;
        if (!problem.Evaluate(options, &cost, nullptr, nullptr, nullptr)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return cost;
    };

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = params_.num_iterations;
    opts.num_threads = params_.num_threads;
    opts.minimizer_progress_to_stdout = false;
    opts.logging_type = ceres::SILENT;

    const double prior_cost_before = evaluate_cost(prior_residuals);
    const double visual_cost_before = evaluate_cost(visual_residuals);
    const double imu_cost_before = evaluate_cost(imu_residuals);

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    bool finite_solution = std::isfinite(state_->param_delay_time);
    for (int i = 0; i <= latest_id && finite_solution; ++i) {
        finite_solution =
            allFinite(state_->frames[i].pose) && allFinite(state_->frames[i].speed_bias);
    }
    finite_solution = finite_solution && allFinite(inv_depth_params);
    if (!summary.IsSolutionUsable() || !finite_solution) {
        spdlog::error("Optimization rejected: {}", summary.BriefReport());
        state_->stateToParams();
        return;
    }

    const double prior_cost_after = evaluate_cost(prior_residuals);
    const double visual_cost_after = evaluate_cost(visual_residuals);
    const double imu_cost_after = evaluate_cost(imu_residuals);
    if (spdlog::should_log(spdlog::level::info)) {
        const Eigen::Vector3d final_ba(
            state_->frames[latest_id].speed_bias[3], state_->frames[latest_id].speed_bias[4],
            state_->frames[latest_id].speed_bias[5]);
        const Eigen::Vector3d final_bg(
            state_->frames[latest_id].speed_bias[6], state_->frames[latest_id].speed_bias[7],
            state_->frames[latest_id].speed_bias[8]);
        spdlog::info(
            "Optimization\n"
            "  visual: {:.3e} -> {:.3e}\n"
            "  prior:  {:.3e} -> {:.3e}\n"
            "  imu:    {:.3e} -> {:.3e}\n"
            "  Ba: ({:.5f}, {:.5f}, {:.5f})\n"
            "  Bg: ({:.5f}, {:.5f}, {:.5f})\n"
            "  delay: {:.6f}",
            visual_cost_before, visual_cost_after, prior_cost_before, prior_cost_after,
            imu_cost_before, imu_cost_after, final_ba.x(), final_ba.y(), final_ba.z(), final_bg.x(),
            final_bg.y(), final_bg.z(), state_->param_delay_time);
    }

    if (optimization_callback_) {
        OptimizationStats stats;
        stats.total_cost_before = summary.initial_cost;
        stats.total_cost_after = summary.final_cost;
        stats.visual_cost_before = visual_cost_before;
        stats.visual_cost_after = visual_cost_after;
        stats.prior_cost_before = prior_cost_before;
        stats.prior_cost_after = prior_cost_after;
        stats.imu_cost_before = imu_cost_before;
        stats.imu_cost_after = imu_cost_after;
        stats.visual_factors_per_frame = visual_factors_per_frame;
        optimization_callback_(timestamp >= 0.0 ? timestamp : last_ts_, stats);
    }

    state_->paramsToState();

    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
        features[k]->estimated_depth = d;
    }

    visitPreintegrators([&](auto& preintegrators) {
        for (int i = state_->firstImuFactorSlot(); i < state_->newest_slot; ++i) {
            const double delta_ba = (state_->frames[i].Ba - preintegrators[i].ba_linearized).norm();
            const double delta_bg = (state_->frames[i].Bg - preintegrators[i].bg_linearized).norm();
            if (delta_ba > params_.imu_repropagate_ba_threshold ||
                delta_bg > params_.imu_repropagate_bg_threshold) {
                preintegrators[i].repropagate(state_->frames[i].Ba, state_->frames[i].Bg, noise_);
            }
        }
    });
}

void Estimator::updateMarginalizationPrior(const MarginalizationLayout& layout) {
    const int window_capacity = state_->max_frame_count;
    TASSEL_ASSERT(window_capacity >= 3);
    state_->stateToParams();
    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;

    // 在当前状态处表达已有先验，但不修改其保存的线性化点。
    // 旋转列仅被传输到当前状态的右扰动切空间。
    MargLinData prior_in_current_tangent;
    const MargLinData* prior_to_linearize = nullptr;
    if (marginalization_prior_) {
        const int num_kept = static_cast<int>(marginalization_prior_->linearization_poses.size());
        TASSEL_ASSERT(num_kept == window_capacity - 1);
        std::vector<std::array<double, 6>> current_poses(num_kept);
        std::vector<std::array<double, 9>> current_speed_bias(num_kept);
        for (int i = 0; i < num_kept; ++i) {
            current_poses[i] = state_->frames[i].pose;
            current_speed_bias[i] = state_->frames[i].speed_bias;
        }
        prior_in_current_tangent = MargHelper::transportPriorToCurrentTangent(
            *marginalization_prior_, current_poses, current_speed_bias, state_->param_delay_time);
        prior_to_linearize = &prior_in_current_tangent;
    }

    // 仅将宿主帧到退出帧之间的观测写入先验。
    // 同一路标的其他观测继续保持活跃，并保留非线性优化能力。
    auto retiring_observations = feature_manager_->collectMarginalizedObservations(0, 1);
    visitPreintegrators([&](auto& preintegrators) {
        using Integrator = typename std::decay_t<decltype(preintegrators)>::value_type;
        std::vector<IntegratorBase<Integrator>*> imu_preintegrators;
        for (int i = 0; i < layout.imuFactorCount(); ++i) {
            const int imu_slot = layout.firstImuFactorSlot() + i;
            TASSEL_ASSERT(preintegrators[imu_slot].buffer.size() >= 2);
            imu_preintegrators.push_back(&preintegrators[imu_slot]);
        }

        auto linearizer = MarginalizationSqrt<Integrator>(
            retiring_observations, std::make_unique<ceres::HuberLoss>(visual_huber_delta), state_,
            imu_preintegrators, params_.ric, params_.tic, prior_to_linearize,
            layout.firstImuFactorSlot());
        linearizer.allocate();
        linearizer.linearize();
        linearizer.eliminateLandmarks();

        Eigen::MatrixXd reduced_jacobian;
        Eigen::VectorXd reduced_residual;
        linearizer.buildReducedSystem(reduced_jacobian, reduced_residual);

        // QR 前的列顺序为 [state0(15), state1(15), ..., delay]。
        // 保留 state0 位姿，消去其已停用的速度/偏置占位列和最旧活跃状态 state1。
        constexpr int host_pose_size = MarginalizationLayout::kPoseSize;
        constexpr int host_speed_bias_size = MarginalizationLayout::kSpeedBiasSize;
        constexpr int full_state_size = MarginalizationLayout::kFullStateSize;
        const int eliminated_size = layout.eliminatedParameterSize();
        const int remaining_size = reduced_jacobian.cols() - eliminated_size;
        Eigen::MatrixXd elimination_ordered_jacobian =
            layout.reorderForElimination(reduced_jacobian);

        Eigen::MatrixXd compact_prior_jacobian;
        Eigen::VectorXd prior_residual;
        MargHelper::eliminateSquareRootSystem(
            eliminated_size, remaining_size, elimination_ordered_jacobian, reduced_residual,
            compact_prior_jacobian, prior_residual);

        Eigen::MatrixXd prior_jacobian;
        if (layout.hostAction() == RetainedHostAction::kKeep) {
            // 槽位 0 已经是保留宿主帧，其速度/偏置列保持结构零，
            // 无需再次参与消元。
            prior_jacobian = std::move(compact_prior_jacobian);
        } else {
            // 创建新的保留宿主帧，将仅含位姿的紧凑块扩展为
            // PriorFactor 使用的统一 15 维先验接口。
            prior_jacobian = Eigen::MatrixXd::Zero(
                compact_prior_jacobian.rows(), (window_capacity - 1) * full_state_size + 1);
            prior_jacobian.leftCols(host_pose_size) =
                compact_prior_jacobian.leftCols(host_pose_size);
            prior_jacobian.block(
                0, full_state_size, compact_prior_jacobian.rows(),
                (window_capacity - 2) * full_state_size) =
                compact_prior_jacobian.block(
                    0, host_pose_size, compact_prior_jacobian.rows(),
                    (window_capacity - 2) * full_state_size);
            prior_jacobian.col(prior_jacobian.cols() - 1) =
                compact_prior_jacobian.col(compact_prior_jacobian.cols() - 1);
        }
        TASSEL_ASSERT(prior_jacobian.allFinite());
        TASSEL_ASSERT(prior_residual.allFinite());
        TASSEL_ASSERT(
            prior_jacobian.middleCols(host_pose_size, host_speed_bias_size).isZero(1e-12));

        auto updated_prior = std::make_unique<MargLinData>();
        updated_prior->H = std::move(prior_jacobian);
        updated_prior->b = std::move(prior_residual);
        updated_prior->linearization_poses.resize(window_capacity - 1);
        updated_prior->linearization_speed_bias.resize(window_capacity - 1);
        updated_prior->linearization_delay_time = state_->param_delay_time;
        const int retained_host_source_slot = layout.nextRetainedHostSourceSlot();
        updated_prior->linearization_poses[0] = state_->frames[retained_host_source_slot].pose;
        updated_prior->linearization_speed_bias[0] =
            state_->frames[retained_host_source_slot].speed_bias;
        for (int i = 2; i < window_capacity; ++i) {
            updated_prior->linearization_poses[i - 1] = state_->frames[i].pose;
            updated_prior->linearization_speed_bias[i - 1] = state_->frames[i].speed_bias;
        }
        marginalization_prior_ = std::move(updated_prior);
        state_->has_retained_host = true;
        spdlog::info(
            "Retained host: id={} retired_state={} prior={}x{}",
            state_->frames[retained_host_source_slot].timestamp_ns,
            layout.replacesRetainedHost() ? state_->frames[0].timestamp_ns
                                          : state_->frames[1].timestamp_ns,
            marginalization_prior_->H.rows(), marginalization_prior_->H.cols());
    });
}

void Estimator::slideInitializationWindow() {
    TASSEL_ASSERT(!state_->has_retained_host);
    const int n = state_->max_frame_count;
    for (int i = 0; i < n - 1; ++i) {
        state_->copyFrameSlot(i + 1, i);
    }
    state_->frames[n - 1].timestamp_ns = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        for (int i = 0; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(state_->frames[n - 2].Ba, state_->frames[n - 2].Bg, noise_);
        preintegrators.back().clearFrameInterval();
    });
}

void Estimator::shiftWindowAfterMarginalization(const MarginalizationLayout& layout) {
    const int window_capacity = state_->max_frame_count;
    constexpr int first_movable_slot = 1;
    TASSEL_ASSERT(state_->has_retained_host);

    if (layout.replacesRetainedHost()) {
        state_->copyFrameSlot(1, 0);
    }

    for (int i = first_movable_slot; i < window_capacity - 1; ++i) {
        state_->copyFrameSlot(i + 1, i);
    }
    state_->frames[window_capacity - 1].timestamp_ns = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(state_->frames[0].Ba, state_->frames[0].Bg, noise_);
        preintegrators[0].clearFrameInterval();
        for (int i = first_movable_slot; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(
            state_->frames[window_capacity - 2].Ba, state_->frames[window_capacity - 2].Bg, noise_);
        preintegrators.back().clearFrameInterval();
    });
}

Eigen::Matrix<double, 18, 18> Estimator::initNoise() const {
    Eigen::Matrix<double, 18, 18> noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (params_.acc_n * params_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (params_.gyr_n * params_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (params_.acc_n * params_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (params_.gyr_n * params_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (params_.acc_w * params_.acc_w) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (params_.gyr_w * params_.gyr_w) * Eigen::Matrix3d::Identity();
    return noise;
}

bool Estimator::tryInitialize() {
    int frame_count = state_->newest_slot;
    int n_frames = frame_count + 1;

    InitialSFM sfm(
        params_.sfm_min_seed_pts, params_.sfm_min_e_inliers, params_.sfm_e_ransac_threshold,
        params_.sfm_min_pnp_pts, params_.sfm_pnp_reproj_threshold, params_.sfm_max_bad_pnp_ratio,
        params_.sfm_ba_max_iterations, params_.sfm_ba_num_threads);
    if (!sfm.construct(*state_, *feature_manager_, params_.ric, Rs_, Ps_)) {
        spdlog::info("VIO initialization: SFM failed");
        return false;
    }

    {
        std::vector<Eigen::Matrix3d> dq_dbgs, delta_qs;
        visitPreintegrators([&](const auto& preintegrators) {
            for (int i = 0; i < frame_count; ++i) {
                dq_dbgs.push_back(preintegrators[i].get_dq_dbg());
                delta_qs.push_back(preintegrators[i].final_delta_q);
            }
        });
        Eigen::Vector3d bg = solveGyroBias(Rs_, dq_dbgs, delta_qs, params_.ric);
        if (!bg.allFinite()) {
            spdlog::info("VIO initialization: gyro bias solve failed");
            return false;
        }
        for (int i = 0; i <= frame_count; ++i) {
            state_->frames[i].Bg = bg;
        }
    }

    visitPreintegrators([&](auto& preintegrators) {
        for (int i = 0; i < frame_count; ++i) {
            preintegrators[i].repropagate(state_->frames[i].Ba, state_->frames[i].Bg, noise_);
        }
    });

    std::vector<Eigen::Vector3d> delta_ps, delta_vs;
    std::vector<double> dts;
    visitPreintegrators([&](const auto& preintegrators) {
        for (int i = 0; i < n_frames - 1; ++i) {
            delta_ps.push_back(preintegrators[i].final_delta_p);
            delta_vs.push_back(preintegrators[i].final_delta_v);
            dts.push_back(preintegrators[i].sum_dt);
        }
    });

    Eigen::Vector3d g;
    double s;
    if (!linearAlignment(
            Rs_, Ps_, Vs_, delta_vs, delta_ps, dts, g, s, params_.ric, params_.tic,
            params_.gravity_diff_threshold, params_.g_norm)) {
        spdlog::info("VI initialization: linear alignment failed");
        return false;
    }

    if (!refineGravitySpeeds(
            Vs_, Rs_, Ps_, delta_vs, delta_ps, dts, g, s, params_.ric, params_.tic,
            params_.g_norm)) {
        spdlog::info("VI initialization: gravity refinement failed");
        return false;
    }

    Eigen::Matrix3d R0 =
        Eigen::Quaterniond::FromTwoVectors((g).normalized(), Eigen::Vector3d(0, 0, 1))
            .toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;

    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    initialized_ = true;

    for (int i = 0; i <= frame_count; ++i) {
        state_->frames[i].R =
            Eigen::Quaterniond(R0 * params_.ric * Rs_[i] * params_.ric.transpose())
                .normalized()
                .toRotationMatrix();
        state_->frames[i].P =
            R0 * (params_.ric * s * Ps_[i] -
                  params_.ric * Rs_[i] * params_.ric.transpose() * params_.tic + params_.tic);
        state_->frames[i].V = R0 * Vs_[i];
    }

    spdlog::info(
        "VI init: |g|={:.4f} s={:.4f} R0_yaw={:.2f}°", tassel_utils::G.norm(), s,
        yaw * 180.0 / M_PI);
    return true;
}

}  // namespace tassel_core
