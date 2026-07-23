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
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <vector>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/reprojection_factor.h"
#include "imu_interpolation.h"
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
        case tassel_utils::IntegratorType::MidPoint:
            preintegrators_ =
                makePreintegrators<MidPointIntegrator>(state_->max_frame_count - 1, noise_);
            break;
        case tassel_utils::IntegratorType::Euler:
            preintegrators_ =
                makePreintegrators<EulerIntegrator>(state_->max_frame_count - 1, noise_);
            break;
    }
    Rs_.resize(state_->max_frame_count, Eigen::Matrix3d::Identity());
    Ps_.resize(state_->max_frame_count, Eigen::Vector3d::Zero());
    Vs_.resize(state_->max_frame_count, Eigen::Vector3d::Zero());
    marginalization_prior_.reset();
    last_measurement_was_keyframe_ = false;
    frame_images_.clear();
    loop_keyframes_.clear();
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    state_->reset();
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay) {
    int& frame_count = state_->latest_frame_index;
    state_->frames[frame_count].timestamp_ns = frame_id;
    const double ts = tassel_utils::frameIdToSeconds(frame_id);
    state_->frames[frame_count].sync_delay = sync_delay;
    if (last_ts_ < 0 && !imu_measurements.empty()) {
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc - params_.acc_bias;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    const bool is_keyframe = feature_manager_->addFeatureFrame(frame_count, feature_frame);
    state_->frames[frame_count].type = is_keyframe ? FrameType::KeyFrame : FrameType::NonKeyFrame;
    last_measurement_was_keyframe_ = is_keyframe;
    predictFrameState(frame_count, imu_measurements);

    const double imu_query_timestamp = ts + sync_delay;
    interpolateBodyImu(
        imu_measurements, imu_query_timestamp, state_->frames[frame_count].gyro,
        state_->frames[frame_count].acc);
    if (!initialized_) {
        // 初始化期间将每个相机帧保留为独立的 VIO 状态。
        // SFM 前端从有限窗口中选择条件良好的种子帧对，关键帧分类不能将
        // 多帧 IMU 样本合并为过长的预积分区间。
        if (frame_count < state_->max_frame_count - 1) {
            ++frame_count;
            state_->copyFrameState(frame_count - 1, frame_count);
            state_->frames[frame_count].timestamp_ns = tassel_utils::kInvalidFrameId;
            int next_idx = frame_count - 1;
            visitPreintegrators([&](auto& preintegrators) {
                if (next_idx < static_cast<int>(preintegrators.size())) {
                    preintegrators[next_idx].reset(
                        state_->frames[frame_count - 1].Ba, state_->frames[frame_count - 1].Bg,
                        noise_);
                }
            });
            return;
        }

        if (!tryInitialize()) {
            spdlog::info("VI initialization not ready; sliding initialization window");
            feature_manager_->removeFrameObservations(0, *state_, params_.ric, params_.tic);
            slideInitializationWindow();
            return;
        }
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
    RetainedHostAction host_action = RetainedHostAction::Keep;
    if (!marginalization_prior_) {
        host_action = RetainedHostAction::Create;
    } else if (state_->frames[1].type == FrameType::KeyFrame) {
        host_action = RetainedHostAction::Replace;
    }
    if (host_action == RetainedHostAction::Replace && loop_closure_) {
        const FrameState& retired_host = state_->frames[0];
        loop_closure_->submitPose(
            {retired_host.timestamp_ns, Sophus::SE3d(retired_host.R, retired_host.P)});
        const auto host_landmarks = feature_manager_->exportHostLandmarks(0, *state_);
        std::vector<tassel_loop::LandmarkInput> landmarks;
        landmarks.reserve(host_landmarks.size());
        for (const HostLandmark& landmark : host_landmarks) {
            landmarks.push_back(
                {landmark.feature_id, landmark.host_pixel, landmark.host_uv, landmark.host_depth});
        }
        const auto keyframe = loop_keyframes_.find(state_->frames[0].timestamp_ns);
        if (keyframe != loop_keyframes_.end()) {
            loop_closure_->submitLandmarks({keyframe->second, std::move(landmarks)});
            loop_keyframes_.erase(keyframe);
        }
    }
    if (host_action != RetainedHostAction::Keep && loop_closure_) {
        const int retained_host_index = host_action == RetainedHostAction::Replace ? 1 : 0;
        const FrameState& retained_host = state_->frames[retained_host_index];
        const auto image = frame_images_.find(retained_host.timestamp_ns);
        if (image != frame_images_.end()) {
            const tassel_loop::KeyframeId keyframe_id = loop_closure_->submitKeyframe(
                {retained_host.timestamp_ns, image->second,
                 Sophus::SE3d(retained_host.R, retained_host.P)});
            loop_keyframes_.emplace(retained_host.timestamp_ns, keyframe_id);
            frame_images_.erase(
                frame_images_.begin(), frame_images_.upper_bound(retained_host.timestamp_ns));
        }
    }
    updateMarginalizationPrior(host_action);
    if (host_action == RetainedHostAction::Replace) {
        feature_manager_->replaceRetainedHost(0, 1, *state_, params_.ric, params_.tic);
    } else {
        feature_manager_->removeFrameObservations(1, *state_, params_.ric, params_.tic);
    }
    shiftWindowAfterMarginalization(host_action);
}

void Estimator::predictFrameState(
    int frame_index, const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    if (frame_index == 0) {
        return;
    }

    TASSEL_ASSERT(frame_index > 0 && frame_index <= state_->latest_frame_index);
    const FrameState& previous_frame = state_->frames[frame_index - 1];
    FrameState& predicted_frame = state_->frames[frame_index];
    TASSEL_ASSERT(previous_frame.timestamp_ns != tassel_utils::kInvalidFrameId);
    TASSEL_ASSERT(predicted_frame.timestamp_ns > previous_frame.timestamp_ns);

    Eigen::Matrix3d rotation = predicted_frame.R;
    Eigen::Vector3d position = predicted_frame.P;
    Eigen::Vector3d velocity = predicted_frame.V;
    const Eigen::Vector3d acc_bias = predicted_frame.Ba;
    const Eigen::Vector3d gyro_bias = predicted_frame.Bg;

    visitPreintegrators([&](auto& preintegrators) {
        auto& preintegrator = preintegrators[frame_index - 1];

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
    const int latest_id = state_->latest_frame_index;
    state_->stateToParams();
    auto features = feature_manager_->collectLandmarks();

    ceres::Problem problem;
    std::vector<int> visual_factors_per_frame(state_->latest_frame_index + 1, 0);

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->frames[i].pose.data(), 6, se3_manifold);
        problem.AddParameterBlock(state_->frames[i].speed_bias.data(), 9);
    }
    if (marginalization_prior_) {
        problem.SetParameterBlockConstant(state_->frames[0].speed_bias.data());
    }
    if (!marginalization_prior_) {
        problem.SetParameterBlockConstant(state_->frames[0].pose.data());
    }

    problem.AddParameterBlock(&state_->param_delay_time, 1);
    int delay_observable_frames = 0;
    for (int i = 0; i <= state_->latest_frame_index; ++i) {
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
        problem.AddResidualBlock(prior_cost, nullptr, prior_blocks);
    }

    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;
    ceres::LossFunction* loss = new ceres::HuberLoss(visual_huber_delta);
    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < params_.max_depth) ? (1.0 / d) : 1.0;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
    }

    Eigen::Matrix2d sqrt_info = state_->visual_sqrt_info;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        const int host_id = f->host_frame_index;
        if (host_id < 0 || host_id > latest_id) {
            throw std::logic_error("Feature host index is outside the active window");
        }
        for (size_t obs_idx = 0; obs_idx < f->observations.size(); ++obs_idx) {
            const int observation_index = f->observationFrameIndex(obs_idx);
            if (observation_index <= latest_id) {
                ++visual_factors_per_frame[observation_index];
            }
        }
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            const int target_id = f->observationFrameIndex(obs_idx);
            if (target_id > latest_id) {
                throw std::logic_error("Feature target index is outside the active window");
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
            problem.AddResidualBlock(
                cost, loss, state_->frames[host_id].pose.data(),
                state_->frames[target_id].pose.data(), &state_->param_delay_time,
                &inv_depth_params[k]);
        }
    }

    visitPreintegrators([&](auto& preintegrators) {
        using Integrator = typename std::decay_t<decltype(preintegrators)>::value_type;
        const int imu_start = marginalization_prior_ ? 1 : 0;
        for (int i = imu_start; i < state_->latest_frame_index; ++i) {
            if (preintegrators[i].buffer.size() < 2) {
                continue;
            }
            auto pint_ptr = std::shared_ptr<Integrator>(&preintegrators[i], [](Integrator*) {});
            auto* imu_cost = new IMUFactor<Integrator>(pint_ptr);
            problem.AddResidualBlock(
                imu_cost, nullptr, state_->frames[i].pose.data(),
                state_->frames[i].speed_bias.data(), state_->frames[i + 1].pose.data(),
                state_->frames[i + 1].speed_bias.data());
        }
    });

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = params_.num_iterations;
    opts.num_threads = params_.num_threads;
    opts.minimizer_progress_to_stdout = false;
    opts.logging_type = ceres::SILENT;

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

    if (spdlog::should_log(spdlog::level::info)) {
        const Eigen::Vector3d final_ba(
            state_->frames[latest_id].speed_bias[3], state_->frames[latest_id].speed_bias[4],
            state_->frames[latest_id].speed_bias[5]);
        const Eigen::Vector3d final_bg(
            state_->frames[latest_id].speed_bias[6], state_->frames[latest_id].speed_bias[7],
            state_->frames[latest_id].speed_bias[8]);
        spdlog::info(
            "Optimization\n"
            "  Ba: ({:.5f}, {:.5f}, {:.5f})\n"
            "  Bg: ({:.5f}, {:.5f}, {:.5f})\n"
            "  delay: {:.6f}",
            final_ba.x(), final_ba.y(), final_ba.z(), final_bg.x(), final_bg.y(), final_bg.z(),
            state_->param_delay_time);
    }

    if (visual_factor_callback_) {
        visual_factor_callback_(timestamp >= 0.0 ? timestamp : last_ts_, visual_factors_per_frame);
    }

    state_->paramsToState();

    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
        features[k]->estimated_depth = d;
    }

    visitPreintegrators([&](auto& preintegrators) {
        const int first_imu_index = marginalization_prior_ ? 1 : 0;
        for (int i = first_imu_index; i < state_->latest_frame_index; ++i) {
            const double delta_ba = (state_->frames[i].Ba - preintegrators[i].ba_linearized).norm();
            const double delta_bg = (state_->frames[i].Bg - preintegrators[i].bg_linearized).norm();
            if (delta_ba > params_.imu_repropagate_ba_threshold ||
                delta_bg > params_.imu_repropagate_bg_threshold) {
                preintegrators[i].repropagate(state_->frames[i].Ba, state_->frames[i].Bg, noise_);
            }
        }
    });
}

void Estimator::updateMarginalizationPrior(RetainedHostAction action) {
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
        const int first_imu_index = action == RetainedHostAction::Create ? 0 : 1;
        const int imu_factor_count = action == RetainedHostAction::Create ? 2 : 1;
        for (int i = 0; i < imu_factor_count; ++i) {
            const int imu_index = first_imu_index + i;
            TASSEL_ASSERT(preintegrators[imu_index].buffer.size() >= 2);
            imu_preintegrators.push_back(&preintegrators[imu_index]);
        }

        auto linearizer = MarginalizationSqrt<Integrator>(
            retiring_observations, std::make_unique<ceres::HuberLoss>(visual_huber_delta), state_,
            imu_preintegrators, params_.ric, params_.tic, prior_to_linearize, first_imu_index);
        linearizer.allocate();
        linearizer.linearize();
        linearizer.marginalizeLandmarks();

        Eigen::MatrixXd reduced_jacobian;
        Eigen::VectorXd reduced_residual;
        linearizer.buildReducedSystem(reduced_jacobian, reduced_residual);

        // QR 前的列顺序为 [state0(15), state1(15), ..., delay]。
        // 保留 state0 位姿，消去其已停用的速度/偏置占位列和最旧活跃状态 state1。
        constexpr int host_pose_size = MargHelper::kPoseSize;
        constexpr int host_speed_bias_size = MargHelper::kSpeedBiasSize;
        constexpr int full_state_size = MargHelper::kFullStateSize;
        const int marginalized_size = action == RetainedHostAction::Keep
                                          ? full_state_size
                                          : full_state_size + host_speed_bias_size;
        const int remaining_size = reduced_jacobian.cols() - marginalized_size;
        Eigen::MatrixXd marginalization_ordered_jacobian =
            MargHelper::reorderForMarginalization(reduced_jacobian, action);

        Eigen::MatrixXd compact_prior_jacobian;
        Eigen::VectorXd prior_residual;
        MargHelper::marginalizeSquareRootSystem(
            marginalized_size, remaining_size, marginalization_ordered_jacobian, reduced_residual,
            compact_prior_jacobian, prior_residual);

        Eigen::MatrixXd prior_jacobian;
        if (action == RetainedHostAction::Keep) {
            // 索引 0 已经是保留宿主帧，其速度/偏置列保持结构零，
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
        const int retained_host_source_index = action == RetainedHostAction::Replace ? 1 : 0;
        updated_prior->linearization_poses[0] = state_->frames[retained_host_source_index].pose;
        updated_prior->linearization_speed_bias[0] =
            state_->frames[retained_host_source_index].speed_bias;
        for (int i = 2; i < window_capacity; ++i) {
            updated_prior->linearization_poses[i - 1] = state_->frames[i].pose;
            updated_prior->linearization_speed_bias[i - 1] = state_->frames[i].speed_bias;
        }
        marginalization_prior_ = std::move(updated_prior);
    });
}

void Estimator::slideInitializationWindow() {
    TASSEL_ASSERT(!marginalization_prior_);
    const int n = state_->max_frame_count;
    for (int i = 0; i < n - 1; ++i) {
        state_->copyFrameState(i + 1, i);
    }
    state_->frames[n - 1].timestamp_ns = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        for (int i = 0; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(state_->frames[n - 2].Ba, state_->frames[n - 2].Bg, noise_);
    });
}

void Estimator::shiftWindowAfterMarginalization(RetainedHostAction action) {
    const int window_capacity = state_->max_frame_count;
    constexpr int first_movable_index = 1;
    TASSEL_ASSERT(marginalization_prior_);

    if (action == RetainedHostAction::Replace) {
        state_->copyFrameState(1, 0);
    }

    for (int i = first_movable_index; i < window_capacity - 1; ++i) {
        state_->copyFrameState(i + 1, i);
    }
    state_->frames[window_capacity - 1].timestamp_ns = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(state_->frames[0].Ba, state_->frames[0].Bg, noise_);
        for (int i = first_movable_index; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(
            state_->frames[window_capacity - 2].Ba, state_->frames[window_capacity - 2].Bg, noise_);
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
    int frame_count = state_->latest_frame_index;
    int n_frames = frame_count + 1;

    InitialSFM sfm(
        params_.sfm_min_seed_pts, params_.sfm_min_e_inliers, params_.sfm_e_ransac_threshold,
        params_.sfm_min_pnp_pts, params_.sfm_pnp_reproj_threshold, params_.sfm_max_bad_pnp_ratio,
        params_.sfm_epipolar_max_iterations, params_.sfm_epipolar_num_threads);
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
    if (std::abs(s) <= params_.init_scale_zero_threshold) {
        spdlog::info("VI initialization: degenerate scale {:.6f} treated as zero", s);
        s = 0.0;
    } else if (s < 0.0) {
        spdlog::info("VI initialization: negative scale {:.6f}", s);
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
