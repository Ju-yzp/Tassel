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
#include <string>
#include <vector>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/visual_factor.h"
#include "marg/marg_helper.h"
#include "marg/marginlization_sqrt.h"
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
    gravity_initialized_ = false;
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
    marg_data_.reset();
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    state_->reset();
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double applied_delay) {
    int& frame_count = state_->cur_frame_count;
    state_->frame_ids[frame_count] = frame_id;
    const double ts = tassel_utils::frameIdToSeconds(frame_id);
    state_->frame_delays[frame_count] = applied_delay;
    if (last_ts_ < 0 && !imu_measurements.empty()) {
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc - params_.acc_bias;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    const bool initialization_keyframe = feature_manager_->checkParallax(frame_id, feature_frame);
    const auto& input_stats = feature_manager_->lastInputStats();
    const double new_feature_ratio = input_stats.input_count > 0
                                         ? static_cast<double>(input_stats.created_count) /
                                               static_cast<double>(input_stats.input_count)
                                         : 1.0;
    const bool weak_keyframe_connection =
        !feature_manager_->hasLatestKeyframe() ||
        input_stats.current_keyframe_connection_ratio <= (1.0 - params_.keyframe_new_feature_ratio);
    const bool is_keyframe = initialization_keyframe || weak_keyframe_connection;
    if (is_keyframe) feature_manager_->acceptKeyframe(frame_id, feature_frame);
    spdlog::info(
        "Frame features: input={} matched={} new={} new_ratio={:.3f} kf_connected={} "
        "kf_current_ratio={:.3f} kf_retention={:.3f} parallax={:.3f} keyframe={}",
        input_stats.input_count, input_stats.matched_count, input_stats.created_count,
        new_feature_ratio, input_stats.connected_to_keyframe_count,
        input_stats.current_keyframe_connection_ratio, input_stats.keyframe_feature_retention_ratio,
        input_stats.average_parallax, gravity_initialized_ ? is_keyframe : initialization_keyframe);

    if (frame_count > 0) {
        TASSEL_ASSERT(state_->frame_ids[frame_count - 1] != tassel_utils::kInvalidFrameId);
        TASSEL_ASSERT(state_->frame_ids[frame_count] > state_->frame_ids[frame_count - 1]);
        Eigen::Matrix3d R = state_->Rs[frame_count];
        Eigen::Vector3d P = state_->Ps[frame_count];
        Eigen::Vector3d V = state_->Vs[frame_count];
        Eigen::Vector3d Ba = state_->Bas[frame_count];
        Eigen::Vector3d Bg = state_->Bgs[frame_count];
        visitPreintegrators([&](auto& preintegrators) {
            auto& preintegrator = preintegrators[frame_count - 1];
            preintegrator.setFrameInterval(
                state_->frame_ids[frame_count - 1], state_->frame_ids[frame_count]);

            for (const auto& imu : imu_measurements) {
                tassel_utils::IMUMeasurement imu_cal = imu;
                imu_cal.acc = imu.acc - params_.acc_bias;
                if (!preintegrator.propagate(imu_cal)) {
                    throw std::runtime_error("Invalid or non-monotonic IMU measurement");
                }

                double dt = imu_cal.timestamp - last_ts_;
                Eigen::Vector3d acc_0 = R * (last_imu_acc_ - Ba) - tassel_utils::G;
                Eigen::Vector3d gyr = 0.5 * (last_imu_gyro_ + imu_cal.gyro) - Bg;
                R = R * Sophus::SO3d::exp(gyr * dt).matrix();
                Eigen::Vector3d acc_1 = R * (imu_cal.acc - Ba) - tassel_utils::G;
                Eigen::Vector3d acc = 0.5 * (acc_0 + acc_1);
                P += V * dt + 0.5 * acc * dt * dt;
                V += acc * dt;
                last_ts_ = imu_cal.timestamp;
                last_imu_gyro_ = imu_cal.gyro;
                last_imu_acc_ = imu_cal.acc;
            }
        });

        Eigen::Quaterniond q(R);
        q.normalize();
        state_->Rs[frame_count] = q.matrix();
        state_->Ps[frame_count] = P;
        state_->Vs[frame_count] = V;
    }

    state_->acc_vec.push_back(last_imu_acc_);
    state_->gyro_vec.push_back(last_imu_gyro_);
    if (!gravity_initialized_) {
        // Keep every camera frame as an independent VIO state during initialization.
        // The SFM front-end selects a well-conditioned seed pair from the bounded
        // window; keyframe classification must not merge IMU samples into a long
        // preintegration interval.
        feature_manager_->triangulate(*state_, params_.ric, params_.tic);
        if (frame_count < state_->max_frame_count - 1) {
            ++frame_count;
            state_->copyFrameSlot(frame_count - 1, frame_count);
            state_->frame_ids[frame_count] = tassel_utils::kInvalidFrameId;
            int next_idx = frame_count - 1;
            visitPreintegrators([&](auto& preintegrators) {
                if (next_idx < static_cast<int>(preintegrators.size())) {
                    preintegrators[next_idx].reset(
                        state_->Bas[frame_count - 1], state_->Bgs[frame_count - 1], noise_);
                    preintegrators[next_idx].clearFrameInterval();
                }
            });
            return;
        }

        if (!tryInitialize()) {
            spdlog::warn("VI initialization failed, sliding initialization window");
            feature_manager_->removeOldest(*state_, params_.ric, params_.tic);
            slideInitializationWindow();
            return;
        }
        feature_manager_->invalidateDepths();
    }

    feature_manager_->triangulate(*state_, params_.ric, params_.tic);
    optimize(ts);
    const Sophus::SE3d optimized_pose(state_->Rs[frame_count], state_->Ps[frame_count]);
    if (pose_callback_) pose_callback_(ts, optimized_pose);
    if (realtime_pose_callback_) realtime_pose_callback_(ts, optimized_pose);
    const OutlierStats outliers =
        feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
    spdlog::info(
        "  reprojection: checked={} removed={} mean={:.3f}px max={:.3f}px "
        "removed_mean={:.3f}px",
        outliers.checked_count, outliers.removed_count, outliers.average_error,
        outliers.maximum_error, outliers.removed_average_error);
    const bool replacing_host = state_->has_retained_host &&
                                feature_manager_->isKeyframe(state_->frame_ids[1]) &&
                                state_->frame_ids[1] != state_->frame_ids[0];
    const WindowMarginalizationPlan marginalization_plan =
        WindowMarginalizationPlan::create(state_->has_retained_host, replacing_host);
    buildPrior(marginalization_plan);
    if (marginalization_plan.replacesHost()) {
        feature_manager_->replaceHost(
            state_->frame_ids[0], state_->frame_ids[1], *state_, params_.ric, params_.tic);
        feature_manager_->retireKeyframe(state_->frame_ids[0]);
    } else {
        feature_manager_->removeFrame(state_->frame_ids[1], *state_, params_.ric, params_.tic);
    }
    slideWindow(marginalization_plan);

    if (cloud_callback_ && gravity_initialized_) {
        auto pts = feature_manager_->getPointCloud(*state_, params_.ric, params_.tic);
        cloud_callback_(ts, pts);
    }
}

void Estimator::optimize(double timestamp) {
    const int latest_id = state_->cur_frame_count;
    const Eigen::Vector3d latest_position_before = state_->Ps[latest_id];
    const Eigen::Vector3d latest_velocity_before = state_->Vs[latest_id];
    const Eigen::Vector3d latest_ba_before = state_->Bas[latest_id];
    const Eigen::Vector3d latest_bg_before = state_->Bgs[latest_id];
    const double delay_before = state_->delay_time;
    const auto window_positions_before = state_->Ps;
    const auto window_rotations_before = state_->Rs;
    const auto window_velocities_before = state_->Vs;

    state_->stateToParams();
    auto features = feature_manager_->collectLandmarks();

    ceres::Problem problem;
    std::vector<ceres::ResidualBlockId> prior_residuals;
    std::vector<ceres::ResidualBlockId> visual_residuals;
    std::vector<ceres::ResidualBlockId> visual_two_observation_residuals;
    std::vector<ceres::ResidualBlockId> visual_mature_residuals;
    std::vector<ceres::ResidualBlockId> imu_residuals;
    std::vector<int> visual_factors_per_frame(state_->cur_frame_count + 1, 0);

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->params_pose[i].data(), 6, se3_manifold);
        problem.AddParameterBlock(state_->params_speed_bias[i].data(), 9);
    }
    if (state_->has_retained_host) {
        problem.SetParameterBlockConstant(state_->params_speed_bias[0].data());
    }
    if (!marg_data_) {
        problem.SetParameterBlockConstant(state_->params_pose[0].data());
    }

    problem.AddParameterBlock(&state_->param_delay_time, 1);
    bool set_dt_constant_flag = true;
    for (int i = 0; i < state_->cur_frame_count; ++i) {
        if ((state_->gyro_vec[i] - state_->Bgs[i]).norm() > params_.dt_gyro_threshold) {
            set_dt_constant_flag = false;
            break;
        }
    }

    if (set_dt_constant_flag) {
        problem.SetParameterBlockConstant(&state_->param_delay_time);
    }

    if (marg_data_) {
        auto* prior_cost = new MarginalizationPriorFactor(*marg_data_);
        std::vector<double*> prior_blocks;
        int num_kept = static_cast<int>(marg_data_->linearization_poses.size());
        for (int i = 0; i < num_kept; ++i) {
            prior_blocks.push_back(state_->params_pose[i].data());
            prior_blocks.push_back(state_->params_speed_bias[i].data());
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
        problem.SetParameterLowerBound(&inv_depth_params[k], 0, 1.0 / params_.max_depth);
        problem.SetParameterUpperBound(&inv_depth_params[k], 0, 1.0 / params_.min_depth);
    }

    Eigen::Matrix2d sqrt_info = state_->visual_sqrt_info;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        const int host_id = state_->findFrameSlot(f->host_frame_id);
        if (host_id < 0) continue;
        for (const auto& observation : f->observations) {
            const int observation_slot = state_->findFrameSlot(observation.frame_id);
            if (observation_slot >= 0) ++visual_factors_per_frame[observation_slot];
        }
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            const int target_id = state_->findFrameSlot(f->observations[obs_idx].frame_id);
            if (target_id < 0) continue;
            Eigen::Vector2d pt_j(f->observations[obs_idx].pt.x, f->observations[obs_idx].pt.y);
            auto* cost = new VisualFactor(
                f->observations[0].uv, pt_j, params_.ric, params_.tic, state_->gyro_vec[host_id],
                state_->gyro_vec[target_id], state_->acc_vec[host_id], state_->acc_vec[target_id],
                state_->params_speed_bias[host_id].data(),
                state_->params_speed_bias[target_id].data(),
                state_->params_speed_bias[host_id].data() + 6,
                state_->params_speed_bias[target_id].data() + 6,
                state_->params_speed_bias[host_id].data() + 3,
                state_->params_speed_bias[target_id].data() + 3, sqrt_info, state_->camera,
                f->observations[0].applied_delay, f->observations[obs_idx].applied_delay);
            const ceres::ResidualBlockId residual = problem.AddResidualBlock(
                cost, loss, state_->params_pose[host_id].data(),
                state_->params_pose[target_id].data(), &state_->param_delay_time,
                &inv_depth_params[k]);
            visual_residuals.push_back(residual);
            if (f->observations.size() == 2) {
                visual_two_observation_residuals.push_back(residual);
            } else {
                visual_mature_residuals.push_back(residual);
            }
        }
    }

    visitPreintegrators([&](auto& preintegrators) {
        using Integrator = typename std::decay_t<decltype(preintegrators)>::value_type;
        const int imu_start = state_->firstActiveImuSlot();
        for (int i = imu_start; i < state_->cur_frame_count; ++i) {
            if (preintegrators[i].buffer.size() < 2) continue;
            TASSEL_ASSERT(preintegrators[i].start_frame_id == state_->frame_ids[i]);
            TASSEL_ASSERT(preintegrators[i].end_frame_id == state_->frame_ids[i + 1]);
            auto pint_ptr = std::shared_ptr<Integrator>(&preintegrators[i], [](Integrator*) {});
            auto* imu_cost = new IMUFactor<Integrator>(pint_ptr);
            imu_residuals.push_back(problem.AddResidualBlock(
                imu_cost, nullptr, state_->params_pose[i].data(),
                state_->params_speed_bias[i].data(), state_->params_pose[i + 1].data(),
                state_->params_speed_bias[i + 1].data()));
        }
    });

    auto evaluate_cost = [&](const std::vector<ceres::ResidualBlockId>& residuals) {
        if (residuals.empty()) return 0.0;
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
    const double visual_two_observation_cost_before =
        evaluate_cost(visual_two_observation_residuals);
    const double visual_mature_cost_before = evaluate_cost(visual_mature_residuals);
    const double imu_cost_before = evaluate_cost(imu_residuals);

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    bool finite_solution = std::isfinite(state_->param_delay_time);
    for (int i = 0; i <= latest_id && finite_solution; ++i) {
        finite_solution =
            allFinite(state_->params_pose[i]) && allFinite(state_->params_speed_bias[i]);
    }
    finite_solution = finite_solution && allFinite(inv_depth_params);
    if (!summary.IsSolutionUsable() || !finite_solution) {
        spdlog::error("Optimization rejected: {}", summary.BriefReport());
        state_->stateToParams();
        return;
    }

    const double prior_cost_after = evaluate_cost(prior_residuals);
    const double visual_cost_after = evaluate_cost(visual_residuals);
    const double visual_two_observation_cost_after =
        evaluate_cost(visual_two_observation_residuals);
    const double visual_mature_cost_after = evaluate_cost(visual_mature_residuals);
    const double imu_cost_after = evaluate_cost(imu_residuals);
    const Eigen::Vector3d latest_position_after(
        state_->params_pose[latest_id][0], state_->params_pose[latest_id][1],
        state_->params_pose[latest_id][2]);
    const Eigen::Vector3d latest_velocity_after(
        state_->params_speed_bias[latest_id][0], state_->params_speed_bias[latest_id][1],
        state_->params_speed_bias[latest_id][2]);
    const Eigen::Vector3d latest_ba_after(
        state_->params_speed_bias[latest_id][3], state_->params_speed_bias[latest_id][4],
        state_->params_speed_bias[latest_id][5]);
    const Eigen::Vector3d latest_bg_after(
        state_->params_speed_bias[latest_id][6], state_->params_speed_bias[latest_id][7],
        state_->params_speed_bias[latest_id][8]);
    double maximum_position_update = 0.0;
    double maximum_rotation_update = 0.0;
    double maximum_velocity_update = 0.0;
    int maximum_position_frame = 0;
    int maximum_rotation_frame = 0;
    int maximum_velocity_frame = 0;
    for (int i = 0; i <= latest_id; ++i) {
        const Eigen::Vector3d position_after(
            state_->params_pose[i][0], state_->params_pose[i][1], state_->params_pose[i][2]);
        const Eigen::Matrix3d rotation_after =
            Sophus::SO3d::exp(Eigen::Vector3d(
                                  state_->params_pose[i][3], state_->params_pose[i][4],
                                  state_->params_pose[i][5]))
                .matrix();
        const Eigen::Vector3d velocity_after(
            state_->params_speed_bias[i][0], state_->params_speed_bias[i][1],
            state_->params_speed_bias[i][2]);
        const double position_update = (position_after - window_positions_before[i]).norm();
        const double rotation_update =
            Sophus::SO3d(window_rotations_before[i].transpose() * rotation_after).log().norm();
        const double velocity_update = (velocity_after - window_velocities_before[i]).norm();
        if (position_update > maximum_position_update) {
            maximum_position_update = position_update;
            maximum_position_frame = i;
        }
        if (rotation_update > maximum_rotation_update) {
            maximum_rotation_update = rotation_update;
            maximum_rotation_frame = i;
        }
        if (velocity_update > maximum_velocity_update) {
            maximum_velocity_update = velocity_update;
            maximum_velocity_frame = i;
        }
    }

    std::string visual_counts;
    for (size_t i = 0; i < visual_factors_per_frame.size(); ++i) {
        if (!visual_counts.empty()) visual_counts += ',';
        visual_counts += std::to_string(visual_factors_per_frame[i]);
    }
    spdlog::info(
        "Optimization\n"
        "  visual: {:.3e} -> {:.3e}\n"
        "  visual obs=2: {:.3e} -> {:.3e}, mature: {:.3e} -> {:.3e}\n"
        "  prior:  {:.3e} -> {:.3e}\n"
        "  imu:    {:.3e} -> {:.3e}\n"
        "  latest state: |V| {:.4f} -> {:.4f}, |dP| {:.4f}, |dV| {:.4f}\n"
        "  delay: {:.6f} -> {:.6f}\n"
        "  Ba: ({:.5f}, {:.5f}, {:.5f}) -> ({:.5f}, {:.5f}, {:.5f})\n"
        "  Bg: ({:.5f}, {:.5f}, {:.5f}) -> ({:.5f}, {:.5f}, {:.5f})\n"
        "  window update: |dP|max {:.4f}@{}, |dR|max {:.4f}@{}, |dV|max {:.4f}@{}\n"
        "  visual landmarks: {}\n"
        "  visual/frame: [{}]",
        visual_cost_before, visual_cost_after, visual_two_observation_cost_before,
        visual_two_observation_cost_after, visual_mature_cost_before, visual_mature_cost_after,
        prior_cost_before, prior_cost_after, imu_cost_before, imu_cost_after,
        latest_velocity_before.norm(), latest_velocity_after.norm(),
        (latest_position_after - latest_position_before).norm(),
        (latest_velocity_after - latest_velocity_before).norm(), delay_before,
        state_->param_delay_time, latest_ba_before.x(), latest_ba_before.y(), latest_ba_before.z(),
        latest_ba_after.x(), latest_ba_after.y(), latest_ba_after.z(), latest_bg_before.x(),
        latest_bg_before.y(), latest_bg_before.z(), latest_bg_after.x(), latest_bg_after.y(),
        latest_bg_after.z(), maximum_position_update, maximum_position_frame,
        maximum_rotation_update, maximum_rotation_frame, maximum_velocity_update,
        maximum_velocity_frame, features.size(), visual_counts);

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
        for (int i = state_->firstActiveImuSlot(); i < state_->cur_frame_count; ++i) {
            const double delta_ba = (state_->Bas[i] - preintegrators[i].ba_linearized).norm();
            const double delta_bg = (state_->Bgs[i] - preintegrators[i].bg_linearized).norm();
            if (delta_ba > params_.imu_repropagate_ba_threshold ||
                delta_bg > params_.imu_repropagate_bg_threshold) {
                preintegrators[i].repropagate(state_->Bas[i], state_->Bgs[i], noise_);
            }
        }
    });
}

void Estimator::buildPrior(const WindowMarginalizationPlan& plan) {
    const int n = state_->max_frame_count;
    TASSEL_ASSERT(n >= 3);
    state_->stateToParams();
    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;

    // Linearize the fixed carried prior at the current state without mutating its stored
    // linearization point. Rotation columns are transported into the current right tangent.
    MargLinData rebased_prior;
    const MargLinData* prior_for_marg = nullptr;
    if (marg_data_) {
        const int num_kept = static_cast<int>(marg_data_->linearization_poses.size());
        TASSEL_ASSERT(num_kept == n - 1);
        std::vector<std::array<double, 6>> current_poses(num_kept);
        std::vector<std::array<double, 9>> current_speed_bias(num_kept);
        for (int i = 0; i < num_kept; ++i) {
            current_poses[i] = state_->params_pose[i];
            current_speed_bias[i] = state_->params_speed_bias[i];
        }
        rebased_prior = MargHelper::rebasePrior(
            *marg_data_, current_poses, current_speed_bias, state_->param_delay_time);
        prior_for_marg = &rebased_prior;
    }

    // Only the host-to-retiring-frame observation enters the prior. Other observations
    // of the same landmark remain active and nonlinear after this retirement.
    auto marg_features = feature_manager_->collectMarginalizedObservations(
        state_->frame_ids[0], state_->frame_ids[1]);
    visitPreintegrators([&](auto& preintegrators) {
        using Integrator = typename std::decay_t<decltype(preintegrators)>::value_type;
        std::vector<IntegratorBase<Integrator>*> pint_ptrs;
        for (int i = 0; i < plan.imu_block_count; ++i) {
            TASSEL_ASSERT(preintegrators[plan.imu_start_slot + i].buffer.size() >= 2);
            pint_ptrs.push_back(&preintegrators[plan.imu_start_slot + i]);
        }

        auto marg = MarginlizationSqrt<Integrator>(
            marg_features, std::make_unique<ceres::HuberLoss>(visual_huber_delta), state_,
            pint_ptrs, params_.ric, params_.tic, prior_for_marg, plan.imu_start_slot);
        marg.allocate();
        marg.linearize();
        marg.performQRAll();

        Eigen::MatrixXd Q2Jp;
        Eigen::VectorXd Q2r;
        marg.get_dense_Jp_b(Q2Jp, Q2r);

        // Column order before QR is [state0(15), state1(15), ..., delay]. Keep
        // state0 pose, marginalize its already-retired speed/bias placeholder and
        // the complete oldest active state1.
        constexpr int host_pose_size = WindowMarginalizationPlan::kPoseSize;
        constexpr int host_speed_bias_size = WindowMarginalizationPlan::kSpeedBiasSize;
        constexpr int full_state_size = WindowMarginalizationPlan::kFullStateSize;
        const int marg_size = host_speed_bias_size + full_state_size;
        const int compact_keep_size = host_pose_size + (n - 2) * full_state_size + 1;
        Eigen::MatrixXd ordered = plan.marginalColumnsFirst(Q2Jp);

        Eigen::MatrixXd compact_H;
        Eigen::VectorXd marg_b;
        MargHelper::marginalizeSqrtToSqrt(
            marg_size, compact_keep_size, ordered, Q2r, compact_H, marg_b);

        // Keep the existing uniform prior-factor interface. The retained host has
        // a 15D block, but its speed/bias columns are structurally zero.
        Eigen::MatrixXd marg_H =
            Eigen::MatrixXd::Zero(compact_H.rows(), (n - 1) * full_state_size + 1);
        marg_H.leftCols(host_pose_size) = compact_H.leftCols(host_pose_size);
        marg_H.block(0, full_state_size, compact_H.rows(), (n - 2) * full_state_size) =
            compact_H.block(0, host_pose_size, compact_H.rows(), (n - 2) * full_state_size);
        marg_H.col(marg_H.cols() - 1) = compact_H.col(compact_H.cols() - 1);
        TASSEL_ASSERT(marg_H.allFinite());
        TASSEL_ASSERT(marg_b.allFinite());
        TASSEL_ASSERT(marg_H.middleCols(host_pose_size, host_speed_bias_size).isZero(1e-12));

        auto new_prior = std::make_unique<MargLinData>();
        new_prior->H = marg_H;
        new_prior->b = marg_b;
        new_prior->linearization_poses.resize(n - 1);
        new_prior->linearization_speed_bias.resize(n - 1);
        new_prior->linearization_delay_time = state_->param_delay_time;
        const int retained_source = plan.retained_pose_source_slot;
        new_prior->linearization_poses[0] = state_->params_pose[retained_source];
        new_prior->linearization_speed_bias[0] = state_->params_speed_bias[retained_source];
        for (int i = 2; i < n; ++i) {
            new_prior->linearization_poses[i - 1] = state_->params_pose[i];
            new_prior->linearization_speed_bias[i - 1] = state_->params_speed_bias[i];
        }
        marg_data_ = std::move(new_prior);
        state_->has_retained_host = true;
        spdlog::info(
            "Retained host: id={} retired_state={} prior={}x{}", state_->frame_ids[retained_source],
            plan.replacesHost() ? state_->frame_ids[0] : state_->frame_ids[1], marg_data_->H.rows(),
            marg_data_->H.cols());
    });
}

void Estimator::slideInitializationWindow() {
    TASSEL_ASSERT(!state_->has_retained_host);
    const int n = state_->max_frame_count;
    for (int i = 0; i < n - 1; ++i) {
        state_->copyFrameSlot(i + 1, i);
    }
    state_->frame_ids[n - 1] = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        for (int i = 0; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(state_->Bas[n - 2], state_->Bgs[n - 2], noise_);
        preintegrators.back().clearFrameInterval();
    });
    state_->acc_vec.erase(state_->acc_vec.begin());
    state_->gyro_vec.erase(state_->gyro_vec.begin());
}

void Estimator::slideWindow(const WindowMarginalizationPlan& plan) {
    const int n = state_->max_frame_count;
    constexpr int first_movable_slot = 1;
    TASSEL_ASSERT(state_->has_retained_host);

    if (plan.replacesHost()) {
        state_->copyFrameSlot(1, 0);
    }

    for (int i = first_movable_slot; i < n - 1; ++i) {
        state_->copyFrameSlot(i + 1, i);
    }
    state_->frame_ids[n - 1] = tassel_utils::kInvalidFrameId;
    // 滑动预积分器
    visitPreintegrators([&](auto& preintegrators) {
        if (state_->has_retained_host) {
            preintegrators[0].reset(state_->Bas[0], state_->Bgs[0], noise_);
            preintegrators[0].clearFrameInterval();
        }
        for (int i = first_movable_slot; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(state_->Bas[n - 2], state_->Bgs[n - 2], noise_);
        preintegrators.back().clearFrameInterval();
    });

    state_->acc_vec.erase(state_->acc_vec.begin() + plan.removed_measurement_slot);
    state_->gyro_vec.erase(state_->gyro_vec.begin() + plan.removed_measurement_slot);
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
    int frame_count = state_->cur_frame_count;
    int n_frames = frame_count + 1;

    InitialSFM sfm(
        params_.sfm_min_seed_pts, params_.sfm_min_e_inliers, params_.sfm_e_ransac_threshold,
        params_.sfm_min_pnp_pts, params_.sfm_pnp_reproj_threshold, params_.sfm_max_bad_pnp_ratio,
        params_.sfm_ba_max_iterations, params_.sfm_ba_num_threads);
    if (!sfm.construct(*state_, *feature_manager_, params_.ric, Rs_, Ps_)) {
        spdlog::warn("VIO init: SFM failed");
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
            spdlog::warn("VIO init: gyro bias solve failed");
            return false;
        }
        for (int i = 0; i <= frame_count; ++i) state_->Bgs[i] = bg;
    }

    visitPreintegrators([&](auto& preintegrators) {
        for (int i = 0; i < frame_count; ++i) {
            preintegrators[i].repropagate(state_->Bas[i], state_->Bgs[i], noise_);
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
        spdlog::warn("VI init: linear alignment failed");
        return false;
    }

    if (!refineGravitySpeeds(
            Vs_, Rs_, Ps_, delta_vs, delta_ps, dts, g, s, params_.ric, params_.tic,
            params_.g_norm)) {
        spdlog::warn("VI init: gravity refinement failed");
        return false;
    }

    Eigen::Matrix3d R0 =
        Eigen::Quaterniond::FromTwoVectors((g).normalized(), Eigen::Vector3d(0, 0, 1))
            .toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;

    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    gravity_initialized_ = true;

    for (int i = 0; i <= frame_count; ++i) {
        state_->Rs[i] = Eigen::Quaterniond(R0 * params_.ric * Rs_[i] * params_.ric.transpose())
                            .normalized()
                            .toRotationMatrix();
        state_->Ps[i] =
            R0 * (params_.ric * s * Ps_[i] -
                  params_.ric * Rs_[i] * params_.ric.transpose() * params_.tic + params_.tic);
        state_->Vs[i] = R0 * Vs_[i];
    }

    spdlog::info(
        "VI init: |g|={:.4f} s={:.4f} R0_yaw={:.2f}°", tassel_utils::G.norm(), s,
        yaw * 180.0 / M_PI);
    return true;
}

}  // namespace tassel_core
