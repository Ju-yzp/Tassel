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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/reprojection_factor.h"
#include "factor/visual_frame_cache.h"
#include "frond_end/reprojection.h"
#include "imu_interpolation.h"
#include "marg/marg_helper.h"
#include "marg/marginalization_sqrt.h"
#include "marg/schmidt/schmidt_prior_covariance.h"
#include "marg/schmidt/schmidt_window_update.h"
#include "profiling/vtune_profile_scope.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

#include "initial/initial_alignment.h"
#include "initial/initial_sfm.h"

namespace tassel_core {

namespace {
constexpr int kRetainedFrameIndex = 0;
constexpr int kFirstActiveFrameIndex = 1;
constexpr double kStationarySpeed = 0.05;
constexpr size_t kStageTimingLogInterval = 100;

using StageClock = std::chrono::steady_clock;

double elapsedMilliseconds(StageClock::time_point begin, StageClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

ceres::TrustRegionStrategyType ceresTrustRegionStrategy(
    tassel_tools::TrustRegionStrategy strategy) {
    switch (strategy) {
        case tassel_tools::TrustRegionStrategy::LevenbergMarquardt:
            return ceres::LEVENBERG_MARQUARDT;
        case tassel_tools::TrustRegionStrategy::Dogleg:
            return ceres::DOGLEG;
    }
    throw std::logic_error("Unknown trust-region strategy");
}
constexpr double kGaugeHeadingMinNorm = 1e-8;
constexpr double kRotationTolerance = 1e-8;

double rotationYaw(const Eigen::Matrix3d& rotation) {
    if (!rotation.allFinite() ||
        !(rotation.transpose() * rotation)
             .isApprox(Eigen::Matrix3d::Identity(), kRotationTolerance) ||
        std::abs(rotation.determinant() - 1.0) > kRotationTolerance) {
        throw std::logic_error("Gauge rotation is not a valid SO(3) matrix");
    }
    if (std::hypot(rotation(0, 0), rotation(1, 0)) < kGaugeHeadingMinNorm) {
        throw std::logic_error("Gauge yaw is singular at vertical body x-axis");
    }
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

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

#if defined(CERES_HAS_SCHUR_LAYOUT_CALLBACK)
ceres::Solver::Options::SchurLayoutCallback makeSchurLayoutCallback() {
    return [](int chunk_index, const std::vector<int>& default_block_order,
              const std::vector<int>& block_sizes, std::vector<int>* block_order) {
        (void)chunk_index;
        if (block_order == nullptr) {
            throw std::invalid_argument("Schur layout callback received a null output");
        }
        if (default_block_order.size() != block_sizes.size()) {
            throw std::invalid_argument("Schur layout callback received mismatched layout data");
        }

        struct RankedBlock {
            int priority;
            int order;
            int block_id;
        };

        std::vector<RankedBlock> ranked;
        ranked.reserve(default_block_order.size());
        auto blockPriority = [](int size) {
            if (size == 6) {
                return 0;
            }
            if (size == 9) {
                return 1;
            }
            if (size == 1) {
                return 2;
            }
            return 3;
        };
        for (int i = 0; i < static_cast<int>(default_block_order.size()); ++i) {
            ranked.push_back(
                {blockPriority(block_sizes[static_cast<size_t>(i)]), i,
                 default_block_order[static_cast<size_t>(i)]});
        }
        std::stable_sort(
            ranked.begin(), ranked.end(), [](const RankedBlock& a, const RankedBlock& b) {
                if (a.priority != b.priority) {
                    return a.priority < b.priority;
                }
                return a.order < b.order;
            });

        block_order->clear();
        block_order->reserve(ranked.size());
        for (const RankedBlock& item : ranked) {
            block_order->push_back(item.block_id);
        }
    };
}
#endif

}  // namespace

Estimator::Estimator(
    const tassel_tools::Parameters& params, std::shared_ptr<State> state,
    std::shared_ptr<FeatureManager> fm)
    : params_(params), state_(std::move(state)), feature_manager_(std::move(fm)) {
    if (!state_ || state_->max_frame_count != static_cast<int>(params_.max_frame_count) + 1) {
        throw std::invalid_argument(
            "Estimator state capacity must include one retained slot in addition to active frames");
    }
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    noise_ = initNoise();
    state_->visual_sqrt_info = Eigen::Matrix2d::Identity() * params_.visual_factor_weight;
    reset();
}

void Estimator::reset() {
    initialized_ = false;
    mode_ = EstimatorMode::Normal;
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
    schmidt_prior_covariance_.reset();
    schmidt_absorbed_feature_ids_.clear();
    last_measurement_was_keyframe_ = false;
    last_retained_keyframe_.reset();
    retained_rotation_.setIdentity();
    retained_position_.setZero();
    stage_time_totals_ = {};
    stage_time_samples_ = 0;
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    state_->reset();
    TASSEL_ASSERT(state_->max_frame_count >= 3);
    state_->latest_frame_index = kFirstActiveFrameIndex;
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay) {
    last_retained_keyframe_.reset();
    const bool record_stage_times = initialized_;
    const auto feature_start = StageClock::now();
    int& frame_count = state_->latest_frame_index;
    state_->frames[frame_count].timestamp_ns = frame_id;
    const double ts = tassel_utils::frameIdToSeconds(frame_id);
    state_->frames[frame_count].sync_delay = sync_delay;
    if (last_ts_ < 0 && !imu_measurements.empty()) {
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc - params_.acc_bias;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    const bool is_first_initial_frame =
        !initialized_ && frame_count == kFirstActiveFrameIndex &&
        state_->frames[kRetainedFrameIndex].timestamp_ns == tassel_utils::kInvalidFrameId;
    const bool is_keyframe = initialized_ || is_first_initial_frame
                                 ? feature_manager_->addFeatureFrame(frame_count, feature_frame)
                                 : feature_manager_->replaceInitializationCandidate(
                                       frame_count - 1, frame_count, feature_frame);
    state_->frames[frame_count].type = is_keyframe ? FrameType::KeyFrame : FrameType::NonKeyFrame;
    last_measurement_was_keyframe_ = is_keyframe;
    const auto feature_end = StageClock::now();

    const auto predict_start = feature_end;
    if (!is_first_initial_frame) {
        predictFrameState(frame_count, imu_measurements);
    }

    const double imu_query_timestamp = ts + sync_delay;
    interpolateBodyImu(
        imu_measurements, imu_query_timestamp, state_->frames[frame_count].gyro,
        state_->frames[frame_count].acc);
    const auto predict_end = StageClock::now();
    if (!initialized_) {
        if (!is_keyframe) {
            // 低视差图像只更新尾部候选槽，状态和 IMU 预积分持续累计，不占用新的窗口帧。
            return;
        }

        // 只有达到初始化视差的候选图像才固定为独立 VIO 状态并推进窗口。
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
            feature_manager_->removeFrameObservations(
                kFirstActiveFrameIndex, *state_, params_.ric, params_.tic);
            slideInitializationWindow();
            return;
        }
    }

    // 初始化成功后，同一状态机动作驱动优化、先验更新和窗口搬迁。
    const RetainedHostAction host_action = selectMarginalizationAction();
    const auto triangulation_start = StageClock::now();
    feature_manager_->triangulate(*state_, params_.ric, params_.tic);
    const auto triangulation_end = StageClock::now();

    const auto optimization_start = triangulation_end;
    optimize();
    const auto optimization_end = StageClock::now();
    const Sophus::SE3d optimized_pose(state_->frames[frame_count].R, state_->frames[frame_count].P);
    if (pose_callback_) {
        pose_callback_(ts, optimized_pose);
    }

    const auto outlier_start = StageClock::now();
    feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
    const auto outlier_end = StageClock::now();

    const auto marginalization_start = outlier_end;
    updateMarginalizationPrior(host_action);
    const auto marginalization_end = StageClock::now();

    const auto migration_start = marginalization_end;
    if (host_action != RetainedHostAction::MarginalizeOldestFrame) {
        const FrameState& retained_source = state_->frames[kFirstActiveFrameIndex];
        if (retained_source.timestamp_ns == tassel_utils::kInvalidFrameId) {
            throw std::logic_error("Cannot publish an invalid retained keyframe");
        }
        last_retained_keyframe_ = RetainedKeyframe{
            retained_source.timestamp_ns,
            Sophus::SE3d(retained_source.R, retained_source.P),
            feature_manager_->exportObservedLandmarks(
                kFirstActiveFrameIndex, *state_, params_.ric, params_.tic),
        };
    }
    migrateMarginalizedData(host_action);
    const auto migration_end = StageClock::now();

    if (record_stage_times) {
        recordStageTimes({
            elapsedMilliseconds(feature_start, feature_end),
            elapsedMilliseconds(predict_start, predict_end),
            elapsedMilliseconds(triangulation_start, triangulation_end),
            elapsedMilliseconds(optimization_start, optimization_end),
            elapsedMilliseconds(outlier_start, outlier_end),
            elapsedMilliseconds(marginalization_start, marginalization_end),
            elapsedMilliseconds(migration_start, migration_end),
        });
    }
}

void Estimator::recordStageTimes(const StageTimes& times) {
    stage_time_totals_.feature_ms += times.feature_ms;
    stage_time_totals_.predict_ms += times.predict_ms;
    stage_time_totals_.triangulation_ms += times.triangulation_ms;
    stage_time_totals_.optimization_ms += times.optimization_ms;
    stage_time_totals_.outlier_ms += times.outlier_ms;
    stage_time_totals_.marginalization_ms += times.marginalization_ms;
    stage_time_totals_.migration_ms += times.migration_ms;
    ++stage_time_samples_;
    if (stage_time_samples_ < kStageTimingLogInterval) {
        return;
    }

    const double scale = 1.0 / static_cast<double>(stage_time_samples_);
    const double feature_ms = stage_time_totals_.feature_ms * scale;
    const double predict_ms = stage_time_totals_.predict_ms * scale;
    const double triangulation_ms = stage_time_totals_.triangulation_ms * scale;
    const double optimization_ms = stage_time_totals_.optimization_ms * scale;
    const double outlier_ms = stage_time_totals_.outlier_ms * scale;
    const double marginalization_ms = stage_time_totals_.marginalization_ms * scale;
    const double migration_ms = stage_time_totals_.migration_ms * scale;
    const double non_optimization_ms =
        feature_ms + predict_ms + triangulation_ms + outlier_ms + marginalization_ms + migration_ms;
    spdlog::info(
        "Estimator stages: samples={} feature={:.3f}ms predict={:.3f}ms "
        "triangulation={:.3f}ms optimization={:.3f}ms outlier={:.3f}ms "
        "marginalization={:.3f}ms migration={:.3f}ms non_optimization={:.3f}ms "
        "total={:.3f}ms",
        stage_time_samples_, feature_ms, predict_ms, triangulation_ms, optimization_ms, outlier_ms,
        marginalization_ms, migration_ms, non_optimization_ms,
        non_optimization_ms + optimization_ms);
    stage_time_totals_ = {};
    stage_time_samples_ = 0;
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
            if (!preintegrator.buffer.empty() &&
                calibrated_imu.timestamp == preintegrator.buffer.back().timestamp) {
                const auto& boundary = preintegrator.buffer.back();
                if (!calibrated_imu.acc.isApprox(boundary.acc, 1e-12) ||
                    !calibrated_imu.gyro.isApprox(boundary.gyro, 1e-12)) {
                    throw std::runtime_error("Shared IMU boundary has inconsistent measurements");
                }
                // 初始化候选槽累计多个同步包时，相邻包会共享同一个边界样本。
                continue;
            }
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

std::optional<TrackingPredictionSnapshot> Estimator::makeTrackingPredictionSnapshot() const {
    if (!initialized_ || last_ts_ < 0.0 || !camera_) {
        return std::nullopt;
    }

    int source_index = -1;
    tassel_utils::FrameId newest_frame_id = tassel_utils::kInvalidFrameId;
    for (int i = 0; i <= state_->latest_frame_index; ++i) {
        const tassel_utils::FrameId frame_id = state_->frames[i].timestamp_ns;
        if (frame_id != tassel_utils::kInvalidFrameId &&
            (source_index < 0 || frame_id > newest_frame_id)) {
            source_index = i;
            newest_frame_id = frame_id;
        }
    }
    if (source_index < 0) {
        throw std::logic_error("Initialized estimator has no valid frame for tracking prediction");
    }

    TrackingPredictionSnapshot snapshot;
    snapshot.source_frame_id = newest_frame_id;
    snapshot.source_state = state_->frames[source_index];
    snapshot.imu_timestamp = last_ts_;
    snapshot.imu_acc = last_imu_acc_;
    snapshot.imu_gyro = last_imu_gyro_;
    snapshot.delay_time = state_->delay_time;
    snapshot.world_landmarks = feature_manager_->exportObservedWorldLandmarks(
        source_index, *state_, params_.ric, params_.tic);
    return snapshot;
}

std::unordered_map<int, cv::Point2f> predictLandmarkPixelsFromSnapshot(
    const TrackingPredictionSnapshot& snapshot, tassel_utils::FrameId target_frame_id,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay,
    const CameraBase& camera, const tassel_tools::Parameters& params) {
    std::unordered_map<int, cv::Point2f> pixels;
    if (snapshot.world_landmarks.empty()) {
        return pixels;
    }
    if (target_frame_id <= snapshot.source_frame_id || !std::isfinite(snapshot.imu_timestamp) ||
        !std::isfinite(sync_delay)) {
        throw std::invalid_argument("Invalid landmark prediction input");
    }

    FrameState target = snapshot.source_state;
    target.timestamp_ns = target_frame_id;
    target.sync_delay = sync_delay;
    double last_timestamp = snapshot.imu_timestamp;
    Eigen::Vector3d last_acc = snapshot.imu_acc;
    Eigen::Vector3d last_gyro = snapshot.imu_gyro;
    // 当前包首样本与快照边界重合，后续样本必须连续覆盖到目标图像同步时刻。
    for (const auto& raw_imu : imu_measurements) {
        if (raw_imu.timestamp < last_timestamp) {
            throw std::logic_error("Tracking prediction IMU is not monotonic");
        }
        if (raw_imu.timestamp == last_timestamp) {
            continue;
        }
        tassel_utils::IMUMeasurement imu = raw_imu;
        imu.acc -= params.acc_bias;
        const double dt = imu.timestamp - last_timestamp;
        const Eigen::Vector3d previous_acceleration =
            target.R * (last_acc - target.Ba) - tassel_utils::G;
        const Eigen::Vector3d angular_velocity = 0.5 * (last_gyro + imu.gyro) - target.Bg;
        target.R *= Sophus::SO3d::exp(angular_velocity * dt).matrix();
        target.R = Eigen::Quaterniond(target.R).normalized().toRotationMatrix();
        const Eigen::Vector3d current_acceleration =
            target.R * (imu.acc - target.Ba) - tassel_utils::G;
        const Eigen::Vector3d average_acceleration =
            0.5 * (previous_acceleration + current_acceleration);
        target.P += target.V * dt + 0.5 * average_acceleration * dt * dt;
        target.V += average_acceleration * dt;
        last_timestamp = imu.timestamp;
        last_acc = imu.acc;
        last_gyro = imu.gyro;
    }

    const double target_time = tassel_utils::frameIdToSeconds(target_frame_id) + sync_delay;
    if (std::abs(last_timestamp - target_time) > 1e-6) {
        throw std::logic_error("Tracking prediction IMU does not reach the target image time");
    }
    interpolateBodyImu(imu_measurements, target_time, target.gyro, target.acc);

    pixels.reserve(snapshot.world_landmarks.size());
    for (const auto& [feature_id, world_point] : snapshot.world_landmarks) {
        Eigen::Vector3d target_point;
        if (!worldPointToTargetCamera(
                target, world_point, sync_delay, snapshot.delay_time, params.ric, params.tic,
                target_point)) {
            continue;
        }
        const Eigen::Vector2d pixel = camera.distort(target_point.head<2>() / target_point.z());
        if (!pixel.allFinite() || pixel.x() < 0.0 || pixel.x() >= camera.get_width() ||
            pixel.y() < 0.0 || pixel.y() >= camera.get_height()) {
            continue;
        }
        pixels.emplace(feature_id, cv::Point2f(pixel.x(), pixel.y()));
    }
    return pixels;
}

std::unordered_map<int, cv::Point2f> Estimator::predictLandmarkPixels(
    const TrackingPredictionSnapshot& snapshot, tassel_utils::FrameId target_frame_id,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay) const {
    if (!camera_) {
        throw std::logic_error("Cannot predict landmarks without a camera");
    }
    return predictLandmarkPixelsFromSnapshot(
        snapshot, target_frame_id, imu_measurements, sync_delay, *camera_, params_);
}

void Estimator::optimize() {
    profiling::VtuneProfileScope vtune_scope;
    const int latest_id = state_->latest_frame_index;
    const int gauge_frame_index =
        marginalization_prior_ ? kRetainedFrameIndex : kFirstActiveFrameIndex;
    state_->stateToParams();
    std::vector<Eigen::Vector3d> active_velocities;
    active_velocities.reserve(static_cast<size_t>(latest_id));
    for (int i = kFirstActiveFrameIndex; i <= latest_id; ++i) {
        active_velocities.push_back(state_->frames[i].V);
    }
    bool stationary_window = !active_velocities.empty();
    for (const Eigen::Vector3d& velocity : active_velocities) {
        stationary_window = stationary_window && velocity.norm() <= kStationarySpeed;
    }
    mode_ = stationary_window ? EstimatorMode::StationaryBaHold : EstimatorMode::Normal;
    auto features = feature_manager_->collectLandmarks();

    std::vector<VisualFrameCacheInput> visual_cache_inputs;
    visual_cache_inputs.reserve(static_cast<size_t>(latest_id + 1));
    for (int i = 0; i <= latest_id; ++i) {
        const auto& frame = state_->frames[i];
        VisualFrameCacheInput input;
        input.pose = frame.pose.data();
        input.velocity = Eigen::Map<const Eigen::Vector3d>(frame.speed_bias.data());
        input.accel_bias = Eigen::Map<const Eigen::Vector3d>(frame.speed_bias.data() + 3);
        input.gyro_bias = Eigen::Map<const Eigen::Vector3d>(frame.speed_bias.data() + 6);
        input.gyro = frame.gyro;
        input.acceleration = frame.acc;
        input.sync_delay = frame.sync_delay;
        visual_cache_inputs.push_back(input);
    }
    VisualFrameCache visual_frame_cache(
        std::move(visual_cache_inputs), &state_->param_delay_time, params_.ric, params_.tic);
    ceres::Problem::Options problem_options;
    problem_options.context = ceres_context_.get();
    problem_options.evaluation_callback = &visual_frame_cache;
    ceres::Problem problem(problem_options);
    std::vector<int> visual_factors_per_frame(state_->latest_frame_index + 1, 0);

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->frames[i].pose.data(), 6, se3_manifold);
        if (i != kRetainedFrameIndex || !marginalization_prior_) {
            // 静止窗口只从切空间移除 ba；残差仍使用当前 ba，速度和 bg 继续更新。
            ceres::Manifold* speed_bias_manifold = nullptr;
            if (holdsBa()) {
                speed_bias_manifold = new ceres::SubsetManifold(9, {3, 4, 5});
            }
            problem.AddParameterBlock(state_->frames[i].speed_bias.data(), 9, speed_bias_manifold);
        }
    }
    if (!marginalization_prior_) {
        // 初始化期间 frame0 为空，不参与任何物理因子。
        problem.SetParameterBlockConstant(state_->frames[kRetainedFrameIndex].speed_bias.data());
        problem.SetParameterBlockConstant(state_->frames[kRetainedFrameIndex].pose.data());
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
        prior_blocks.push_back(state_->frames[0].pose.data());
        for (int i = 1; i < num_kept; ++i) {
            prior_blocks.push_back(state_->frames[i].pose.data());
            prior_blocks.push_back(state_->frames[i].speed_bias.data());
        }
        prior_blocks.push_back(&state_->param_delay_time);
        problem.AddResidualBlock(prior_cost, nullptr, prior_blocks);
    }

    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;
    ceres::LossFunction* loss = new ceres::HuberLoss(visual_huber_delta);
    std::vector<double> inv_depth_params(features.size());
    std::vector<int> visual_landmark_cache_indices(features.size());
    visual_frame_cache.reserveLandmarks(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        const double depth = features[k]->estimated_depth;
        TASSEL_ASSERT(std::isfinite(depth) && depth > 1e-12);
        inv_depth_params[k] = 1.0 / depth;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
        visual_landmark_cache_indices[k] =
            visual_frame_cache.addLandmark(features[k]->observations[0].uv, &inv_depth_params[k]);
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
            // sync_delay 是帧级采样属性；若同帧观测不一致，按帧复用时间补偿将不再成立。
            if (f->observations[0].sync_delay != state_->frames[host_id].sync_delay ||
                f->observations[obs_idx].sync_delay != state_->frames[target_id].sync_delay) {
                throw std::logic_error("Feature sync delay does not match its frame state");
            }
            visual_frame_cache.addPair(host_id, target_id);
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
                f->observations[0].sync_delay, f->observations[obs_idx].sync_delay,
                &visual_frame_cache, host_id, target_id, visual_landmark_cache_indices[k]);
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
#if defined(CERES_HAS_SCHUR_STRUCTURE_HINTS)
    // VIO视觉残差布局固定为2维像素残差、1维逆深度消元块；其余状态块尺寸可变。
    opts.schur_structure_row_block_size = 2;
    opts.schur_structure_e_block_size = 1;
    opts.schur_structure_f_block_size = -1;
#endif
    opts.trust_region_strategy_type = ceresTrustRegionStrategy(params_.trust_region_strategy);
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    // 每个视觉因子只包含一个逆深度，组0路标必须保持为 Hessian 图的独立集。
    for (double& inverse_depth : inv_depth_params) {
        ordering->AddElementToGroup(&inverse_depth, 0);
    }
    int layout_group = 1;
    for (int i = 0; i < state_->max_frame_count; ++i) {
        ordering->AddElementToGroup(state_->frames[i].pose.data(), layout_group++);
        if (i != kRetainedFrameIndex || !marginalization_prior_) {
            ordering->AddElementToGroup(state_->frames[i].speed_bias.data(), layout_group++);
        }
    }
    ordering->AddElementToGroup(&state_->param_delay_time, layout_group++);
    opts.linear_solver_ordering = std::move(ordering);
#if defined(CERES_HAS_SCHUR_LAYOUT_CALLBACK)
    opts.schur_layout_callback = makeSchurLayoutCallback();
#endif
    opts.max_num_iterations = params_.num_iterations;
    if (params_.max_solver_time > 0.0) {
        opts.max_solver_time_in_seconds = params_.max_solver_time;
    }
    // 滑窗 BA 规模较小，固定单线程避免线程调度和同步成本超过并行收益。
    opts.num_threads = 1;
    opts.minimizer_progress_to_stdout = false;
    opts.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    spdlog::info(
        "Ceres summary: threads={}/{} parameter_blocks={}/{} residual_blocks={}/{} "
        "residuals={}/{} iterations={} "
        "successful={} rejected={} residual={:.3f}ms jacobian={:.3f}ms "
        "linear_solver={:.3f}ms schur_setup={:.3f}ms schur_eliminate={:.3f}ms "
        "schur_reduced={:.3f}ms schur_back_substitute={:.3f}ms "
        "schur_diagonal={:.3f}ms schur_chunks={:.3f}ms schur_no_e={:.3f}ms "
        "preprocessor={:.3f}ms total={:.3f}ms",
        summary.num_threads_used, summary.num_threads_given, summary.num_parameter_blocks,
        summary.num_parameter_blocks_reduced, summary.num_residual_blocks,
        summary.num_residual_blocks_reduced, summary.num_residuals, summary.num_residuals_reduced,
        summary.iterations.size(), summary.num_successful_steps, summary.num_unsuccessful_steps,
        1000.0 * summary.residual_evaluation_time_in_seconds,
        1000.0 * summary.jacobian_evaluation_time_in_seconds,
        1000.0 * summary.linear_solver_time_in_seconds,
        1000.0 * summary.schur_setup_time_in_seconds,
        1000.0 * summary.schur_elimination_time_in_seconds,
        1000.0 * summary.schur_reduced_solve_time_in_seconds,
        1000.0 * summary.schur_back_substitution_time_in_seconds,
        1000.0 * summary.schur_diagonal_time_in_seconds,
        1000.0 * summary.schur_chunks_time_in_seconds, 1000.0 * summary.schur_no_e_time_in_seconds,
        1000.0 * summary.preprocessor_time_in_seconds, 1000.0 * summary.total_time_in_seconds);
    bool finite_solution = std::isfinite(state_->param_delay_time);
    for (int i = 0; i <= latest_id && finite_solution; ++i) {
        finite_solution =
            allFinite(state_->frames[i].pose) && allFinite(state_->frames[i].speed_bias);
    }
    finite_solution = finite_solution && allFinite(inv_depth_params);
    if (!summary.IsSolutionUsable() || !finite_solution) {
        spdlog::error("Optimization rejected: {}", summary.BriefReport());
        state_->stateToParams();
        throw std::runtime_error("VIO optimization failed");
    }

    if (visual_factor_callback_) {
        visual_factor_callback_(
            tassel_utils::frameIdToSeconds(state_->frames[latest_id].timestamp_ns),
            visual_factors_per_frame);
    }

    state_->paramsToState();
    if (marginalization_prior_) {
        const int num_kept = static_cast<int>(marginalization_prior_->linearization_poses.size());
        std::vector<std::array<double, 6>> current_poses(num_kept);
        std::vector<std::array<double, 9>> current_speed_bias(num_kept);
        for (int i = 0; i < num_kept; ++i) {
            current_poses[i] = state_->frames[i].pose;
            current_speed_bias[i] = state_->frames[i].speed_bias;
        }
        if (schmidt_prior_covariance_) {
            recenterPriorCovariance(
                *schmidt_prior_covariance_, *marginalization_prior_, current_poses);
        }
        MargHelper::recenterPrior(
            *marginalization_prior_, current_poses, current_speed_bias, state_->param_delay_time);
    }
    restoreGauge(gauge_frame_index);

    if (spdlog::should_log(spdlog::level::info)) {
        const FrameState& final_frame = state_->frames[latest_id];
        spdlog::info(
            "Optimization\n"
            "  Ba: ({:.5f}, {:.5f}, {:.5f})\n"
            "  Bg: ({:.5f}, {:.5f}, {:.5f})\n"
            "  delay: {:.6f}",
            final_frame.Ba.x(), final_frame.Ba.y(), final_frame.Ba.z(), final_frame.Bg.x(),
            final_frame.Bg.y(), final_frame.Bg.z(), state_->delay_time);
    }

    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : Feature::InvalidDepth;
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

RetainedHostAction Estimator::selectMarginalizationAction() const {
    if (!initialized_) {
        throw std::logic_error("Marginalization action requested before VIO initialization");
    }
    if (!marginalization_prior_) {
        if (state_->frames[kRetainedFrameIndex].timestamp_ns != tassel_utils::kInvalidFrameId) {
            throw std::logic_error("Retained slot initialization was requested more than once");
        }
        return RetainedHostAction::InitializeRetainedSlot;
    }
    if (state_->frames[kRetainedFrameIndex].timestamp_ns == tassel_utils::kInvalidFrameId) {
        throw std::logic_error("Marginalization prior has an invalid retained slot");
    }
    return state_->frames[kFirstActiveFrameIndex].type == FrameType::KeyFrame
               ? RetainedHostAction::ReplaceRetainedSlot
               : RetainedHostAction::MarginalizeOldestFrame;
}

void Estimator::updateMarginalizationPrior(RetainedHostAction action) {
    profiling::VtuneProfileScope vtune_scope;
    const int window_capacity = state_->max_frame_count;
    TASSEL_ASSERT(window_capacity >= 3);
    state_->stateToParams();
    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;

    const bool initializes_retained_slot = action == RetainedHostAction::InitializeRetainedSlot;
    if (initializes_retained_slot) {
        if (marginalization_prior_ ||
            state_->frames[kRetainedFrameIndex].timestamp_ns != tassel_utils::kInvalidFrameId) {
            throw std::logic_error("Retained slot can only be initialized once");
        }
    } else if (!marginalization_prior_) {
        throw std::logic_error("Marginalization action requires an initialized retained slot");
    }
    const MargLinData* prior_to_linearize = marginalization_prior_.get();
    if (prior_to_linearize) {
        const int num_kept = static_cast<int>(marginalization_prior_->linearization_poses.size());
        TASSEL_ASSERT(num_kept == window_capacity - 1);
    }

    // 常规边缘化消去 frame1 宿主的所有后续观测；其他动作只消去 frame0 到 frame1 的约束。
    const int landmark_host_frame_index = action == RetainedHostAction::MarginalizeOldestFrame
                                              ? kFirstActiveFrameIndex
                                              : kRetainedFrameIndex;
    const int landmark_target_frame_index =
        action == RetainedHostAction::MarginalizeOldestFrame ? -1 : kFirstActiveFrameIndex;
    auto retiring_features = feature_manager_->collectMarginalizedFeatures(
        landmark_host_frame_index, landmark_target_frame_index);
    std::vector<Feature*> schmidt_visual_features;
    std::vector<int> schmidt_visual_feature_ids;
    if (maintainsSchmidtCovariance() && schmidt_prior_covariance_) {
        auto& all_features = feature_manager_->features();
        for (auto it = schmidt_absorbed_feature_ids_.begin();
             it != schmidt_absorbed_feature_ids_.end();) {
            if (all_features.find(*it) == all_features.end()) {
                it = schmidt_absorbed_feature_ids_.erase(it);
            } else {
                ++it;
            }
        }
        std::unordered_map<const Feature*, int> feature_ids;
        feature_ids.reserve(all_features.size());
        for (auto& [feature_id, feature] : all_features) {
            feature_ids.emplace(&feature, feature_id);
        }
        for (Feature* feature : retiring_features) {
            const auto id = feature_ids.find(feature);
            if (id == feature_ids.end()) {
                throw std::logic_error("Retiring feature ID mapping is incomplete");
            }
            if (schmidt_absorbed_feature_ids_.find(id->second) ==
                schmidt_absorbed_feature_ids_.end()) {
                schmidt_visual_features.push_back(feature);
                schmidt_visual_feature_ids.push_back(id->second);
            }
        }
    }
    visitPreintegrators([&](auto& preintegrators) {
        using Integrator = typename std::decay_t<decltype(preintegrators)>::value_type;
        std::vector<IntegratorBase<Integrator>*> imu_preintegrators;
        const int first_imu_index = 1;
        const int imu_factor_count = 1;
        for (int i = 0; i < imu_factor_count; ++i) {
            const int imu_index = first_imu_index + i;
            if (preintegrators[imu_index].buffer.size() < 2) {
                throw std::logic_error(
                    "Marginalization IMU interval has fewer than two measurements at index " +
                    std::to_string(imu_index));
            }
            imu_preintegrators.push_back(&preintegrators[imu_index]);
        }

        auto linearizer = MarginalizationSqrt<Integrator>(
            std::move(retiring_features), landmark_target_frame_index,
            std::make_unique<ceres::HuberLoss>(visual_huber_delta), state_, imu_preintegrators,
            params_.ric, params_.tic, prior_to_linearize, first_imu_index);
        linearizer.allocate();
        linearizer.linearize();
        linearizer.marginalizeLandmarks();

        Eigen::MatrixXd reduced_jacobian;
        Eigen::VectorXd reduced_residual;
        linearizer.buildReducedSystem(reduced_jacobian, reduced_residual);

        std::unique_ptr<SchmidtPriorCovariance> updated_schmidt_covariance;
        if (maintainsSchmidtCovariance() && schmidt_prior_covariance_) {
            if (action == RetainedHostAction::InitializeRetainedSlot) {
                throw std::logic_error(
                    "Initialized retained slot cannot have an old Schmidt prior");
            }
            Eigen::MatrixXd visual_jacobian;
            Eigen::VectorXd visual_residual;
            linearizer.buildReducedVisualSystem(
                schmidt_visual_features, visual_jacobian, visual_residual);

            const int latest_imu_index = window_capacity - 2;
            const int parent_index = latest_imu_index;
            const int child_index = parent_index + 1;
            updated_schmidt_covariance =
                std::make_unique<SchmidtPriorCovariance>(propagateAndUpdateSchmidtWindow(
                    *schmidt_prior_covariance_, &preintegrators[latest_imu_index],
                    state_->frames[parent_index], state_->frames[child_index], visual_jacobian,
                    visual_residual, window_capacity, action));
            schmidt_absorbed_feature_ids_.insert(
                schmidt_visual_feature_ids.begin(), schmidt_visual_feature_ids.end());
        }

        // QR 前的列顺序为 [state0(15), state1(15), ..., delay]。
        // Initialize 消去空 state0 和首帧运动状态；其他动作按状态机重排列。
        constexpr int host_pose_size = MargHelper::kPoseSize;
        constexpr int host_speed_bias_size = MargHelper::kSpeedBiasSize;
        constexpr int full_state_size = MargHelper::kFullStateSize;
        const int marginalized_size = action == RetainedHostAction::MarginalizeOldestFrame
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
        if (action == RetainedHostAction::MarginalizeOldestFrame) {
            // 索引 0 已是保留宿主帧，其速度/偏置列保持结构零。
            prior_jacobian = std::move(compact_prior_jacobian);
        } else {
            // 新的保留宿主帧将仅含位姿的紧凑块扩展为 PriorFactor 的统一 15 维接口。
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

        // 保留宿主的速度和偏置只作为运动补偿快照，不作为先验优化变量。
        Eigen::MatrixXd pose_only_host_prior(
            prior_jacobian.rows(), host_pose_size + (window_capacity - 2) * full_state_size + 1);
        pose_only_host_prior.leftCols(host_pose_size) = prior_jacobian.leftCols(host_pose_size);
        pose_only_host_prior.middleCols(host_pose_size, (window_capacity - 2) * full_state_size) =
            prior_jacobian.middleCols(full_state_size, (window_capacity - 2) * full_state_size);
        pose_only_host_prior.col(pose_only_host_prior.cols() - 1) =
            prior_jacobian.col(prior_jacobian.cols() - 1);

        auto updated_prior = std::make_unique<MargLinData>();
        updated_prior->H = std::move(pose_only_host_prior);
        updated_prior->b = std::move(prior_residual);
        updated_prior->linearization_poses.resize(window_capacity - 1);
        updated_prior->linearization_speed_bias.resize(window_capacity - 1);
        updated_prior->linearization_delay_time = state_->param_delay_time;
        const int retained_host_source_index = action == RetainedHostAction::MarginalizeOldestFrame
                                                   ? kRetainedFrameIndex
                                                   : kFirstActiveFrameIndex;
        updated_prior->linearization_poses[0] = state_->frames[retained_host_source_index].pose;
        updated_prior->linearization_speed_bias[0] =
            state_->frames[retained_host_source_index].speed_bias;
        for (int i = 2; i < window_capacity; ++i) {
            updated_prior->linearization_poses[i - 1] = state_->frames[i].pose;
            updated_prior->linearization_speed_bias[i - 1] = state_->frames[i].speed_bias;
        }
        if (!updated_schmidt_covariance && maintainsSchmidtCovariance()) {
            std::vector<Feature*> full_features = feature_manager_->collectLandmarks();
            const std::optional<SchmidtPriorCovariance> current_covariance =
                tryBuildGaugeFixedWindowPosteriorCovariance(
                    full_features, state_, preintegrators, 1, prior_to_linearize, params_.ric,
                    params_.tic, visual_huber_delta);
            if (current_covariance) {
                updated_schmidt_covariance = std::make_unique<SchmidtPriorCovariance>(
                    retainMarginalizedPriorCovariance(*current_covariance, action));
                schmidt_absorbed_feature_ids_.clear();
                const std::unordered_set<const Feature*> full_feature_set(
                    full_features.begin(), full_features.end());
                for (const auto& [feature_id, feature] : feature_manager_->features()) {
                    if (full_feature_set.find(&feature) != full_feature_set.end()) {
                        schmidt_absorbed_feature_ids_.insert(feature_id);
                    }
                }
                spdlog::info(
                    "Schmidt covariance initialized from full posterior: states={} features={}",
                    current_covariance->state_count, full_features.size());
            } else {
                spdlog::warn(
                    "Schmidt covariance unavailable: full window remains deficient after gauge "
                    "fixing");
            }
        }
        if (!maintainsSchmidtCovariance()) {
            updated_schmidt_covariance.reset();
            schmidt_absorbed_feature_ids_.clear();
        }
        marginalization_prior_ = std::move(updated_prior);
        schmidt_prior_covariance_ = std::move(updated_schmidt_covariance);
    });
}

void Estimator::slideInitializationWindow() {
    TASSEL_ASSERT(!marginalization_prior_);
    const int n = state_->max_frame_count;
    for (int i = kFirstActiveFrameIndex; i < n - 1; ++i) {
        state_->copyFrameState(i + 1, i);
    }
    state_->frames[n - 1].timestamp_ns = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
        for (int i = kFirstActiveFrameIndex; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(state_->frames[n - 2].Ba, state_->frames[n - 2].Bg, noise_);
    });
}

void Estimator::migrateMarginalizedData(RetainedHostAction action) {
    const int window_capacity = state_->max_frame_count;
    constexpr int first_movable_index = 1;
    TASSEL_ASSERT(marginalization_prior_);

    switch (action) {
        case RetainedHostAction::InitializeRetainedSlot:
            // frame0 在初始化前为空，只移除其占位索引。
            feature_manager_->removeFrameObservations(0, *state_, params_.ric, params_.tic);
            break;
        case RetainedHostAction::ReplaceRetainedSlot:
            feature_manager_->replaceRetainedHost(0, 1, *state_, params_.ric, params_.tic);
            break;
        case RetainedHostAction::MarginalizeOldestFrame:
            feature_manager_->removeFrameObservations(1, *state_, params_.ric, params_.tic);
            break;
    }

    if (action != RetainedHostAction::MarginalizeOldestFrame) {
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

    if (action != RetainedHostAction::MarginalizeOldestFrame) {
        captureGauge(kRetainedFrameIndex);
    }
}

void Estimator::captureGauge(int frame_index) {
    if (frame_index < 0 || frame_index > state_->latest_frame_index) {
        throw std::out_of_range("Gauge frame is outside the active window");
    }
    const FrameState& frame = state_->frames[frame_index];
    if (frame.timestamp_ns == tassel_utils::kInvalidFrameId || !frame.R.allFinite() ||
        !frame.P.allFinite()) {
        throw std::logic_error("Cannot capture gauge from an invalid frame");
    }
    retained_rotation_ = frame.R;
    retained_position_ = frame.P;
}

void Estimator::restoreGauge(int reference_frame_index) {
    if (!initialized_) {
        throw std::logic_error("Gauge restore requested before VIO initialization");
    }
    if (reference_frame_index < 0 || reference_frame_index > state_->latest_frame_index) {
        throw std::out_of_range("Gauge reference frame is outside the active window");
    }
    if (!retained_position_.allFinite()) {
        throw std::logic_error("Gauge anchor position is not finite");
    }

    const FrameState& optimized_reference = state_->frames[reference_frame_index];
    if (!optimized_reference.P.allFinite()) {
        throw std::logic_error("Gauge reference position is not finite");
    }
    const double yaw_correction =
        rotationYaw(retained_rotation_) - rotationYaw(optimized_reference.R);
    const Eigen::Matrix3d rotation_correction =
        Eigen::AngleAxisd(yaw_correction, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d optimized_reference_position = optimized_reference.P;
    const Eigen::Vector3d gauge_translation =
        retained_position_ - rotation_correction * optimized_reference_position;
    for (int frame_index = reference_frame_index; frame_index <= state_->latest_frame_index;
         ++frame_index) {
        const FrameState& frame = state_->frames[frame_index];
        if (!frame.P.allFinite() || !frame.R.allFinite() || !frame.V.allFinite()) {
            throw std::logic_error("Gauge window contains a non-finite state");
        }
    }
    for (int frame_index = reference_frame_index; frame_index <= state_->latest_frame_index;
         ++frame_index) {
        FrameState& frame = state_->frames[frame_index];
        frame.P =
            rotation_correction * (frame.P - optimized_reference_position) + retained_position_;
        frame.R = rotation_correction * frame.R;
        frame.V = rotation_correction * frame.V;
    }
    if (marginalization_prior_) {
        MargHelper::transformPriorGauge(
            *marginalization_prior_, rotation_correction, gauge_translation);
        if (schmidt_prior_covariance_) {
            transformPriorCovarianceGauge(*schmidt_prior_covariance_, rotation_correction);
        }
    }
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
    const int last_frame_index = state_->latest_frame_index;
    const int n_frames = last_frame_index - kFirstActiveFrameIndex + 1;

    InitialSFM sfm(
        params_.sfm_min_correspondences, params_.sfm_min_e_inliers, params_.sfm_e_ransac_threshold,
        params_.sfm_min_correspondences, params_.sfm_pnp_reproj_threshold,
        params_.sfm_max_bad_pnp_ratio, params_.sfm_ba_max_iterations, params_.sfm_ba_num_threads);
    const bool sfm_succeeded =
        sfm.construct(*state_, *feature_manager_, params_.ric, Rs_, Ps_, kFirstActiveFrameIndex);
    if (!sfm_succeeded) {
        spdlog::info("VIO initialization: SFM failed");
        return false;
    }
    Vs_.assign(n_frames, Eigen::Vector3d::Zero());

    {
        std::vector<Eigen::Matrix3d> dq_dbgs, delta_qs;
        visitPreintegrators([&](const auto& preintegrators) {
            for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
                dq_dbgs.push_back(preintegrators[i].get_dq_dbg());
                delta_qs.push_back(preintegrators[i].final_delta_q);
            }
        });
        Eigen::Vector3d bg = solveGyroBias(Rs_, dq_dbgs, delta_qs, params_.ric);
        if (!bg.allFinite()) {
            spdlog::info("VIO initialization: gyro bias solve failed");
            return false;
        }
        for (int i = kFirstActiveFrameIndex; i <= last_frame_index; ++i) {
            state_->frames[i].Bg = bg;
        }
    }

    visitPreintegrators([&](auto& preintegrators) {
        for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
            preintegrators[i].repropagate(state_->frames[i].Ba, state_->frames[i].Bg, noise_);
        }
    });

    std::vector<Eigen::Vector3d> delta_ps, delta_vs;
    std::vector<double> dts;
    visitPreintegrators([&](const auto& preintegrators) {
        for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
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
    if (!std::isfinite(s) || s < params_.init_min_scale) {
        spdlog::info("VI initialization: degenerate or invalid scale {:.6f}", s);
        return false;
    }

    Eigen::Matrix3d R0 =
        Eigen::Quaterniond::FromTwoVectors((g).normalized(), Eigen::Vector3d(0, 0, 1))
            .toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;

    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    initialized_ = true;

    // SFM 数组的 local_index=0 对应活动窗口的 frame1，不包含保留槽 frame0。
    for (int local_index = 0; local_index < n_frames; ++local_index) {
        const int frame_index = kFirstActiveFrameIndex + local_index;
        state_->frames[frame_index].R =
            Eigen::Quaterniond(R0 * params_.ric * Rs_[local_index] * params_.ric.transpose())
                .normalized()
                .toRotationMatrix();
        state_->frames[frame_index].P =
            R0 *
            (params_.ric * s * Ps_[local_index] -
             params_.ric * Rs_[local_index] * params_.ric.transpose() * params_.tic + params_.tic);
        state_->frames[frame_index].V = R0 * Vs_[local_index];
    }
    // 首次优化以 frame1 为参考；首次滑窗后它会迁移为保留槽 frame0。
    captureGauge(kFirstActiveFrameIndex);

    spdlog::info(
        "VI init: |g|={:.4f} s={:.4f} R0_yaw={:.2f}°", tassel_utils::G.norm(), s,
        yaw * 180.0 / M_PI);
    return true;
}

}  // namespace tassel_core
