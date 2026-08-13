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
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cam/camera_base.h"
#include "estimator/window_optimizer.h"
#include "factor/integrator_base.h"
#include "frond_end/reprojection.h"
#include "imu_interpolation.h"
#include "marg/marg_helper.h"
#include "marg/marginalization_sqrt.h"
#include "tassel_utils/macros.h"

#include "initial/initial_alignment.h"
#include "initial/initial_sfm.h"

namespace tassel_core {

namespace {
constexpr int kRetainedFrameIndex = 0;
constexpr int kFirstActiveFrameIndex = 1;
constexpr double kStationarySpeed = 0.05;

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
    window_optimizer_ = std::make_unique<WindowOptimizer>(params_, state_);
    reset();
}

Estimator::~Estimator() = default;

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
    last_retained_keyframe_.reset();
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    state_->reset();
    TASSEL_ASSERT(state_->max_frame_count >= 3);
    state_->latest_active_frame_index = kFirstActiveFrameIndex;
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay) {
    last_retained_keyframe_.reset();
    int& frame_count = state_->latest_active_frame_index;
    state_->frames[frame_count].frame_id = frame_id;
    const double ts = tassel_utils::frameIdToSeconds(frame_id);
    state_->frames[frame_count].image_sync_delay = sync_delay;
    if (last_ts_ < 0 && !imu_measurements.empty()) {
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc - params_.acc_bias;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    const bool is_first_initial_frame =
        !initialized_ && frame_count == kFirstActiveFrameIndex &&
        state_->frames[kRetainedFrameIndex].frame_id == tassel_utils::kInvalidFrameId;
    const bool is_keyframe = initialized_ || is_first_initial_frame
                                 ? feature_manager_->addFeatureFrame(frame_count, feature_frame)
                                 : feature_manager_->replaceInitializationCandidate(
                                       frame_count - 1, frame_count, feature_frame);
    state_->frames[frame_count].frame_type =
        is_keyframe ? FrameType::KeyFrame : FrameType::NonKeyFrame;
    last_measurement_was_keyframe_ = is_keyframe;

    if (!is_first_initial_frame) {
        predictFrameState(frame_count, imu_measurements);
    }

    const double imu_query_timestamp = ts + sync_delay;
    interpolateBodyImu(
        imu_measurements, imu_query_timestamp, state_->frames[frame_count].imu_gyro,
        state_->frames[frame_count].imu_acc);
    if (!initialized_) {
        if (!is_keyframe) {
            // 低视差图像只更新尾部候选槽，状态和 IMU 预积分持续累计，不占用新的窗口帧。
            return;
        }

        // 只有达到初始化视差的候选图像才固定为独立 VIO 状态并推进窗口。
        if (frame_count < state_->max_frame_count - 1) {
            ++frame_count;
            state_->copyFrameState(frame_count - 1, frame_count);
            state_->frames[frame_count].frame_id = tassel_utils::kInvalidFrameId;
            int next_idx = frame_count - 1;
            visitPreintegrators([&](auto& preintegrators) {
                if (next_idx < static_cast<int>(preintegrators.size())) {
                    preintegrators[next_idx].reset(
                        state_->frames[frame_count - 1].accel_bias,
                        state_->frames[frame_count - 1].gyro_bias, noise_);
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

    runSlidingWindowUpdate(frame_count, ts);
}

void Estimator::runSlidingWindowUpdate(int latest_active_frame_index, double timestamp) {
    const RetainedHostAction action = selectMarginalizationAction();
    feature_manager_->triangulate(*state_, params_.ric, params_.tic);
    optimize();

    const Sophus::SE3d optimized_pose(
        state_->frames[latest_active_frame_index].rot_w_i,
        state_->frames[latest_active_frame_index].pos_w_i);
    if (pose_callback_) {
        pose_callback_(timestamp, optimized_pose);
    }

    // 离群特征必须在构建新先验前删除，避免其约束写入边缘化先验。
    feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
    updateMarginalizationPrior(action);

    if (action != RetainedHostAction::MarginalizeOldestFrame) {
        const FrameState& retained_source = state_->frames[kFirstActiveFrameIndex];
        if (retained_source.frame_id == tassel_utils::kInvalidFrameId) {
            throw std::logic_error("Cannot publish an invalid retained keyframe");
        }
        last_retained_keyframe_ = RetainedKeyframe{
            retained_source.frame_id,
            Sophus::SE3d(retained_source.rot_w_i, retained_source.pos_w_i),
            feature_manager_->exportObservedLandmarks(
                kFirstActiveFrameIndex, *state_, params_.ric, params_.tic),
        };
    }
    migrateMarginalizedData(action);
}

void Estimator::predictFrameState(
    int frame_index, const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    if (frame_index == 0) {
        return;
    }

    TASSEL_ASSERT(frame_index > 0 && frame_index <= state_->latest_active_frame_index);
    const FrameState& previous_frame = state_->frames[frame_index - 1];
    FrameState& predicted_frame = state_->frames[frame_index];
    TASSEL_ASSERT(previous_frame.frame_id != tassel_utils::kInvalidFrameId);
    TASSEL_ASSERT(predicted_frame.frame_id > previous_frame.frame_id);

    Eigen::Matrix3d rotation = predicted_frame.rot_w_i;
    Eigen::Vector3d position = predicted_frame.pos_w_i;
    Eigen::Vector3d velocity = predicted_frame.vel_w;
    const Eigen::Vector3d acc_bias = predicted_frame.accel_bias;
    const Eigen::Vector3d gyro_bias = predicted_frame.gyro_bias;

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

    predicted_frame.rot_w_i = Eigen::Quaterniond(rotation).normalized().toRotationMatrix();
    predicted_frame.pos_w_i = position;
    predicted_frame.vel_w = velocity;
}

std::optional<TrackingPredictionSnapshot> Estimator::makeTrackingPredictionSnapshot() const {
    if (!initialized_ || last_ts_ < 0.0 || !camera_) {
        return std::nullopt;
    }

    int source_index = -1;
    tassel_utils::FrameId newest_frame_id = tassel_utils::kInvalidFrameId;
    for (int i = 0; i <= state_->latest_active_frame_index; ++i) {
        const tassel_utils::FrameId frame_id = state_->frames[i].frame_id;
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
    snapshot.time_delay = state_->time_delay;
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
    target.frame_id = target_frame_id;
    target.image_sync_delay = sync_delay;
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
            target.rot_w_i * (last_acc - target.accel_bias) - tassel_utils::G;
        const Eigen::Vector3d angular_velocity = 0.5 * (last_gyro + imu.gyro) - target.gyro_bias;
        target.rot_w_i *= Sophus::SO3d::exp(angular_velocity * dt).matrix();
        target.rot_w_i = Eigen::Quaterniond(target.rot_w_i).normalized().toRotationMatrix();
        const Eigen::Vector3d current_acceleration =
            target.rot_w_i * (imu.acc - target.accel_bias) - tassel_utils::G;
        const Eigen::Vector3d average_acceleration =
            0.5 * (previous_acceleration + current_acceleration);
        target.pos_w_i += target.vel_w * dt + 0.5 * average_acceleration * dt * dt;
        target.vel_w += average_acceleration * dt;
        last_timestamp = imu.timestamp;
        last_acc = imu.acc;
        last_gyro = imu.gyro;
    }

    const double target_time = tassel_utils::frameIdToSeconds(target_frame_id) + sync_delay;
    if (std::abs(last_timestamp - target_time) > 1e-6) {
        throw std::logic_error("Tracking prediction IMU does not reach the target image time");
    }
    interpolateBodyImu(imu_measurements, target_time, target.imu_gyro, target.imu_acc);

    pixels.reserve(snapshot.world_landmarks.size());
    for (const auto& [feature_id, world_point] : snapshot.world_landmarks) {
        Eigen::Vector3d target_point;
        if (!worldPointToTargetCamera(
                target, world_point, sync_delay, snapshot.time_delay, params.ric, params.tic,
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
    const int latest_id = state_->latest_active_frame_index;
    const int gauge_frame_index =
        marginalization_prior_ ? kRetainedFrameIndex : kFirstActiveFrameIndex;
    const bool hold_accel_bias = isStationaryWindow();
    state_->stateToParams();
    auto features = feature_manager_->collectLandmarks();
    WindowOptimizationResult result;
    try {
        result = visitPreintegrators([&](auto& preintegrators) {
            return window_optimizer_->solve(
                features, preintegrators, marginalization_prior_.get(), hold_accel_bias);
        });
    } catch (...) {
        state_->stateToParams();
        throw;
    }

    if (visual_factor_callback_) {
        visual_factor_callback_(
            tassel_utils::frameIdToSeconds(state_->frames[latest_id].frame_id),
            result.visual_factors_per_frame);
    }

    // 先验先迁移到优化状态，再与窗口状态一起执行同一 gauge 变换；顺序不可交换。
    state_->paramsToState();
    if (marginalization_prior_) {
        const int num_kept = static_cast<int>(marginalization_prior_->linearization_poses.size());
        std::vector<std::array<double, 6>> current_poses(num_kept);
        std::vector<std::array<double, 9>> current_speed_bias(num_kept);
        for (int i = 0; i < num_kept; ++i) {
            current_poses[i] = state_->frames[i].param_pose;
            current_speed_bias[i] = state_->frames[i].param_speed_bias;
        }
        MargHelper::recenterPrior(
            *marginalization_prior_, current_poses, current_speed_bias, state_->param_time_delay);
    }
    if (state_->gauge_reference) {
        normalizeGaugeAfterOptimization(gauge_frame_index);
    } else {
        state_->captureGauge(gauge_frame_index);
    }

    feature_manager_->updateFeatureDepths(result.feature_depths);

    visitPreintegrators([&](auto& preintegrators) {
        const int first_imu_index = marginalization_prior_ ? 1 : 0;
        for (int i = first_imu_index; i < state_->latest_active_frame_index; ++i) {
            const double delta_ba =
                (state_->frames[i].accel_bias - preintegrators[i].ba_linearized).norm();
            const double delta_bg =
                (state_->frames[i].gyro_bias - preintegrators[i].bg_linearized).norm();
            if (delta_ba > params_.imu_repropagate_ba_threshold ||
                delta_bg > params_.imu_repropagate_bg_threshold) {
                preintegrators[i].repropagate(
                    state_->frames[i].accel_bias, state_->frames[i].gyro_bias, noise_);
            }
        }
    });
}

bool Estimator::isStationaryWindow() const {
    if (state_->latest_active_frame_index < kFirstActiveFrameIndex) {
        throw std::logic_error("Stationary detection requires an active window");
    }
    for (int i = kFirstActiveFrameIndex; i <= state_->latest_active_frame_index; ++i) {
        if (!state_->frames[i].vel_w.allFinite()) {
            throw std::logic_error("Stationary detection encountered a non-finite velocity");
        }
        if (state_->frames[i].vel_w.norm() > kStationarySpeed) {
            return false;
        }
    }
    return true;
}

RetainedHostAction Estimator::selectMarginalizationAction() const {
    if (!initialized_) {
        throw std::logic_error("Marginalization action requested before VIO initialization");
    }
    if (!marginalization_prior_) {
        if (state_->frames[kRetainedFrameIndex].frame_id != tassel_utils::kInvalidFrameId) {
            throw std::logic_error("Retained slot initialization was requested more than once");
        }
        return RetainedHostAction::InitializeRetainedSlot;
    }
    if (state_->frames[kRetainedFrameIndex].frame_id == tassel_utils::kInvalidFrameId) {
        throw std::logic_error("Marginalization prior has an invalid retained slot");
    }
    return state_->frames[kFirstActiveFrameIndex].frame_type == FrameType::KeyFrame
               ? RetainedHostAction::ReplaceRetainedSlot
               : RetainedHostAction::MarginalizeOldestFrame;
}

void Estimator::updateMarginalizationPrior(RetainedHostAction action) {
    const int window_capacity = state_->max_frame_count;
    TASSEL_ASSERT(window_capacity >= 3);
    state_->stateToParams();
    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;

    const bool initializes_retained_slot = action == RetainedHostAction::InitializeRetainedSlot;
    if (initializes_retained_slot) {
        if (marginalization_prior_ ||
            state_->frames[kRetainedFrameIndex].frame_id != tassel_utils::kInvalidFrameId) {
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
        updated_prior->linearization_delay_time = state_->param_time_delay;
        const int retained_host_source_index = action == RetainedHostAction::MarginalizeOldestFrame
                                                   ? kRetainedFrameIndex
                                                   : kFirstActiveFrameIndex;
        updated_prior->linearization_poses[0] =
            state_->frames[retained_host_source_index].param_pose;
        updated_prior->linearization_speed_bias[0] =
            state_->frames[retained_host_source_index].param_speed_bias;
        for (int i = 2; i < window_capacity; ++i) {
            updated_prior->linearization_poses[i - 1] = state_->frames[i].param_pose;
            updated_prior->linearization_speed_bias[i - 1] = state_->frames[i].param_speed_bias;
        }
        marginalization_prior_ = std::move(updated_prior);
    });
}

void Estimator::slideInitializationWindow() {
    TASSEL_ASSERT(!marginalization_prior_);
    const int n = state_->max_frame_count;
    for (int i = kFirstActiveFrameIndex; i < n - 1; ++i) {
        state_->copyFrameState(i + 1, i);
    }
    state_->frames[n - 1].frame_id = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
        for (int i = kFirstActiveFrameIndex; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(
            state_->frames[n - 2].accel_bias, state_->frames[n - 2].gyro_bias, noise_);
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
    state_->frames[window_capacity - 1].frame_id = tassel_utils::kInvalidFrameId;
    visitPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(state_->frames[0].accel_bias, state_->frames[0].gyro_bias, noise_);
        for (int i = first_movable_index; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(
            state_->frames[window_capacity - 2].accel_bias,
            state_->frames[window_capacity - 2].gyro_bias, noise_);
    });

    if (action != RetainedHostAction::MarginalizeOldestFrame) {
        state_->captureGauge(kRetainedFrameIndex);
    }
}

void Estimator::normalizeGaugeAfterOptimization(int reference_frame_index) {
    if (!initialized_) {
        throw std::logic_error("Gauge normalization requested before VIO initialization");
    }
    if (reference_frame_index < 0 || reference_frame_index > state_->latest_active_frame_index) {
        throw std::out_of_range("Gauge reference frame is outside the active window");
    }
    if (!state_->gauge_reference) {
        throw std::logic_error("Gauge anchor is unavailable");
    }

    const FrameState& optimized_reference = state_->frames[reference_frame_index];
    const GaugeAnchor& anchor = *state_->gauge_reference;
    if (optimized_reference.frame_id != anchor.reference_frame_id) {
        throw std::logic_error("Gauge anchor does not match its reference frame");
    }
    if (!anchor.reference_rotation.allFinite() || !anchor.reference_position.allFinite() ||
        !optimized_reference.pos_w_i.allFinite()) {
        throw std::logic_error("Gauge anchor or reference state is not finite");
    }
    const double yaw_correction =
        rotationYaw(anchor.reference_rotation) - rotationYaw(optimized_reference.rot_w_i);
    const Eigen::Matrix3d rotation_correction =
        Eigen::AngleAxisd(yaw_correction, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d optimized_reference_position = optimized_reference.pos_w_i;
    const Eigen::Vector3d translation =
        anchor.reference_position - rotation_correction * optimized_reference_position;
    for (int frame_index = reference_frame_index; frame_index <= state_->latest_active_frame_index;
         ++frame_index) {
        const FrameState& frame = state_->frames[frame_index];
        if (!frame.pos_w_i.allFinite() || !frame.rot_w_i.allFinite() || !frame.vel_w.allFinite()) {
            throw std::logic_error("Gauge window contains a non-finite state");
        }
    }
    for (int frame_index = reference_frame_index; frame_index <= state_->latest_active_frame_index;
         ++frame_index) {
        FrameState& frame = state_->frames[frame_index];
        frame.pos_w_i = rotation_correction * (frame.pos_w_i - optimized_reference_position) +
                        anchor.reference_position;
        frame.rot_w_i = rotation_correction * frame.rot_w_i;
        frame.vel_w = rotation_correction * frame.vel_w;
        frame.stateToParam();
    }
    if (marginalization_prior_) {
        MargHelper::transformPriorGauge(*marginalization_prior_, rotation_correction, translation);
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
    const int last_frame_index = state_->latest_active_frame_index;
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
            state_->frames[i].gyro_bias = bg;
        }
    }

    visitPreintegrators([&](auto& preintegrators) {
        for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
            preintegrators[i].repropagate(
                state_->frames[i].accel_bias, state_->frames[i].gyro_bias, noise_);
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
        state_->frames[frame_index].rot_w_i =
            Eigen::Quaterniond(R0 * params_.ric * Rs_[local_index] * params_.ric.transpose())
                .normalized()
                .toRotationMatrix();
        state_->frames[frame_index].pos_w_i =
            R0 *
            (params_.ric * s * Ps_[local_index] -
             params_.ric * Rs_[local_index] * params_.ric.transpose() * params_.tic + params_.tic);
        state_->frames[frame_index].vel_w = R0 * Vs_[local_index];
    }
    spdlog::info(
        "VI init: |g|={:.4f} s={:.4f} R0_yaw={:.2f}°", tassel_utils::G.norm(), s,
        yaw * 180.0 / M_PI);
    return true;
}

}  // namespace tassel_core
