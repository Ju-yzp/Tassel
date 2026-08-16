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
#include "factor/midpoint_integrator.h"
#include "frond_end/reprojection.h"
#include "imu_interpolation.h"
#include "marg/marg_helper.h"
#include "marg/marginalization_sqrt.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

namespace {
constexpr int kRetainedFrameIndex = 0;
constexpr int kFirstActiveFrameIndex = 1;
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

std::vector<MidPointIntegrator> makePreintegrators(
    size_t count, const Eigen::Matrix<double, 18, 18>& noise) {
    return std::vector<MidPointIntegrator>(
        count, MidPointIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise));
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
    dynamic_initializer_ = std::make_unique<DynamicInitializer>(
        params_, state_, feature_manager_, preintegrators_, noise_);
    reset();
}

Estimator::~Estimator() = default;

void Estimator::reset() {
    initialized_ = false;
    last_ts_ = -1;
    last_imu_acc_ = Eigen::Vector3d::Zero();
    last_imu_gyro_ = Eigen::Vector3d::Zero();
    preintegrators_ = makePreintegrators(state_->max_frame_count - 1, noise_);
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
        last_imu_acc_ = imu_measurements.back().acc;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    const bool is_first_initial_frame =
        !initialized_ && frame_count == kFirstActiveFrameIndex &&
        state_->frames[kRetainedFrameIndex].frame_id == tassel_utils::kInvalidFrameId;
    const bool is_keyframe =
        initialized_ ? feature_manager_->addFeatureFrame(frame_count, feature_frame)
                     : feature_manager_->tryAddInitializationKeyframe(frame_count, feature_frame);
    state_->frames[frame_count].frame_type =
        is_keyframe ? FrameType::KeyFrame : FrameType::NonKeyFrame;
    last_measurement_was_keyframe_ = is_keyframe;

    if (!is_first_initial_frame) {
        predictLatestFrame(frame_count, imu_measurements);
    }

    const double imu_query_timestamp = ts + sync_delay;
    interpolateBodyImu(
        imu_measurements, imu_query_timestamp, state_->frames[frame_count].imu_gyro,
        state_->frames[frame_count].imu_acc);
    if (!initialized_) {
        if (!is_keyframe) {
            // 初始化只保存关键帧；候选期间状态和 IMU 预积分继续在当前槽累计。
            return;
        }

        // 接受的关键帧固定为独立 VIO 状态并推进窗口。
        if (frame_count < state_->max_frame_count - 1) {
            ++frame_count;
            state_->seedFrameFromPosterior(frame_count - 1, frame_count);
            int next_idx = frame_count - 1;
            withPreintegrators([&](auto& preintegrators) {
                if (next_idx < static_cast<int>(preintegrators.size())) {
                    preintegrators[next_idx].reset(
                        state_->frames[frame_count - 1].accel_bias,
                        state_->frames[frame_count - 1].gyro_bias, noise_);
                }
            });
            return;
        }

        if (!dynamic_initializer_->initialize()) {
            spdlog::debug("VI initialization not ready; sliding initialization window");
            feature_manager_->removeFrameObservations(
                kFirstActiveFrameIndex, *state_, params_.ric, params_.tic);
            slideInitializationWindow();
            return;
        }
        initialized_ = true;
    }

    processWindow(frame_count, ts);
}

void Estimator::processWindow(int latest_active_frame_index, double timestamp) {
    const RetainedHostAction action = selectMarginalizationAction();
    feature_manager_->triangulate(*state_, params_.ric, params_.tic);
    optimizeWindow();

    const Sophus::SE3d optimized_pose(
        state_->frames[latest_active_frame_index].rot_w_i,
        state_->frames[latest_active_frame_index].pos_w_i);
    if (pose_callback_) {
        pose_callback_(timestamp, optimized_pose);
    }

    // 离群特征必须在构建新先验前删除，避免其约束写入边缘化先验。
    feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
    updatePrior(action);

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
    slideWindow(action);
}

void Estimator::predictLatestFrame(
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

    withPreintegrators([&](auto& preintegrators) {
        auto& preintegrator = preintegrators[frame_index - 1];

        for (const auto& imu : imu_measurements) {
            tassel_utils::IMUMeasurement calibrated_imu = imu;
            if (!preintegrator.buffer.empty() &&
                calibrated_imu.timestamp == preintegrator.buffer.back().timestamp) {
                const auto& boundary = preintegrator.buffer.back();
                if (!calibrated_imu.acc.isApprox(boundary.acc, 1e-12) ||
                    !calibrated_imu.gyro.isApprox(boundary.gyro, 1e-12)) {
                    throw std::runtime_error("Shared IMU boundary has inconsistent measurements");
                }
                // 同一预积分器连续接收同步包时，相邻包会共享同一个边界样本。
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

void Estimator::updatePrior(RetainedHostAction action) {
    const int window_capacity = state_->max_frame_count;
    TASSEL_ASSERT(window_capacity >= 3);
    state_->prepareOptimization();
    const double visual_huber_delta = params_.reproj_huber_thres * params_.visual_factor_weight;

    if (marginalization_prior_) {
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
    withPreintegrators([&](auto& preintegrators) {
        constexpr int imu_index = 1;
        if (preintegrators[imu_index].buffer.size() < 2) {
            throw std::logic_error(
                "Marginalization IMU interval has fewer than two measurements at index 1");
        }
        std::vector<MidPointIntegrator*> imu_preintegrators = {&preintegrators[imu_index]};

        auto linearizer = MarginalizationSqrt(
            std::move(retiring_features), landmark_target_frame_index,
            std::make_unique<ceres::HuberLoss>(visual_huber_delta), state_, imu_preintegrators,
            params_.ric, params_.tic, marginalization_prior_.get(), imu_index);
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
        updated_prior->linearization_delay_time = *state_->linearizedTimeDelay();
        if (action == RetainedHostAction::MarginalizeOldestFrame) {
            updated_prior->linearization_poses[0] =
                state_->frames[kRetainedFrameIndex].linearized_pose;
            updated_prior->linearization_speed_bias[0] =
                state_->frames[kRetainedFrameIndex].linearized_speed_bias;
        } else {
            updated_prior->linearization_poses[0] =
                state_->frames[kFirstActiveFrameIndex].linearized_pose;
            updated_prior->linearization_speed_bias[0] =
                state_->frames[kFirstActiveFrameIndex].linearized_speed_bias;
        }
        for (int i = 2; i < window_capacity; ++i) {
            updated_prior->linearization_poses[i - 1] = state_->frames[i].linearized_pose;
            updated_prior->linearization_speed_bias[i - 1] =
                state_->frames[i].linearized_speed_bias;
        }
        marginalization_prior_ = std::move(updated_prior);
    });
}

void Estimator::slideInitializationWindow() {
    TASSEL_ASSERT(!marginalization_prior_);
    const int n = state_->max_frame_count;
    for (int i = kFirstActiveFrameIndex; i < n - 1; ++i) {
        state_->copyFrame(i + 1, i);
    }
    // 尾槽代表下一帧的新状态身份：继承上一后验作为预测初值，但不能继承被移出帧的 FEJ 点。
    state_->seedFrameFromPosterior(n - 2, n - 1);
    withPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
        for (int i = kFirstActiveFrameIndex; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(
            state_->frames[n - 2].accel_bias, state_->frames[n - 2].gyro_bias, noise_);
    });
}

void Estimator::slideWindow(RetainedHostAction action) {
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
        // frame0 是 retained 状态的唯一所有者；替换时完整继承 frame1 的 current 和 FEJ。
        state_->copyFrame(kFirstActiveFrameIndex, kRetainedFrameIndex);
    }

    for (int i = first_movable_index; i < window_capacity - 1; ++i) {
        state_->copyFrame(i + 1, i);
    }
    // 尾槽复用时必须创建新的 FEJ 身份，否则固定内存槽会永久携带首次占用者的线性化点。
    state_->seedFrameFromPosterior(window_capacity - 2, window_capacity - 1);
    withPreintegrators([&](auto& preintegrators) {
        preintegrators[0].reset(state_->frames[0].accel_bias, state_->frames[0].gyro_bias, noise_);
        for (int i = first_movable_index; i < static_cast<int>(preintegrators.size()) - 1; ++i) {
            preintegrators[i] = std::move(preintegrators[i + 1]);
        }
        preintegrators.back().reset(
            state_->frames[window_capacity - 2].accel_bias,
            state_->frames[window_capacity - 2].gyro_bias, noise_);
    });
}

void Estimator::normalizeGauge(
    int reference_frame_index, const Eigen::Matrix3d& reference_rotation,
    const Eigen::Vector3d& reference_position) {
    if (reference_frame_index < 0 || reference_frame_index > state_->latest_active_frame_index) {
        throw std::out_of_range("Gauge reference frame is outside the active window");
    }
    if (marginalization_prior_ && reference_frame_index != kRetainedFrameIndex) {
        throw std::logic_error("Prior gauge reference is not the retained frame");
    }
    const FrameState& optimized_reference = state_->frames[reference_frame_index];
    if (!reference_position.allFinite() || !optimized_reference.pos_w_i.allFinite()) {
        throw std::logic_error("Gauge reference state is not finite");
    }
    const double yaw_correction =
        rotationYaw(reference_rotation) - rotationYaw(optimized_reference.rot_w_i);
    const Eigen::Matrix3d rotation_correction =
        Eigen::AngleAxisd(yaw_correction, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d optimized_reference_position = optimized_reference.pos_w_i;
    for (int frame_index = reference_frame_index; frame_index <= state_->latest_active_frame_index;
         ++frame_index) {
        const FrameState& frame = state_->frames[frame_index];
        if (!frame.pos_w_i.allFinite() || !frame.rot_w_i.allFinite() || !frame.vel_w.allFinite()) {
            throw std::logic_error("Gauge window contains a non-finite current state");
        }
        if (frame.has_linearized &&
            (!Eigen::Map<const Eigen::Matrix<double, 6, 1>>(frame.linearized_pose.data())
                  .allFinite() ||
             !Eigen::Map<const Eigen::Matrix<double, 9, 1>>(frame.linearized_speed_bias.data())
                  .allFinite())) {
            throw std::logic_error("Gauge window contains a non-finite linearization point");
        }
    }
    if (marginalization_prior_) {
        marginalization_prior_->transformGauge(
            rotation_correction, optimized_reference_position, reference_position);
    }
    for (int frame_index = reference_frame_index; frame_index <= state_->latest_active_frame_index;
         ++frame_index) {
        FrameState& frame = state_->frames[frame_index];
        frame.transformGauge(rotation_correction, optimized_reference_position, reference_position);
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

}  // namespace tassel_core

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <spdlog/spdlog.h>

#include "factor/imu_factor.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/reprojection_factor.h"
#include "factor/visual_frame_cache.h"
#include "marg/marg_lin_data.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

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

        const auto blockPriority = [](int size) {
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
        std::vector<RankedBlock> ranked;
        ranked.reserve(default_block_order.size());
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

std::unique_ptr<VisualFrameCache> createVisualFrameCache(
    State& state, const tassel_tools::Parameters& params) {
    return std::make_unique<VisualFrameCache>(state, params.ric);
}

void addParameterBlocks(ceres::Problem& problem, State& state, const MargLinData* prior) {
    for (int i = 0; i < state.max_frame_count; ++i) {
        problem.AddParameterBlock(state.frames[i].param_pose.data(), 6, new SE3RightManifold());
        if (i != kRetainedFrameIndex || prior == nullptr) {
            problem.AddParameterBlock(state.frames[i].param_speed_bias.data(), 9);
        }
    }
    if (prior == nullptr) {
        // 初始化期间 frame0 为空，不参与任何物理因子。
        problem.SetParameterBlockConstant(
            state.frames[kRetainedFrameIndex].param_speed_bias.data());
        problem.SetParameterBlockConstant(state.frames[kRetainedFrameIndex].param_pose.data());
    }
    problem.AddParameterBlock(state.currentTimeDelay(), 1);
}

ceres::ResidualBlockId addPriorFactor(
    ceres::Problem& problem, State& state, const MargLinData* prior) {
    if (prior == nullptr) {
        return nullptr;
    }
    auto* factor = new MarginalizationPriorFactor(*prior);
    std::vector<double*> blocks;
    blocks.push_back(state.frames[0].param_pose.data());
    for (int i = 1; i < prior->stateCount(); ++i) {
        blocks.push_back(state.frames[i].param_pose.data());
        blocks.push_back(state.frames[i].param_speed_bias.data());
    }
    blocks.push_back(state.currentTimeDelay());
    return problem.AddResidualBlock(factor, nullptr, blocks);
}

void addVisualFactors(
    ceres::Problem& problem, VisualFrameCache& cache, State& state,
    const tassel_tools::Parameters& params, const std::vector<std::pair<int, Feature>>& features,
    std::vector<double>& inverse_depths, std::vector<int>& factors_per_frame) {
    const double huber_delta = params.reproj_huber_thres * params.visual_factor_weight;
    ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);
    inverse_depths.resize(features.size());
    // inverse depth 与 feature 索引一一对应，求解后按同一索引生成深度结果。
    for (size_t k = 0; k < features.size(); ++k) {
        const Feature& feature = features[k].second;
        TASSEL_ASSERT(std::isfinite(feature.estimated_depth) && feature.estimated_depth > 1e-12);
        inverse_depths[k] = 1.0 / feature.estimated_depth;
        problem.AddParameterBlock(&inverse_depths[k], 1);
    }

    for (size_t k = 0; k < features.size(); ++k) {
        const Feature& feature = features[k].second;
        const int host_index = feature.host_frame_index;
        if (host_index < 0 || host_index > state.latest_active_frame_index) {
            throw std::logic_error("Feature host index is outside the active window");
        }
        for (size_t observation_index = 0; observation_index < feature.observations.size();
             ++observation_index) {
            const int frame_index = feature.observationFrameIndex(observation_index);
            if (frame_index <= state.latest_active_frame_index) {
                ++factors_per_frame[frame_index];
            }
        }
        for (size_t observation_index = 1; observation_index < feature.observations.size();
             ++observation_index) {
            const int target_index = feature.observationFrameIndex(observation_index);
            if (target_index > state.latest_active_frame_index) {
                throw std::logic_error("Feature target index is outside the active window");
            }
            const FeaturePerFrame& host = feature.observations.front();
            const FeaturePerFrame& target = feature.observations[observation_index];
            if (host.sync_delay != state.frames[host_index].image_sync_delay ||
                target.sync_delay != state.frames[target_index].image_sync_delay) {
                throw std::logic_error("Feature sync delay does not match its frame state");
            }
            const Eigen::Vector2d target_pixel(target.pt.x, target.pt.y);
            auto* cost = new ReprojectionFactor(
                host.uv, target_pixel, params.ric, params.tic, state.frames[host_index].imu_gyro,
                state.frames[target_index].imu_gyro, state.frames[host_index].imu_acc,
                state.frames[target_index].imu_acc,
                state.frames[host_index].param_speed_bias.data(),
                state.frames[target_index].param_speed_bias.data(),
                state.frames[host_index].param_speed_bias.data() + 6,
                state.frames[target_index].param_speed_bias.data() + 6,
                state.frames[host_index].param_speed_bias.data() + 3,
                state.frames[target_index].param_speed_bias.data() + 3, state.visual_sqrt_info,
                state.camera, host.sync_delay, target.sync_delay, &state, host_index, target_index);
            problem.AddResidualBlock(
                cost, loss, state.frames[host_index].param_pose.data(),
                state.frames[target_index].param_pose.data(), state.currentTimeDelay(),
                &inverse_depths[k]);
        }
    }
}

ceres::Solver::Options createSolverOptions(
    const tassel_tools::Parameters& params, State& state, const MargLinData* prior,
    std::vector<double>& inverse_depths) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
#if defined(CERES_HAS_SCHUR_STRUCTURE_HINTS)
    options.schur_structure_row_block_size = 2;
    options.schur_structure_e_block_size = 1;
    options.schur_structure_f_block_size = -1;
#endif
    options.trust_region_strategy_type = ceresTrustRegionStrategy(params.trust_region_strategy);
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (double& inverse_depth : inverse_depths) {
        ordering->AddElementToGroup(&inverse_depth, 0);
    }
    int group = 1;
    for (int i = 0; i < state.max_frame_count; ++i) {
        ordering->AddElementToGroup(state.frames[i].param_pose.data(), group++);
        if (i != kRetainedFrameIndex || prior == nullptr) {
            ordering->AddElementToGroup(state.frames[i].param_speed_bias.data(), group++);
        }
    }
    ordering->AddElementToGroup(state.currentTimeDelay(), group);
    options.linear_solver_ordering = std::move(ordering);
#if defined(CERES_HAS_SCHUR_LAYOUT_CALLBACK)
    options.schur_layout_callback = makeSchurLayoutCallback();
#endif
    options.max_num_iterations = params.num_iterations;
    if (params.max_solver_time > 0.0) {
        options.max_solver_time_in_seconds = params.max_solver_time;
    }
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    return options;
}

}  // namespace

void Estimator::optimizeWindow() {
    const int latest_id = state_->latest_active_frame_index;
    const int gauge_frame_index =
        marginalization_prior_ ? kRetainedFrameIndex : kFirstActiveFrameIndex;
    // 本轮参考必须来自求解前的 retained current；使用 frozen pose 会在 retained 换代时切换世界系。
    const Eigen::Matrix3d gauge_reference_rotation = state_->frames[gauge_frame_index].rot_w_i;
    const Eigen::Vector3d gauge_reference_position = state_->frames[gauge_frame_index].pos_w_i;
    state_->prepareOptimization();
    const auto features = feature_manager_->collectLandmarks();
    auto visual_cache = createVisualFrameCache(*state_, params_);
    ceres::Problem::Options problem_options;
    problem_options.evaluation_callback = visual_cache.get();
    ceres::Problem problem(problem_options);
    addParameterBlocks(problem, *state_, marginalization_prior_.get());
    addPriorFactor(problem, *state_, marginalization_prior_.get());

    std::vector<double> inverse_depths;
    std::vector<int> visual_factors_per_frame(latest_id + 1, 0);
    addVisualFactors(
        problem, *visual_cache, *state_, params_, features, inverse_depths,
        visual_factors_per_frame);

    withPreintegrators([&](auto& preintegrators) {
        const int imu_start = marginalization_prior_ ? 1 : 0;
        for (int i = imu_start; i < latest_id; ++i) {
            if (preintegrators[i].buffer.size() < 2) {
                continue;
            }
            auto integrator =
                std::shared_ptr<MidPointIntegrator>(&preintegrators[i], [](MidPointIntegrator*) {});
            problem.AddResidualBlock(
                new IMUFactor(std::move(integrator)), nullptr, state_->frames[i].param_pose.data(),
                state_->frames[i].param_speed_bias.data(), state_->frames[i + 1].param_pose.data(),
                state_->frames[i + 1].param_speed_bias.data());
        }
    });

    ceres::Solver::Summary summary;
    ceres::Solver::Options options =
        createSolverOptions(params_, *state_, marginalization_prior_.get(), inverse_depths);
    ceres::Solve(options, &problem, &summary);
    bool finite_solution = std::isfinite(*state_->currentTimeDelay());
    for (int i = 0; i <= latest_id && finite_solution; ++i) {
        finite_solution = allFinite(state_->frames[i].param_pose) &&
                          allFinite(state_->frames[i].param_speed_bias);
    }
    finite_solution = finite_solution && allFinite(inverse_depths);
    if (!summary.IsSolutionUsable() || !finite_solution) {
        spdlog::error("Optimization rejected: {}", summary.BriefReport());
        state_->prepareOptimization();
        throw std::runtime_error("VIO optimization failed");
    }

    std::vector<std::pair<int, double>> feature_depths;
    feature_depths.reserve(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        const double inverse_depth = inverse_depths[k];
        const double depth = inverse_depth > 1e-6 ? 1.0 / inverse_depth : Feature::InvalidDepth;
        feature_depths.emplace_back(features[k].first, depth);
    }
    if (visual_factor_callback_) {
        visual_factor_callback_(
            tassel_utils::frameIdToSeconds(state_->frames[latest_id].frame_id),
            visual_factors_per_frame);
    }
    state_->acceptOptimization();
    normalizeGauge(gauge_frame_index, gauge_reference_rotation, gauge_reference_position);
    feature_manager_->updateFeatureDepths(feature_depths);
}

}  // namespace tassel_core
