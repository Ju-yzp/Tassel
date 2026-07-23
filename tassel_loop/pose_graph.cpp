#include "pose_graph.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tassel_loop {
namespace {

constexpr double kOdometryRotationSigma = 0.02;
constexpr double kOdometryTranslationSigma = 0.05;

gtsam::Pose3 makePose(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    return gtsam::Pose3(gtsam::Rot3(rotation), gtsam::Point3(translation));
}

}  // namespace

void PoseGraph::addKeyframe(
    KeyframeId frame_id, const Eigen::Matrix3d& local_R_camera,
    const Eigen::Vector3d& local_t_camera) {
    if (frame_keys_.find(frame_id) != frame_keys_.end()) {
        throw std::invalid_argument("Duplicate pose graph frame id");
    }
    const gtsam::Key key = gtsam::Symbol('x', frame_keys_.size());
    const gtsam::Pose3 local_pose = makePose(local_R_camera, local_t_camera);
    frame_keys_[frame_id] = key;
    frame_order_.push_back(frame_id);

    if (!has_last_key_) {
        values_.insert(key, local_pose);
        const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
        factors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, local_pose, prior_noise));
    } else {
        const gtsam::Pose3 odometry = last_local_pose_.between(local_pose);
        const gtsam::Pose3 graph_pose = values_.at<gtsam::Pose3>(last_key_).compose(odometry);
        values_.insert(key, graph_pose);
        const auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << kOdometryRotationSigma, kOdometryRotationSigma,
             kOdometryRotationSigma, kOdometryTranslationSigma, kOdometryTranslationSigma,
             kOdometryTranslationSigma)
                .finished());
        factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(last_key_, key, odometry, odometry_noise));
        ++odometry_factor_count_;
    }
    last_key_ = key;
    last_local_pose_ = local_pose;
    has_last_key_ = true;
}

bool PoseGraph::addPoseLoop(
    KeyframeId candidate_frame_id, KeyframeId current_frame_id,
    const Eigen::Matrix3d& candidate_R_current, const Eigen::Vector3d& candidate_t_current,
    const PoseLoopNoise& noise) {
    const auto candidate = frame_keys_.find(candidate_frame_id);
    const auto current = frame_keys_.find(current_frame_id);
    if (candidate == frame_keys_.end() || current == frame_keys_.end()) {
        return false;
    }
    if (!candidate_R_current.allFinite() || !candidate_t_current.allFinite() ||
        !(candidate_R_current.transpose() * candidate_R_current)
             .isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
        candidate_R_current.determinant() < 0.0 || noise.rotation_variance <= 0.0 ||
        noise.translation_variance <= 0.0 || pending_loop_.has_value()) {
        throw std::invalid_argument("Pose loop measurement is invalid");
    }

    PoseLoopNoise limited_noise;
    limited_noise.rotation_variance =
        std::max(noise.rotation_variance, std::pow(kOdometryRotationSigma, 2));
    limited_noise.translation_variance =
        std::max(noise.translation_variance, std::pow(kOdometryTranslationSigma, 2));
    const gtsam::Pose3 measurement = makePose(candidate_R_current, candidate_t_current);
    const auto diagonal_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << std::sqrt(limited_noise.rotation_variance),
         std::sqrt(limited_noise.rotation_variance), std::sqrt(limited_noise.rotation_variance),
         std::sqrt(limited_noise.translation_variance),
         std::sqrt(limited_noise.translation_variance),
         std::sqrt(limited_noise.translation_variance))
            .finished());
    const auto robust_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), diagonal_noise);
    factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        candidate->second, current->second, measurement, robust_noise));
    pending_loop_ =
        PendingLoop{candidate->second, current->second, measurement, limited_noise, optimized_};
    ++loop_factor_count_;
    optimized_ = false;
    return true;
}

bool PoseGraph::optimize(double max_normalized_loop_error) {
    if (max_normalized_loop_error < 0.0) {
        throw std::invalid_argument("Maximum normalized loop error cannot be negative");
    }
    if (loop_factor_count_ == 0 || values_.empty()) {
        return false;
    }
    last_optimization_stats_ = {};
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SILENT");
    const double error_before = factors_.error(values_);
    gtsam::Values optimized_values =
        gtsam::LevenbergMarquardtOptimizer(factors_, values_, params).optimize();
    const double error_after = factors_.error(optimized_values);
    if (!std::isfinite(error_after) || error_after > error_before + 1e-9) {
        last_optimization_stats_.error_before = error_before;
        last_optimization_stats_.error_after = error_after;
        if (pending_loop_) {
            last_optimization_stats_.loop_rejected = true;
            factors_.resize(factors_.size() - 1);
            --loop_factor_count_;
            optimized_ = pending_loop_->was_optimized;
            pending_loop_.reset();
        }
        return false;
    }
    PoseGraphOptimizationStats optimization_stats;
    optimization_stats.error_before = error_before;
    optimization_stats.error_after = error_after;
    if (pending_loop_) {
        optimization_stats.loop_translation_variance = pending_loop_->noise.translation_variance;
        optimization_stats.loop_rotation_variance = pending_loop_->noise.rotation_variance;
        const gtsam::Pose3 predicted =
            optimized_values.at<gtsam::Pose3>(pending_loop_->candidate_key)
                .between(optimized_values.at<gtsam::Pose3>(pending_loop_->current_key));
        const gtsam::Vector6 residual =
            gtsam::Pose3::Logmap(pending_loop_->measurement.between(predicted));
        const double rotation_error = residual.head<3>().cwiseAbs().maxCoeff() /
                                      std::sqrt(pending_loop_->noise.rotation_variance);
        const double translation_error = residual.tail<3>().cwiseAbs().maxCoeff() /
                                         std::sqrt(pending_loop_->noise.translation_variance);
        optimization_stats.max_normalized_loop_error = std::max(rotation_error, translation_error);
        if (max_normalized_loop_error > 0.0 &&
            optimization_stats.max_normalized_loop_error > max_normalized_loop_error) {
            optimization_stats.loop_rejected = true;
            factors_.resize(factors_.size() - 1);
            --loop_factor_count_;
            optimized_ = pending_loop_->was_optimized;
            pending_loop_.reset();
            last_optimization_stats_ = optimization_stats;
            return false;
        }
    }
    for (const auto& [frame_id, key] : frame_keys_) {
        (void)frame_id;
        const gtsam::Pose3 correction =
            values_.at<gtsam::Pose3>(key).between(optimized_values.at<gtsam::Pose3>(key));
        optimization_stats.max_translation_correction = std::max(
            optimization_stats.max_translation_correction, correction.translation().norm());
        optimization_stats.max_rotation_correction = std::max(
            optimization_stats.max_rotation_correction,
            gtsam::Rot3::Logmap(correction.rotation()).norm());
    }
    values_ = std::move(optimized_values);
    pending_loop_.reset();
    last_optimization_stats_ = optimization_stats;
    optimized_ = true;
    return true;
}

std::optional<GraphPose> PoseGraph::pose(KeyframeId frame_id) const {
    const auto key = frame_keys_.find(frame_id);
    if (key == frame_keys_.end()) {
        return std::nullopt;
    }
    const gtsam::Pose3 pose = values_.at<gtsam::Pose3>(key->second);
    return GraphPose{pose.rotation().matrix(), pose.translation()};
}

std::vector<std::pair<KeyframeId, GraphPose>> PoseGraph::poses() const {
    std::vector<std::pair<KeyframeId, GraphPose>> output;
    output.reserve(frame_order_.size());
    for (KeyframeId frame_id : frame_order_) {
        output.emplace_back(frame_id, *pose(frame_id));
    }
    return output;
}

PoseGraphStats PoseGraph::stats() const {
    return {frame_keys_.size(), odometry_factor_count_, loop_factor_count_, optimized_};
}

}  // namespace tassel_loop
