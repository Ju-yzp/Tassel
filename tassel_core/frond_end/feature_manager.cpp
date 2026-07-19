// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

// Eigen
#include <Eigen/Core>
#include <Eigen/SVD>

// 标准库
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

// Tassel
#include "cam/camera_base.h"
#include "feature.h"
#include "feature_manager.h"
#include "initial/initial_sfm.h"
#include "reprojection.h"
#include "state/state.h"

#include <spdlog/spdlog.h>

namespace tassel_core {

namespace {
inline double computeParallax(const cv::Point2f& p1, const cv::Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

inline bool canUseFeature(const Feature& feature, int tracked_times_thres) {
    const int observation_count = static_cast<int>(feature.observations.size());
    return feature.estimated_depth != INVALID_DEPTH && observation_count >= 2 &&
           (feature.has_been_marginalized || observation_count >= tracked_times_thres);
}

bool computeReprojectionError(
    const FrameState& host, const FrameState& target, const FeaturePerFrame& host_observation,
    const FeaturePerFrame& target_observation, double depth, double delay_time,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const CameraBase& camera,
    double& error) {
    Eigen::Vector3d point_c;
    if (!reprojectToTargetCamera(
            host, target, host_observation.uv, depth, host_observation.sync_delay,
            target_observation.sync_delay, delay_time, ric, tic, point_c)) {
        return false;
    }

    const Eigen::Vector2d predicted = camera.distort(point_c.head<2>() / point_c.z());
    const Eigen::Vector2d measured(target_observation.pt.x, target_observation.pt.y);
    if (!predicted.allFinite() || !measured.allFinite()) {
        return false;
    }
    error = (predicted - measured).norm();
    return std::isfinite(error);
}
}  // namespace

FeatureManager::FeatureManager(
    double reproj_err_thres, double parallax_thres, int tracked_times_thres, int min_tracked_pts,
    double min_translation, double min_depth, double max_depth)
    : reproj_err_thres_(reproj_err_thres),
      parallax_thres_(parallax_thres),
      tracked_times_thres_(tracked_times_thres),
      min_tracked_pts_(min_tracked_pts),
      min_translation_(min_translation),
      min_depth_(min_depth),
      max_depth_(max_depth) {
    if (reproj_err_thres_ <= 0.0 || parallax_thres_ < 0.0 || tracked_times_thres_ < 2 ||
        min_tracked_pts_ < 1 || min_translation_ < 0.0 || min_depth_ <= 0.0 ||
        max_depth_ <= min_depth_) {
        throw std::invalid_argument("Invalid FeatureManager configuration");
    }
    features_.reserve(1000);
}

void FeatureManager::logInputStats(bool is_keyframe) const {
    const double new_feature_ratio = last_input_stats_.input_count > 0
                                         ? static_cast<double>(last_input_stats_.created_count) /
                                               static_cast<double>(last_input_stats_.input_count)
                                         : 1.0;
    spdlog::info(
        "Frame features: input={} matched={} new={} new_ratio={:.3f} kf_connected={} "
        "kf_current_ratio={:.3f} kf_retention={:.3f} parallax={:.3f} keyframe={}",
        last_input_stats_.input_count, last_input_stats_.matched_count,
        last_input_stats_.created_count, new_feature_ratio,
        last_input_stats_.connected_to_keyframe_count,
        last_input_stats_.current_keyframe_connection_ratio,
        last_input_stats_.keyframe_feature_retention_ratio, last_input_stats_.average_parallax,
        is_keyframe);
}

bool FeatureManager::checkParallax(
    int frame_slot, const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    double keyframe_parallax_sum = 0.0;
    size_t keyframe_parallax_count = 0;
    last_input_stats_ = {};
    last_input_stats_.input_count = feature_frame.size();

    for (const auto& [id, per_frame_feature] : feature_frame) {
        const auto keyframe_feature = latest_keyframe_features_.find(id);
        if (keyframe_feature != latest_keyframe_features_.end()) {
            ++last_input_stats_.connected_to_keyframe_count;
        }
        FeaturePerFrame observation = per_frame_feature;
        auto it = features_.find(id);
        if (it != features_.end()) {
            const int expected_slot =
                it->second.start_slot + static_cast<int>(it->second.observations.size());
            if (expected_slot != frame_slot) {
                throw std::logic_error("Feature observation is not continuous");
            }
            ++last_input_stats_.matched_count;
            if (keyframe_feature != latest_keyframe_features_.end()) {
                keyframe_parallax_sum +=
                    computeParallax(keyframe_feature->second, per_frame_feature.pt);
                ++keyframe_parallax_count;
            }
            it->second.observations.emplace_back(std::move(observation));
        } else {
            ++last_input_stats_.created_count;
            Feature feature(frame_slot, 15);
            feature.observations.emplace_back(std::move(observation));
            features_.emplace(id, std::move(feature));
        }
    }

    if (keyframe_parallax_count > 0) {
        last_input_stats_.average_parallax =
            keyframe_parallax_sum / static_cast<double>(keyframe_parallax_count);
    }
    if (last_input_stats_.input_count > 0) {
        last_input_stats_.current_keyframe_connection_ratio =
            static_cast<double>(last_input_stats_.connected_to_keyframe_count) /
            static_cast<double>(last_input_stats_.input_count);
    }
    if (!latest_keyframe_features_.empty()) {
        last_input_stats_.keyframe_feature_retention_ratio =
            static_cast<double>(last_input_stats_.connected_to_keyframe_count) /
            static_cast<double>(latest_keyframe_features_.size());
    }

    return !hasLatestKeyframe() ||
           keyframe_parallax_count < static_cast<size_t>(min_tracked_pts_) ||
           last_input_stats_.average_parallax > parallax_thres_;
}

void FeatureManager::acceptKeyframe(const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    latest_keyframe_features_.clear();
    latest_keyframe_features_.reserve(feature_frame.size());
    for (const auto& [feature_id, observation] : feature_frame) {
        latest_keyframe_features_.emplace(feature_id, observation.pt);
    }
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    for (auto& item : features_) {
        item.second.monoTriangulate(state, ric, tic, min_translation_, min_depth_, max_depth_);
    }
}

void FeatureManager::removeOldestFrameObservations(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (state.newest_slot > 0) {
        removeFrameObservations(0, state, ric, tic);
    }
}

void FeatureManager::removeFrameObservations(
    int removed_slot, const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (state.newest_slot > 0) {
        std::erase_if(features_, [&](const auto& item) {
            return item.second.start_slot == removed_slot && item.second.observations.size() == 1;
        });
        for (auto& item : features_) {
            Feature& feature = item.second;
            feature.removeFrame(removed_slot, state, ric, tic);
            if (feature.start_slot > removed_slot) {
                --feature.start_slot;
            }
        }

        std::erase_if(
            features_, [&](const auto& item) { return item.second.observations.empty(); });
    }
}

void FeatureManager::replaceRetainedHost(
    int old_host_slot, int new_host_slot, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    std::erase_if(features_, [&](auto& item) {
        Feature& feature = item.second;
        if (feature.start_slot != old_host_slot) {
            return false;
        }
        if (!feature.transferHost(new_host_slot, state, ric, tic)) {
            return true;
        }
        feature.observations.erase(feature.observations.begin() + 1);
        feature.start_slot = 0;
        return feature.observations.size() < 2;
    });

    for (auto& [_, feature] : features_) {
        if (feature.start_slot > old_host_slot) {
            --feature.start_slot;
        }
    }
    std::erase_if(features_, [](const auto& item) { return item.second.observations.empty(); });
}

void FeatureManager::removeNewestFrameObservations(int frame_slot) {
    for (auto& item : features_) {
        item.second.removeFrameObservation(frame_slot);
    }

    std::erase_if(features_, [&](const auto& item) { return item.second.observations.empty(); });
}

void FeatureManager::removeOutliers(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    struct OutlierStats {
        size_t checked_count = 0;
        size_t removed_count = 0;
        size_t invalid_count = 0;
        double average_error = 0.0;
        double maximum_error = 0.0;
        double removed_average_error = 0.0;
    } stats;
    size_t finite_error_count = 0;
    size_t finite_removed_error_count = 0;
    if (!state.camera) {
        return;
    }

    std::set<int> removed_ids;
    for (auto& [id, feature] : features_) {
        double depth = feature.estimated_depth;
        std::vector<FeaturePerFrame>& observations = feature.observations;
        if (!canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }

        const int host_slot = feature.start_slot;
        if (host_slot < 0 || host_slot > state.newest_slot) {
            removed_ids.insert(id);
            ++stats.checked_count;
            ++stats.invalid_count;
            continue;
        }

        double error_sum = 0.0;
        size_t valid_observation_count = 0;
        bool invalid_projection = false;
        for (size_t k = 1; k < observations.size(); ++k) {
            const int target_slot = feature.observationSlot(k);
            if (target_slot < 0 || target_slot > state.newest_slot) {
                invalid_projection = true;
                break;
            }
            double reprojection_error = 0.0;
            if (!computeReprojectionError(
                    state.frames[host_slot], state.frames[target_slot], observations[0],
                    observations[k], depth, state.delay_time, ric, tic, *state.camera,
                    reprojection_error)) {
                invalid_projection = true;
                break;
            }
            error_sum += reprojection_error;
            ++valid_observation_count;
        }

        ++stats.checked_count;
        if (invalid_projection || valid_observation_count == 0) {
            removed_ids.insert(id);
            ++stats.invalid_count;
            continue;
        }
        double average_error = error_sum / static_cast<double>(valid_observation_count);
        ++finite_error_count;
        stats.average_error += average_error;
        stats.maximum_error = std::max(stats.maximum_error, average_error);
        if (average_error > reproj_err_thres_) {
            removed_ids.insert(id);
            stats.removed_average_error += average_error;
            ++finite_removed_error_count;
        }
    }

    std::erase_if(features_, [&](const auto& item) { return removed_ids.count(item.first) > 0; });
    stats.removed_count = removed_ids.size();
    if (finite_error_count > 0) {
        stats.average_error /= static_cast<double>(finite_error_count);
    }
    if (finite_removed_error_count > 0) {
        stats.removed_average_error /= static_cast<double>(finite_removed_error_count);
    }
    spdlog::info(
        "Feature outliers: checked={} removed={} invalid={} mean={:.3f}px max={:.3f}px "
        "removed_mean={:.3f}px",
        stats.checked_count, stats.removed_count, stats.invalid_count, stats.average_error,
        stats.maximum_error, stats.removed_average_error);
}

void FeatureManager::reset() {
    features_.clear();
    latest_keyframe_features_.clear();
    last_input_stats_ = {};
}

std::vector<MarginalizedFeatureObservation> FeatureManager::collectMarginalizedObservations(
    int host_slot, int target_slot) {
    std::vector<MarginalizedFeatureObservation> result;
    for (auto& item : features_) {
        auto& feature = item.second;
        if (feature.start_slot != host_slot || !canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }
        const int observation_index = target_slot - feature.start_slot;
        if (observation_index <= 0 ||
            observation_index >= static_cast<int>(feature.observations.size())) {
            continue;
        }
        feature.has_been_marginalized = true;
        result.push_back({&feature, target_slot});
    }
    return result;
}

std::vector<Eigen::Vector3d> FeatureManager::getPointCloud(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) const {
    auto cs = state.get_compensated_state();
    std::vector<Eigen::Vector3d> points;
    for (const auto& item : features_) {
        const auto& feature = item.second;
        if (feature.estimated_depth <= 0) {
            continue;
        }
        const int host_slot = feature.start_slot;
        if (host_slot < 0 || host_slot > cs.newest_slot) {
            continue;
        }
        Eigen::Vector3d pt_in_C = feature.observations[0].uv * feature.estimated_depth;
        Eigen::Vector3d pt_in_I = ric * pt_in_C + tic;
        Eigen::Vector3d pt_in_W = cs.frames[host_slot].R * pt_in_I + cs.frames[host_slot].P;
        points.push_back(pt_in_W);
    }
    return points;
}

std::vector<Feature*> FeatureManager::collectLandmarks() {
    std::vector<Feature*> result;
    for (auto& item : features_) {
        auto& feature = item.second;
        if (!canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }
        result.push_back(&feature);
    }
    return result;
}

std::vector<SFMFeature> FeatureManager::collectSFMFeatures(const State& state) const {
    std::vector<SFMFeature> sfm_features;
    sfm_features.reserve(features_.size());
    for (const auto& [id, feature] : features_) {
        SFMFeature sfm_f;
        sfm_f.state = false;
        sfm_f.id = id;
        sfm_f.position[0] = 0;
        sfm_f.position[1] = 0;
        sfm_f.position[2] = 0;
        for (size_t observation_index = 0; observation_index < feature.observations.size();
             ++observation_index) {
            const auto& observation = feature.observations[observation_index];
            const int frame_slot = feature.observationSlot(observation_index);
            if (frame_slot > state.newest_slot) {
                continue;
            }
            Eigen::Vector2d uv_norm(observation.uv(0), observation.uv(1));
            sfm_f.observation.emplace_back(frame_slot, uv_norm);
        }
        sfm_features.push_back(std::move(sfm_f));
    }
    return sfm_features;
}

}  // namespace tassel_core
