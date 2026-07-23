// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

// Eigen
#include <Eigen/Core>
#include <Eigen/SVD>

// 标准库
#include <cmath>
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

namespace tassel_core {

namespace {
inline bool canUseFeature(const Feature& feature, int tracked_times_thres) {
    const int observation_count = static_cast<int>(feature.observations.size());
    return feature.estimated_depth != INVALID_DEPTH && observation_count >= 2 &&
           (feature.has_been_marginalized || observation_count >= tracked_times_thres);
}

bool computeReprojectionError(
    const FrameState& target, const Eigen::Vector3d& world_point,
    const FeaturePerFrame& target_observation, double delay_time, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, const CameraBase& camera, double& error) {
    Eigen::Vector3d point_c;
    if (!worldPointToTargetCamera(
            target, world_point, target_observation.sync_delay, delay_time, ric, tic, point_c)) {
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
    double reproj_err_thres, int tracked_times_thres, double min_translation,
    double keyframe_new_feature_ratio, double min_depth, double max_depth)
    : reproj_err_thres_(reproj_err_thres),
      tracked_times_thres_(tracked_times_thres),
      min_translation_(min_translation),
      keyframe_new_feature_ratio_(keyframe_new_feature_ratio),
      min_depth_(min_depth),
      max_depth_(max_depth) {
    if (reproj_err_thres_ <= 0.0 || tracked_times_thres_ < 2 || min_translation_ < 0.0 ||
        keyframe_new_feature_ratio_ < 0.0 || keyframe_new_feature_ratio_ > 1.0 ||
        min_depth_ <= 0.0 || max_depth_ <= min_depth_) {
        throw std::invalid_argument("Invalid FeatureManager configuration");
    }
    features_.reserve(1000);
}

bool FeatureManager::addFeatureFrame(
    int frame_index, const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    const bool has_keyframe = hasLatestKeyframe();
    size_t connected_to_keyframe_count = 0;
    std::unordered_set<int> current_feature_ids;
    current_feature_ids.reserve(feature_frame.size());

    for (const auto& [id, per_frame_feature] : feature_frame) {
        current_feature_ids.insert(id);
        if (latest_keyframe_feature_ids_.contains(id)) {
            ++connected_to_keyframe_count;
        }
        FeaturePerFrame observation = per_frame_feature;
        auto it = features_.find(id);
        if (it != features_.end()) {
            const int expected_frame_index =
                it->second.host_frame_index + static_cast<int>(it->second.observations.size());
            if (expected_frame_index != frame_index) {
                throw std::logic_error("Feature observation is not continuous");
            }
            it->second.observations.emplace_back(std::move(observation));
        } else {
            Feature feature(frame_index, 15);
            feature.observations.emplace_back(std::move(observation));
            features_.emplace(id, std::move(feature));
        }
    }

    const double connection_ratio = feature_frame.empty()
                                        ? 0.0
                                        : static_cast<double>(connected_to_keyframe_count) /
                                              static_cast<double>(feature_frame.size());
    const bool is_keyframe =
        !has_keyframe || connection_ratio <= (1.0 - keyframe_new_feature_ratio_);
    if (is_keyframe) {
        latest_keyframe_feature_ids_.swap(current_feature_ids);
    }
    return is_keyframe;
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    for (auto& item : features_) {
        item.second.monoTriangulate(state, ric, tic, min_translation_, min_depth_, max_depth_);
    }
}

void FeatureManager::removeFrameObservations(
    int removed_frame_index, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    if (state.latest_frame_index > 0) {
        std::erase_if(features_, [&](const auto& item) {
            return item.second.host_frame_index == removed_frame_index &&
                   item.second.observations.size() == 1;
        });
        for (auto& item : features_) {
            Feature& feature = item.second;
            feature.removeFrame(removed_frame_index, state, ric, tic);
            if (feature.host_frame_index > removed_frame_index) {
                --feature.host_frame_index;
            }
        }

        std::erase_if(
            features_, [&](const auto& item) { return item.second.observations.empty(); });
    }
}

void FeatureManager::replaceRetainedHost(
    int old_host_index, int new_host_index, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    std::erase_if(features_, [&](auto& item) {
        Feature& feature = item.second;
        if (feature.host_frame_index != old_host_index) {
            return false;
        }
        if (!feature.transferHost(new_host_index, state, ric, tic)) {
            return true;
        }
        feature.observations.erase(feature.observations.begin() + 1);
        feature.host_frame_index = 0;
        return feature.observations.size() < 2;
    });

    for (auto& [_, feature] : features_) {
        if (feature.host_frame_index > old_host_index) {
            --feature.host_frame_index;
        }
    }
    std::erase_if(features_, [](const auto& item) { return item.second.observations.empty(); });
}

void FeatureManager::removeNewestFrameObservations(int frame_index) {
    for (auto& item : features_) {
        item.second.removeFrameObservation(frame_index);
    }

    std::erase_if(features_, [&](const auto& item) { return item.second.observations.empty(); });
}

void FeatureManager::removeOutliers(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (!state.camera) {
        return;
    }

    std::erase_if(features_, [&](const auto& item) {
        const Feature& feature = item.second;
        if (!canUseFeature(feature, tracked_times_thres_)) {
            return false;
        }

        const int host_frame_index = feature.host_frame_index;
        if (host_frame_index < 0 || host_frame_index > state.latest_frame_index) {
            return true;
        }

        const std::vector<FeaturePerFrame>& observations = feature.observations;
        Eigen::Vector3d world_point;
        if (!hostPointToWorld(
                state.frames[host_frame_index], observations.front().uv, feature.estimated_depth,
                observations.front().sync_delay, state.delay_time, ric, tic, world_point)) {
            return true;
        }

        double error_sum = 0.0;
        for (size_t k = 1; k < observations.size(); ++k) {
            const int target_frame_index = feature.observationFrameIndex(k);
            if (target_frame_index < 0 || target_frame_index > state.latest_frame_index) {
                return true;
            }
            double reprojection_error = 0.0;
            if (!computeReprojectionError(
                    state.frames[target_frame_index], world_point, observations[k],
                    state.delay_time, ric, tic, *state.camera, reprojection_error)) {
                return true;
            }
            error_sum += reprojection_error;
        }

        const size_t target_count = observations.size() - 1;
        return error_sum / static_cast<double>(target_count) > reproj_err_thres_;
    });
}

void FeatureManager::reset() {
    features_.clear();
    latest_keyframe_feature_ids_.clear();
}

std::vector<MarginalizedFeatureObservation> FeatureManager::collectMarginalizedObservations(
    int host_frame_index, int target_frame_index) {
    std::vector<MarginalizedFeatureObservation> result;
    for (auto& item : features_) {
        auto& feature = item.second;
        if (feature.host_frame_index != host_frame_index ||
            !canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }
        const int observation_index = target_frame_index - feature.host_frame_index;
        if (observation_index <= 0 ||
            observation_index >= static_cast<int>(feature.observations.size())) {
            continue;
        }
        feature.has_been_marginalized = true;
        result.push_back({&feature, target_frame_index});
    }
    return result;
}

std::vector<HostLandmark> FeatureManager::exportHostLandmarks(
    int host_frame_index, const State& state) const {
    std::vector<HostLandmark> landmarks;
    if (host_frame_index < 0 || host_frame_index > state.latest_frame_index) {
        return landmarks;
    }

    for (const auto& [feature_id, feature] : features_) {
        if (feature.host_frame_index != host_frame_index || feature.observations.empty() ||
            !std::isfinite(feature.estimated_depth) || feature.estimated_depth < min_depth_ ||
            feature.estimated_depth > max_depth_) {
            continue;
        }

        const FeaturePerFrame& host_observation = feature.observations.front();
        if (!host_observation.uv.allFinite() || !std::isfinite(host_observation.pt.x) ||
            !std::isfinite(host_observation.pt.y)) {
            continue;
        }

        landmarks.push_back(
            {feature_id, host_observation.pt, host_observation.uv, feature.estimated_depth});
    }
    return landmarks;
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
            const int frame_index = feature.observationFrameIndex(observation_index);
            if (frame_index > state.latest_frame_index) {
                throw std::logic_error("SFM observation index is outside the active window");
            }
            Eigen::Vector2d uv_norm(observation.uv(0), observation.uv(1));
            sfm_f.observation.emplace_back(frame_index, uv_norm);
        }
        sfm_features.push_back(std::move(sfm_f));
    }
    return sfm_features;
}

}  // namespace tassel_core
