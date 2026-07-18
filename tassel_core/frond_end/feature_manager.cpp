// eigen
#include <Eigen/Core>
#include <Eigen/SVD>

// cpp
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

// opencv
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

// tassel
#include "cam/camera_base.h"
#include "factor/visual_factor.h"
#include "feature.h"
#include "feature_manager.h"
#include "initial/initial_sfm.h"
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

bool FeatureManager::checkParallax(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    double keyframe_parallax_sum = 0.0;
    size_t keyframe_parallax_count = 0;
    last_input_stats_ = {};
    last_input_stats_.input_count = feature_frame.size();

    for (const auto& [id, per_frame_feature] : feature_frame) {
        if (latest_keyframe_feature_ids_.count(id) > 0) {
            ++last_input_stats_.connected_to_keyframe_count;
        }
        FeaturePerFrame observation = per_frame_feature;
        observation.frame_id = frame_id;
        auto it = features_.find(id);
        if (it != features_.end()) {
            ++last_input_stats_.matched_count;
            if (latest_keyframe_feature_ids_.count(id) > 0) {
                const auto keyframe_observation = std::find_if(
                    it->second.observations.begin(), it->second.observations.end(),
                    [&](const auto& obs) { return obs.frame_id == latest_keyframe_id_; });
                if (keyframe_observation != it->second.observations.end()) {
                    keyframe_parallax_sum +=
                        computeParallax(keyframe_observation->pt, per_frame_feature.pt);
                    ++keyframe_parallax_count;
                }
            }
            it->second.observations.emplace_back(std::move(observation));
        } else {
            ++last_input_stats_.created_count;
            Feature feature(frame_id, 15);
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
    if (!latest_keyframe_feature_ids_.empty()) {
        last_input_stats_.keyframe_feature_retention_ratio =
            static_cast<double>(last_input_stats_.connected_to_keyframe_count) /
            static_cast<double>(latest_keyframe_feature_ids_.size());
    }

    return !hasLatestKeyframe() ||
           keyframe_parallax_count < static_cast<size_t>(min_tracked_pts_) ||
           last_input_stats_.average_parallax > parallax_thres_;
}

void FeatureManager::acceptKeyframe(
    tassel_utils::FrameId frame_id, const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    latest_keyframe_id_ = frame_id;
    keyframe_ids_.insert(frame_id);
    latest_keyframe_feature_ids_.clear();
    latest_keyframe_feature_ids_.reserve(feature_frame.size());
    for (const auto& [feature_id, _] : feature_frame) {
        latest_keyframe_feature_ids_.insert(feature_id);
    }
}

void FeatureManager::retireKeyframe(tassel_utils::FrameId frame_id) {
    keyframe_ids_.erase(frame_id);
    if (latest_keyframe_id_ == frame_id) {
        latest_keyframe_id_ = tassel_utils::kInvalidFrameId;
        latest_keyframe_feature_ids_.clear();
    }
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    for (auto& item : features_) {
        item.second.monoTriangulate(state, ric, tic, min_translation_, min_depth_, max_depth_);
    }
}

void FeatureManager::removeOldest(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (state.cur_frame_count > 0) {
        removeFrame(state.frame_ids[0], state, ric, tic);
    }
}

void FeatureManager::removeFrame(
    tassel_utils::FrameId removed_frame_id, const State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    if (state.cur_frame_count > 0) {
        retireKeyframe(removed_frame_id);
        std::erase_if(features_, [&](const auto& item) {
            return item.second.host_frame_id == removed_frame_id &&
                   item.second.observations.size() == 1;
        });
        for (auto& item : features_) {
            item.second.removeFrame(removed_frame_id, state, ric, tic);
        }

        std::erase_if(
            features_, [&](const auto& item) { return item.second.observations.empty(); });
    }
}

void FeatureManager::replaceHost(
    tassel_utils::FrameId old_host_frame_id, tassel_utils::FrameId new_host_frame_id,
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    std::erase_if(features_, [&](auto& item) {
        Feature& feature = item.second;
        if (feature.host_frame_id != old_host_frame_id) return false;
        if (!feature.transferHost(new_host_frame_id, state, ric, tic)) return true;
        feature.removeFrameObservation(old_host_frame_id);
        return feature.observations.size() < 2;
    });

    for (auto& [_, feature] : features_) {
        if (feature.host_frame_id != old_host_frame_id) {
            feature.removeFrameObservation(old_host_frame_id);
        }
    }
    std::erase_if(features_, [](const auto& item) { return item.second.observations.empty(); });
}

void FeatureManager::removeNewest(tassel_utils::FrameId frame_id) {
    retireKeyframe(frame_id);
    for (auto& item : features_) {
        item.second.removeFrameObservation(frame_id);
    }

    std::erase_if(features_, [&](const auto& item) { return item.second.observations.empty(); });
}

OutlierStats FeatureManager::removeOutliers(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    OutlierStats stats;
    if (!state.camera) {
        return stats;
    }

    std::set<int> removed_ids;
    for (auto& [id, feature] : features_) {
        double depth = feature.estimated_depth;
        std::vector<FeaturePerFrame>& observations = feature.observations;
        if (!canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }

        const int host_slot = state.findFrameSlot(feature.host_frame_id);
        if (host_slot < 0) continue;

        double error_sum = 0.0;
        size_t valid_observation_count = 0;
        for (size_t k = 1; k < observations.size(); ++k) {
            const int target_slot = state.findFrameSlot(observations[k].frame_id);
            if (target_slot < 0) continue;
            Eigen::Vector2d pt_meas(observations[k].pt.x, observations[k].pt.y);
            VisualFactor factor(
                observations[0].uv, pt_meas, ric, tic, state.gyro_vec[host_slot],
                state.gyro_vec[target_slot], state.acc_vec[host_slot], state.acc_vec[target_slot],
                state.params_speed_bias[host_slot].data(),
                state.params_speed_bias[target_slot].data(),
                state.params_speed_bias[host_slot].data() + 6,
                state.params_speed_bias[target_slot].data() + 6,
                state.params_speed_bias[host_slot].data() + 3,
                state.params_speed_bias[target_slot].data() + 3, Eigen::Matrix2d::Identity(),
                state.camera, observations[0].applied_delay, observations[k].applied_delay);
            const double inv_depth = 1.0 / depth;
            const double* parameters[] = {
                state.params_pose[host_slot].data(), state.params_pose[target_slot].data(),
                &state.param_delay_time, &inv_depth};
            Eigen::Vector2d residual;
            factor.Evaluate(parameters, residual.data(), nullptr);
            error_sum += residual.norm();
            ++valid_observation_count;
        }

        if (valid_observation_count == 0) continue;
        double average_error = error_sum / static_cast<double>(valid_observation_count);
        ++stats.checked_count;
        stats.average_error += average_error;
        stats.maximum_error = std::max(stats.maximum_error, average_error);
        if (average_error > reproj_err_thres_) {
            removed_ids.insert(id);
            stats.removed_average_error += average_error;
        }
    }

    std::erase_if(features_, [&](const auto& item) { return removed_ids.count(item.first) > 0; });
    stats.removed_count = removed_ids.size();
    if (stats.checked_count > 0) {
        stats.average_error /= static_cast<double>(stats.checked_count);
    }
    if (stats.removed_count > 0) {
        stats.removed_average_error /= static_cast<double>(stats.removed_count);
    }
    return stats;
}

void FeatureManager::reset() {
    features_.clear();
    latest_keyframe_id_ = tassel_utils::kInvalidFrameId;
    keyframe_ids_.clear();
    latest_keyframe_feature_ids_.clear();
    last_input_stats_ = {};
}

std::vector<MarginalizedFeatureObservation> FeatureManager::collectMarginalizedObservations(
    tassel_utils::FrameId host_frame_id, tassel_utils::FrameId target_frame_id) {
    std::vector<MarginalizedFeatureObservation> result;
    for (auto& item : features_) {
        auto& feature = item.second;
        if (feature.host_frame_id != host_frame_id ||
            !canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }
        const auto target = std::find_if(
            feature.observations.begin(), feature.observations.end(),
            [&](const auto& observation) { return observation.frame_id == target_frame_id; });
        if (target == feature.observations.end()) continue;
        feature.has_been_marginalized = true;
        result.push_back({&feature, target_frame_id});
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
        const int host_slot = cs.findFrameSlot(feature.host_frame_id);
        if (host_slot < 0) continue;
        Eigen::Vector3d pt_in_C = feature.observations[0].uv * feature.estimated_depth;
        Eigen::Vector3d pt_in_I = ric * pt_in_C + tic;
        Eigen::Vector3d pt_in_W = cs.Rs[host_slot] * pt_in_I + cs.Ps[host_slot];
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
        for (const auto& observation : feature.observations) {
            const int frame_slot = state.findFrameSlot(observation.frame_id);
            if (frame_slot < 0) continue;
            Eigen::Vector2d uv_norm(observation.uv(0), observation.uv(1));
            sfm_f.observation.emplace_back(frame_slot, uv_norm);
        }
        sfm_features.push_back(std::move(sfm_f));
    }
    return sfm_features;
}

}  // namespace tassel_core
