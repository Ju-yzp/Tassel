// eigen
#include <Eigen/Core>
#include <Eigen/SVD>

// cpp
#include <cmath>
#include <set>
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
    features_.reserve(1000);
}

bool FeatureManager::checkParallax(
    size_t frame_count, const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    double parallax_sum = 0.0;
    double parallax_num = 0.0;
    last_input_stats_ = {};
    last_input_stats_.input_count = feature_frame.size();

    for (const auto& [id, per_frame_feature] : feature_frame) {
        auto it = features_.find(id);
        if (it != features_.end()) {
            ++last_input_stats_.matched_count;
            ++parallax_num;
            parallax_sum +=
                computeParallax((*it).second.observations.back().pt, per_frame_feature.pt);
            it->second.observations.emplace_back(per_frame_feature);
        } else {
            ++last_input_stats_.created_count;
            Feature feature(frame_count, 15);
            feature.observations.emplace_back(per_frame_feature);
            features_.emplace(id, std::move(feature));
        }
    }

    if (parallax_num > 0.0) {
        last_input_stats_.average_parallax = parallax_sum / parallax_num;
    }

    return (
        parallax_num == 0 || (parallax_sum / parallax_num) > parallax_thres_ ||
        parallax_num < min_tracked_pts_);
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    for (auto& item : features_) {
        item.second.monoTriangulate(state, ric, tic, min_translation_, min_depth_, max_depth_);
    }
}

void FeatureManager::removeOldest(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (state.cur_frame_count > 1) {
        std::erase_if(features_, [&](const auto& item) {
            return item.second.start_frame_id == 0 && item.second.observations.size() == 1;
        });
        for (auto& item : features_) {
            item.second.removeOldest(state, ric, tic);
        }

        std::erase_if(
            features_, [&](const auto& item) { return item.second.observations.empty(); });
    }
}

void FeatureManager::removeNewest(size_t frame_count) {
    for (auto& item : features_) {
        item.second.removeNewest(frame_count);
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

        int start_frame_id = feature.start_frame_id;

        double error_sum = 0.0;
        for (size_t k = 1; k < observations.size(); ++k) {
            int j = start_frame_id + static_cast<int>(k);
            Eigen::Vector2d pt_meas(observations[k].pt.x, observations[k].pt.y);
            VisualFactor factor(
                observations[0].uv, pt_meas, ric, tic, state.gyro_vec[start_frame_id],
                state.gyro_vec[j], state.acc_vec[start_frame_id], state.acc_vec[j],
                state.params_speed_bias[start_frame_id].data(), state.params_speed_bias[j].data(),
                state.params_speed_bias[start_frame_id].data() + 6,
                state.params_speed_bias[j].data() + 6,
                state.params_speed_bias[start_frame_id].data() + 3,
                state.params_speed_bias[j].data() + 3, Eigen::Matrix2d::Identity(), state.camera,
                observations[0].applied_delay, observations[k].applied_delay);
            const double inv_depth = 1.0 / depth;
            const double* parameters[] = {
                state.params_pose[start_frame_id].data(), state.params_pose[j].data(),
                &state.param_delay_time, &inv_depth};
            Eigen::Vector2d residual;
            factor.Evaluate(parameters, residual.data(), nullptr);
            error_sum += residual.norm();
        }

        double average_error = error_sum / static_cast<double>(observations.size() - 1);
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

void FeatureManager::reset() { features_.clear(); }

std::vector<Feature*> FeatureManager::collectMargFeatures() {
    std::vector<Feature*> result;
    for (auto& item : features_) {
        auto& feature = item.second;
        if (feature.start_frame_id != 0 || !canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }
        feature.has_been_marginalized = true;
        result.push_back(&feature);
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
        int start_frame_id = feature.start_frame_id;
        if (start_frame_id >= cs.cur_frame_count) {
            continue;
        }
        Eigen::Vector3d pt_in_C = feature.observations[0].uv * feature.estimated_depth;
        Eigen::Vector3d pt_in_I = ric * pt_in_C + tic;
        Eigen::Vector3d pt_in_W = cs.Rs[start_frame_id] * pt_in_I + cs.Ps[start_frame_id];
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

std::vector<SFMFeature> FeatureManager::collectSFMFeatures(int frame_num) const {
    std::vector<SFMFeature> sfm_features;
    sfm_features.reserve(features_.size());
    for (const auto& [id, feature] : features_) {
        SFMFeature sfm_f;
        sfm_f.state = false;
        sfm_f.id = id;
        sfm_f.position[0] = 0;
        sfm_f.position[1] = 0;
        sfm_f.position[2] = 0;
        for (size_t k = 0; k < feature.observations.size(); k++) {
            int abs_frame_id = static_cast<int>(feature.start_frame_id) + static_cast<int>(k);
            if (abs_frame_id >= frame_num) continue;
            Eigen::Vector2d uv_norm(feature.observations[k].uv(0), feature.observations[k].uv(1));
            sfm_f.observation.emplace_back(abs_frame_id, uv_norm);
        }
        sfm_features.push_back(std::move(sfm_f));
    }
    return sfm_features;
}

}  // namespace tassel_core
