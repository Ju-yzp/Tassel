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
#include <sophus/so3.hpp>

// tassel
#include "cam/camera_base.h"
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
    const int obs_num = static_cast<int>(feature.observations.size());
    return feature.estimated_depth != INVALID_DEPTH && obs_num >= 2 &&
           (feature.has_been_used || obs_num >= tracked_times_thres);
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

    for (const auto& [id, per_frame_feature] : feature_frame) {
        auto it = features_.find(id);
        if (it != features_.end()) {
            ++parallax_num;
            parallax_sum +=
                computeParallax((*it).second.observations.back().pt, per_frame_feature.pt);
            it->second.observations.emplace_back(per_frame_feature);
        } else {
            Feature feature(frame_count, 15);
            feature.observations.emplace_back(per_frame_feature);
            features_.emplace(id, std::move(feature));
        }
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

void FeatureManager::removeOutliers(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (!state.camera) {
        return;
    }

    std::set<int> removed_ids;
    for (auto& [id, feature] : features_) {
        double depth = feature.estimated_depth;
        std::vector<FeaturePerFrame>& observations = feature.observations;
        if (depth == INVALID_DEPTH ||
            static_cast<int>(observations.size()) < tracked_times_thres_) {
            continue;
        }

        int start_frame_id = feature.start_frame_id;

        double error_sum = 0.0;
        for (size_t k = 1; k < observations.size(); ++k) {
            int j = start_frame_id + static_cast<int>(k);
            const double dt_i = state.delay_time - observations[0].applied_delay;
            const double dt_j = state.delay_time - observations[k].applied_delay;
            const Eigen::Matrix3d A_i =
                Sophus::SO3d::exp(
                    (state.gyro_vec[start_frame_id] - state.Bgs[start_frame_id]) * dt_i)
                    .matrix();
            const Eigen::Matrix3d A_j =
                Sophus::SO3d::exp((state.Bgs[j] - state.gyro_vec[j]) * dt_j).matrix();

            const Eigen::Vector3d pi_in_C = observations[0].uv * depth;
            const Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
            const Eigen::Vector3d pi_in_G =
                state.Rs[start_frame_id] * A_i * pi_in_I + state.Ps[start_frame_id] +
                state.Vs[start_frame_id] * dt_i +
                0.5 * state.Rs[start_frame_id] *
                    (state.acc_vec[start_frame_id] - state.Bas[start_frame_id]) * dt_i * dt_i;
            const Eigen::Vector3d pj_in_I =
                A_j * state.Rs[j].transpose() *
                (pi_in_G - (state.Ps[j] + state.Vs[j] * dt_j +
                            0.5 * state.Rs[j] * (state.acc_vec[j] - state.Bas[j]) * dt_j * dt_j));
            const Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);

            const double inv_z = 1.0 / pj_in_C.z();
            const Eigen::Vector2d uv_pred_norm(pj_in_C.x() * inv_z, pj_in_C.y() * inv_z);
            const Eigen::Vector2d uv_pixel = state.camera->distort(uv_pred_norm);
            Eigen::Vector2d pt_meas(observations[k].pt.x, observations[k].pt.y);
            error_sum += (uv_pixel - pt_meas).norm();
        }

        double average_error = error_sum / static_cast<double>(observations.size() - 1);
        if (average_error > reproj_err_thres_) {
            removed_ids.insert(id);
        }
    }

    std::erase_if(features_, [&](const auto& item) { return removed_ids.count(item.first) > 0; });
}

void FeatureManager::reset() { features_.clear(); }

std::vector<Feature*> FeatureManager::collectMargFeatures() {
    std::vector<Feature*> result;
    for (auto& item : features_) {
        auto& feature = item.second;
        if (feature.start_frame_id != 0 || !canUseFeature(feature, tracked_times_thres_)) {
            continue;
        }
        feature.has_been_used = true;
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
        feature.has_been_used = true;
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
