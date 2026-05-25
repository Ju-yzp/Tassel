// eigen
#include <Eigen/Core>
#include <Eigen/SVD>

// cpp
#include <set>
#include <vector>

// opencv
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

// tassel
#include "feature.h"
#include "feature_manager.h"
#include "state/state.h"

#include <spdlog/spdlog.h>

namespace tassel_core {

namespace {
inline double computeParallax(const cv::Point2f& p1, const cv::Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}
}  // namespace

FeatureManager::FeatureManager(
    double reprojection_error_thres, double parallax_thres, int tracked_times_thres,
    int min_tracked_pts_num, int min_pnp_pt_num, double min_pnp_inliers_ratio,
    double min_translation, double min_depth, double max_depth)
    : reprojection_error_thres_(reprojection_error_thres),
      parallax_thres_(parallax_thres),
      tracked_times_thres_(tracked_times_thres),
      min_tracked_pts_num_(min_tracked_pts_num),
      min_pnp_pt_num_(min_pnp_pt_num),

      min_pnp_inliers_ratio_(min_pnp_inliers_ratio),
      min_translation_(min_translation),
      min_depth_(min_depth),
      max_depth_(max_depth) {
    features_.reserve(1000);
}

bool FeatureManager::checkKeyFrameByParallax(
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
            Feature feature(frame_count, max_depth_ > 3.0 ? 15 : 15);
            feature.observations.emplace_back(per_frame_feature);
            features_[id] = feature;
        }
    }

    // spdlog::info("Current frame track {}.f old feature from last frame", parallax_num);
    return (
        parallax_num == 0 || (parallax_sum / parallax_num) > parallax_thres_ ||
        parallax_num < min_tracked_pts_num_);
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1) {
    int frame_count = state.cur_frame_count;
    bool mono_triangulate = frame_count > 0;
    for (auto& [id, feature] : features_) {
        feature.stereoTriangulate(ric, tic, ric1, tic1, min_depth_, max_depth_);
        if (mono_triangulate) {
            feature.monoTriangulate(state, ric, tic, min_translation_, min_depth_, max_depth_);
        }
    }
}

void FeatureManager::initPoseByPNP(
    State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    const int frame_count = state.cur_frame_count;

    if (frame_count <= 0) {
        return;
    }

    state.Rs[frame_count] = state.Rs[frame_count - 1];
    state.Ps[frame_count] = state.Ps[frame_count - 1];
    std::vector<cv::Point3f> object_pts;
    std::vector<cv::Point2f> normalize_pts;
    std::vector<size_t> candidate_ids;
    for (const auto& [id, feature] : features_) {
        int start_frame_id = feature.start_frame_id;
        int observation_num = static_cast<int>(feature.observations.size());
        double depth = feature.estimated_depth;
        int obs_idx = frame_count - start_frame_id;
        if (obs_idx >= 0 && obs_idx < observation_num && depth != INVALID_DEPTH) {
            Eigen::Vector3d p_in_I = ric * feature.observations[0].uv * depth + tic;
            Eigen::Vector3d p_in_W = state.Rs[start_frame_id] * p_in_I + state.Ps[start_frame_id];
            object_pts.push_back(cv::Point3f(p_in_W(0), p_in_W(1), p_in_W(2)));
            Eigen::Vector3d uv = feature.observations[obs_idx].uv;
            normalize_pts.push_back(cv::Point2f(uv(0), uv(1)));
            candidate_ids.push_back(id);
        }
    }

    if (static_cast<int>(object_pts.size()) < min_pnp_pt_num_) {
        spdlog::error(
            "Not enough points for PnP. Only {} points.", static_cast<int>(object_pts.size()));
        return;
    }

    Eigen::Matrix3d guess_R = state.Rs[frame_count - 1] * ric;
    Eigen::Vector3d guess_P = state.Rs[frame_count - 1] * tic + state.Ps[frame_count - 1];

    cv::Mat R_cv, rvec, tvec;
    guess_R.transposeInPlace();
    guess_P = (-guess_R * guess_P).eval();
    cv::eigen2cv(guess_R, R_cv);
    cv::eigen2cv(guess_P, tvec);
    cv::Rodrigues(R_cv, rvec);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(
        object_pts, normalize_pts, K, cv::Mat(), rvec, tvec, true, 30, reprojection_error_thres_,
        min_pnp_inliers_ratio_, inliers, cv::SOLVEPNP_EPNP);

    if (success) {
        cv::Mat R_result_cv;
        cv::Rodrigues(rvec, R_result_cv);
        cv::cv2eigen(R_result_cv, guess_R);
        cv::cv2eigen(tvec, guess_P);

        guess_R.transposeInPlace();
        guess_P = (-guess_R * guess_P).eval();

        Eigen::Matrix3d R_candidate = guess_R * ric.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            R_candidate, Eigen::ComputeFullU | Eigen::ComputeFullV);
        state.Rs[frame_count] = svd.matrixU() * svd.matrixV().transpose();
        state.Ps[frame_count] = guess_P - guess_R * tic;
        spdlog::info("PNP success");
    } else {
        spdlog::error(
            "PnP failed,inliers ratio:{}",
            static_cast<double>(inliers.size()) / static_cast<double>(object_pts.size()));
    }
}

void FeatureManager::removeOldest(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (state.cur_frame_count > 1) {
        Eigen::Matrix3d prev_r = state.Rs[0];
        Eigen::Vector3d prev_t = state.Ps[0];
        Eigen::Matrix3d cur_r = state.Rs[1];
        Eigen::Vector3d cur_t = state.Ps[1];

        std::erase_if(features_, [&](const auto& item) {
            return item.second.start_frame_id == 0 && item.second.observations.size() == 1;
        });
        for (auto& [id, feature] : features_) {
            feature.removeOldest(prev_r, prev_t, cur_r, cur_t, ric, tic);
        }

        std::erase_if(
            features_, [&](const auto& item) { return item.second.observations.empty(); });
    }
}

void FeatureManager::removeNewest(size_t frame_count) {
    for (auto& [id, feature] : features_) {
        feature.removeNewest(frame_count);
    }

    std::erase_if(features_, [&](const auto& item) { return item.second.observations.empty(); });
}

void FeatureManager::removeOutliers(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    std::set<int> removed_ids;
    for (auto& [id, feature] : features_) {
        double depth = feature.estimated_depth;
        std::vector<FeaturePerFrame>& observations = feature.observations;
        if (depth == INVALID_DEPTH ||
            static_cast<int>(observations.size()) < tracked_times_thres_) {
            continue;
        }

        int start_frame_id = feature.start_frame_id;
        Eigen::Matrix3d R_i = state.Rs[start_frame_id];
        Eigen::Vector3d P_i = state.Ps[start_frame_id];
        Eigen::Vector3d pi_in_C = depth * observations[0].uv;
        Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
        Eigen::Vector3d pi_in_W = R_i * pi_in_I + P_i;

        double cosine_sum = 0.0;
        for (size_t k = 1; k < observations.size(); ++k) {
            int j = start_frame_id + static_cast<int>(k);
            Eigen::Matrix3d R_j = state.Rs[j];
            Eigen::Vector3d P_j = state.Ps[j];
            Eigen::Vector3d pj_in_I = R_j.transpose() * (pi_in_W - P_j);
            Eigen::Vector3d pj_in_C = ric.transpose() * (pj_in_I - tic);
            double cos_angle = pj_in_C.normalized().dot(observations[k].uv.normalized());
            cosine_sum += cos_angle;
        }

        double average_cosine = cosine_sum / static_cast<double>((observations.size() - 1));
        if (average_cosine < reprojection_error_thres_) {
            removed_ids.insert(id);
        }
    }

    std::erase_if(features_, [&](const auto& item) { return removed_ids.count(item.first) > 0; });
}

void FeatureManager::reset() { features_.clear(); }

std::vector<Feature> FeatureManager::collectMarginalizationFeatures() {
    std::vector<Feature> result;
    for (auto& [id, feature] : features_) {
        if (feature.start_frame_id != 0) continue;
        if (feature.estimated_depth == INVALID_DEPTH) continue;
        if (static_cast<int>(feature.observations.size()) < tracked_times_thres_) continue;
        result.push_back(feature);
    }
    return result;
}

void FeatureManager::removeMarginalizedFeatures() {
    std::erase_if(features_, [&](const auto& item) { return item.second.start_frame_id == 0; });
}

std::vector<Eigen::Vector3d> FeatureManager::getMonoPointCloud(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) const {
    std::vector<Eigen::Vector3d> points;
    for (const auto& [id, feature] : features_) {
        if (feature.tri_source != TriangulationSource::Monocular) continue;
        if (feature.estimated_depth <= 0) continue;
        int start_frame_id = feature.start_frame_id;
        if (start_frame_id >= state.cur_frame_count) continue;
        Eigen::Vector3d pt_in_C = feature.observations[0].uv * feature.estimated_depth;
        Eigen::Vector3d pt_in_I = ric * pt_in_C + tic;
        Eigen::Vector3d pt_in_W = state.Rs[start_frame_id] * pt_in_I + state.Ps[start_frame_id];
        points.push_back(pt_in_W);
    }
    return points;
}

std::vector<Eigen::Vector3d> FeatureManager::getStereoPointCloud(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) const {
    std::vector<Eigen::Vector3d> points;
    for (const auto& [id, feature] : features_) {
        if (feature.tri_source != TriangulationSource::Stereo) continue;
        if (feature.estimated_depth <= 0) continue;
        int start_frame_id = feature.start_frame_id;
        if (start_frame_id >= state.cur_frame_count) continue;
        Eigen::Vector3d pt_in_C = feature.observations[0].uv * feature.estimated_depth;
        Eigen::Vector3d pt_in_I = ric * pt_in_C + tic;
        Eigen::Vector3d pt_in_W = state.Rs[start_frame_id] * pt_in_I + state.Ps[start_frame_id];
        points.push_back(pt_in_W);
    }
    return points;
}

std::vector<Feature*> FeatureManager::collectOptimizedFeatures() {
    std::vector<Feature*> result;
    for (auto& [id, feature] : features_) {
        if (feature.estimated_depth == INVALID_DEPTH ||
            static_cast<int>(feature.observations.size()) < tracked_times_thres_) {
            continue;
        }
        result.push_back(&feature);
    }
    return result;
}
}  // namespace tassel_core
