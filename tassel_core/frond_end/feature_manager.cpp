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
    double reprojection_error_thres, double pnp_reprojection_error_thres, double parallax_thres,
    int tracked_times_thres, int min_tracked_pts_num, int min_pnp_pt_num,
    double min_pnp_inliers_ratio, double min_translation, double min_depth, double max_depth)
    : reprojection_error_thres_(reprojection_error_thres),
      pnp_reprojection_error_thres_(pnp_reprojection_error_thres),
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

    return (
        parallax_num == 0 || (parallax_sum / parallax_num) > parallax_thres_ ||
        parallax_num < min_tracked_pts_num_);
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1) {
    bool mono_triangulate = state.cur_frame_count > 0;
    auto cs = state.get_compensated_state();
    for (auto& [id, feature] : features_) {
        feature.stereoTriangulate(state.ric, state.tic, ric1, tic1, min_depth_, max_depth_);
        if (mono_triangulate) {
            feature.monoTriangulate(
                cs, state.ric, state.tic, min_translation_, min_depth_, max_depth_);
        }
    }
}

bool FeatureManager::initPoseByPNP(
    int frame_count, std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps,
    Eigen::Matrix3d ric, Eigen::Vector3d tic) {
    if (frame_count <= 0) {
        return true;
    }

    std::vector<cv::Point3f> object_pts;
    std::vector<cv::Point2f> normalize_pts;
    for (const auto& [id, feature] : features_) {
        int start_frame_id = feature.start_frame_id;
        if (start_frame_id < frame_count - 3) continue;
        double depth = feature.estimated_depth;
        int obs_idx = frame_count - start_frame_id;
        if (obs_idx < 1 || obs_idx >= static_cast<int>(feature.observations.size()) ||
            depth == INVALID_DEPTH) {
            continue;
        }
        Eigen::Vector3d p_in_C = feature.observations[0].uv * depth;
        Eigen::Vector3d p_in_I = ric * p_in_C + tic;
        Eigen::Vector3d p_in_W = Rs[start_frame_id] * p_in_I + Ps[start_frame_id];
        object_pts.emplace_back(p_in_W(0), p_in_W(1), p_in_W(2));
        Eigen::Vector3d uv = feature.observations[obs_idx].uv;
        normalize_pts.emplace_back(uv(0), uv(1));
    }

    if (static_cast<int>(object_pts.size()) < min_pnp_pt_num_) {
        spdlog::error(
            "Not enough points for PnP. Only {} points.", static_cast<int>(object_pts.size()));
        return false;
    }

    Eigen::Matrix3d guess_R = Rs[frame_count - 1] * ric;
    Eigen::Vector3d guess_P = Ps[frame_count - 1] + Rs[frame_count - 1] * tic;

    cv::Mat R_cv, rvec, tvec;
    guess_R.transposeInPlace();
    guess_P = (-guess_R * guess_P).eval();
    cv::eigen2cv(guess_R, R_cv);
    cv::eigen2cv(guess_P, tvec);
    cv::Rodrigues(R_cv, rvec);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(
        object_pts, normalize_pts, K, cv::Mat(), rvec, tvec, true, 30,
        pnp_reprojection_error_thres_, 0.90, inliers, cv::SOLVEPNP_EPNP);

    double inlier_ratio =
        static_cast<double>(inliers.size()) / static_cast<double>(object_pts.size());
    spdlog::info(
        "PNP: {} 3D pts, {} inliers ({:.1f}%)", object_pts.size(), inliers.size(),
        100.0 * inlier_ratio);

    if (success && inlier_ratio >= min_pnp_inliers_ratio_) {
        cv::Mat R_result_cv;
        cv::Rodrigues(rvec, R_result_cv);
        cv::cv2eigen(R_result_cv, guess_R);
        cv::cv2eigen(tvec, guess_P);

        guess_R.transposeInPlace();
        guess_P = (-guess_R * guess_P).eval();

        Eigen::Quaterniond q_opt(guess_R);
        q_opt.normalize();
        Rs[frame_count] = q_opt.toRotationMatrix() * ric.transpose();
        Ps[frame_count] = guess_P - Rs[frame_count] * tic;
        spdlog::info("PNP success (camera pose)");
        return true;
    } else {
        spdlog::error(
            "PnP failed (success={}), inlier ratio: {:.1f}% < {:.0f}%", success,
            100.0 * inlier_ratio, 100.0 * min_pnp_inliers_ratio_);
        return false;
    }
}

void FeatureManager::removeOldest(const State& state) {
    if (state.cur_frame_count > 1) {
        auto cs = state.get_compensated_state();
        std::erase_if(features_, [&](const auto& item) {
            return item.second.start_frame_id == 0 && item.second.observations.size() == 1;
        });
        for (auto& [id, feature] : features_) {
            feature.removeOldest(cs.Rs[0], cs.Ps[0], cs.Rs[1], cs.Ps[1], cs.ric, cs.tic);
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

void FeatureManager::removeOutliers(const State& state) {
    auto cs = state.get_compensated_state();
    if (!cs.camera) {
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
        Eigen::Vector3d pi_in_C = depth * observations[0].uv;
        Eigen::Vector3d pi_in_I = cs.ric * pi_in_C + cs.tic;
        Eigen::Vector3d pi_in_W = cs.Rs[start_frame_id] * pi_in_I + cs.Ps[start_frame_id];

        double error_sum = 0.0;
        for (size_t k = 1; k < observations.size(); ++k) {
            int j = start_frame_id + static_cast<int>(k);
            Eigen::Vector3d pj_in_I = cs.Rs[j].transpose() * (pi_in_W - cs.Ps[j]);
            Eigen::Vector3d pj_in_C = cs.ric.transpose() * (pj_in_I - cs.tic);

            double inv_z = 1.0 / pj_in_C.z();
            Eigen::Vector2d uv_norm(pj_in_C.x() * inv_z, pj_in_C.y() * inv_z);
            Eigen::Vector2d uv_pixel = cs.camera->distort(uv_norm);
            Eigen::Vector2d pt_meas(observations[k].pt.x, observations[k].pt.y);
            error_sum += (uv_pixel - pt_meas).norm();
        }

        double average_error = error_sum / static_cast<double>((observations.size() - 1));
        if (average_error > reprojection_error_thres_) {
            removed_ids.insert(id);
        }
    }

    std::erase_if(features_, [&](const auto& item) { return removed_ids.count(item.first) > 0; });
}

void FeatureManager::reset() { features_.clear(); }

std::vector<Feature*> FeatureManager::collectMarginalizationFeatures() {
    std::vector<Feature*> result;
    for (auto& [id, feature] : features_) {
        bool is_marginalized =
            !((feature.start_frame_id != 0) || (feature.estimated_depth == INVALID_DEPTH) ||
              (static_cast<int>(feature.observations.size()) < tracked_times_thres_));
        if (is_marginalized) {
            result.push_back(&feature);
        }
    }
    return result;
}

void FeatureManager::removeMarginalizedFeatures() {
    std::erase_if(features_, [&](const auto& item) { return item.second.start_frame_id == 0; });
}

std::vector<Eigen::Vector3d> FeatureManager::getPointCloud(const State& state) const {
    auto cs = state.get_compensated_state();
    std::vector<Eigen::Vector3d> points;
    for (const auto& [id, feature] : features_) {
        if (feature.estimated_depth <= 0) {
            continue;
        }
        int start_frame_id = feature.start_frame_id;
        if (start_frame_id >= cs.cur_frame_count) {
            continue;
        }
        Eigen::Vector3d pt_in_C = feature.observations[0].uv * feature.estimated_depth;
        Eigen::Vector3d pt_in_I = cs.ric * pt_in_C + cs.tic;
        Eigen::Vector3d pt_in_W = cs.Rs[start_frame_id] * pt_in_I + cs.Ps[start_frame_id];
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
        ++feature.optimized_count;
        result.push_back(&feature);
    }
    return result;
}
}  // namespace tassel_core
