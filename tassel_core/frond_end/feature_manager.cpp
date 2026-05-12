// eigen
#include <Eigen/Core>

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
    int max_pnp_needed_num, int min_pnp_pt_num, double min_pnp_inliers_ratio,
    double min_translation, double min_depth, double max_depth)
    : reprojection_error_thres_(reprojection_error_thres),
      parallax_thres_(parallax_thres),
      tracked_times_thres_(tracked_times_thres),
      min_pnp_pt_num_(min_pnp_pt_num),
      min_pnp_inliers_ratio_(min_pnp_inliers_ratio),
      min_translation_(min_translation),
      min_depth_(min_depth),
      max_depth_(max_depth),
      max_pnp_needed_num_(max_pnp_needed_num) {
    if (min_pnp_pt_num >= max_pnp_needed_num) {
        throw std::runtime_error(
            std::string("[") + __FUNCTION__ +
            "] Input 'min_pnp_pt_num' must be less than 'max_pnp_needed_num'.");
    }
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

    return (parallax_num == 0 || (parallax_sum / parallax_num) > parallax_thres_);
}

void FeatureManager::triangulate(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1) {
    int frame_count = state.cur_frame_count;
    bool mono_triangulate = frame_count > 0;
    for (auto& [id, feature] : features_) {
        feature.stereoTriangulate(ric, tic, ric1, tic1, min_depth_, max_depth_);
        if (mono_triangulate) {
            feature.monoTriangulate(state, ric, tic, min_depth_, max_depth_);
        }
    }
}

void FeatureManager::initPoseByPNP(
    State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    const int frame_count = state.cur_frame_count;

    if (frame_count <= 0) {
        return;
    }

    Pose prev_pose = state.poses[frame_count - 1].get_optimized_pose();
    state.poses[frame_count].set_optimized_pose(prev_pose);
    std::vector<cv::Point3f> object_pts;
    std::vector<cv::Point2f> normalize_pts;
    std::vector<size_t> candidate_ids;
    for (const auto& [id, feature] : features_) {
        int start_frame_id = feature.start_frame_id;
        int observation_num = static_cast<int>(feature.observations.size());
        double depth = feature.estimated_depth;
        if (frame_count + 1 == observation_num && depth != INVALID_DEPTH) {
            Eigen::Vector3d p_in_I = ric * feature.observations[0].uv * depth + tic;
            Eigen::Vector3d p_in_W =
                state.poses[start_frame_id].get_pose().rotationMatrix() * p_in_I +
                state.poses[start_frame_id].get_pose().translation();
            object_pts.push_back(cv::Point3f(p_in_W(0), p_in_W(1), p_in_W(2)));
            Eigen::Vector3d uv = feature.observations[frame_count].uv;
            normalize_pts.push_back(cv::Point2f(uv(0), uv(1)));
            candidate_ids.push_back(id);
            if (static_cast<int>(object_pts.size()) >= max_pnp_needed_num_) {
                break;
            }
        }
    }

    if (static_cast<int>(object_pts.size()) < min_pnp_pt_num_) {
        return;
    }

    Eigen::Matrix3d guess_R = state.poses[frame_count - 1].get_pose().rotationMatrix() * ric;
    Eigen::Vector3d guess_P = state.poses[frame_count - 1].get_pose().rotationMatrix() * tic +
                              state.poses[frame_count - 1].get_pose().translation();

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

        Pose final_pose(guess_R * ric.transpose(), guess_P - guess_R * tic);
        state.poses[frame_count].set_optimized_pose(final_pose);
    }
}

void FeatureManager::removeOldest(
    const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    if (state.cur_frame_count > 0) {
        Pose prev_pose = state.poses[state.cur_frame_count - 1].get_optimized_pose();
        Pose cur_pose = state.poses[state.cur_frame_count].get_optimized_pose();
        Eigen::Matrix3d prev_r = prev_pose.rotationMatrix();
        Eigen::Vector3d prev_t = prev_pose.translation();
        Eigen::Matrix3d cur_r = cur_pose.rotationMatrix();
        Eigen::Vector3d cur_t = cur_pose.translation();

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
        Eigen::Matrix3d R_i = state.poses[start_frame_id].get_pose().rotationMatrix();
        Eigen::Vector3d P_i = state.poses[start_frame_id].get_pose().translation();
        Eigen::Vector3d pi_in_C = depth * observations[0].uv;
        Eigen::Vector3d pi_in_I = ric * pi_in_C + tic;
        Eigen::Vector3d pi_in_W = R_i * pi_in_I + P_i;

        double cosine_sum = 0.0;
        for (size_t k = 1; k < observations.size(); ++k) {
            int j = start_frame_id + static_cast<int>(k);
            Eigen::Matrix3d R_j = state.poses[j].get_pose().rotationMatrix();
            Eigen::Vector3d P_j = state.poses[j].get_pose().translation();
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

std::vector<std::pair<int, const Feature*>> FeatureManager::collectMarginalizationFeatures() const {
    std::vector<std::pair<int, const Feature*>> result;
    for (const auto& [id, feature] : features_) {
        if (feature.start_frame_id != 0) continue;
        if (feature.estimated_depth == INVALID_DEPTH) continue;
        if (static_cast<int>(feature.observations.size()) < tracked_times_thres_) continue;
        result.emplace_back(id, &feature);
    }
    return result;
}

std::vector<Feature*> FeatureManager::collectOptimizationFeatures() {
    std::vector<Feature*> result;
    for (auto& [id, feature] : features_) {
        if (feature.estimated_depth == INVALID_DEPTH) continue;
        if (static_cast<int>(feature.observations.size()) < tracked_times_thres_) continue;
        result.push_back(&feature);
    }
    return result;
}
}  // namespace tassel_core
