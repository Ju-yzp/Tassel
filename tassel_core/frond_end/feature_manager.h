#ifndef TASSEL_CORE_FEATURE_MANAGER_H_
#define TASSEL_CORE_FEATURE_MANAGER_H_

// 标准库
#include <Eigen/Core>
#include <unordered_map>
#include <vector>

// Tassel
#include "feature.h"

namespace tassel_core {

struct State;
struct SFMFeature;

struct FeatureInputStats {
    size_t input_count = 0;
    size_t matched_count = 0;
    size_t created_count = 0;
    size_t connected_to_keyframe_count = 0;
    double average_parallax = 0.0;
    double current_keyframe_connection_ratio = 0.0;
    double keyframe_feature_retention_ratio = 0.0;
};

class FeatureManager {
public:
    FeatureManager(
        double reproj_err_thres, double parallax_thres, int tracked_times_thres,
        int min_tracked_pts, double min_translation, double min_depth = MIN_DISTANCE,
        double max_depth = MAX_DISTANCE);

    bool checkParallax(
        int frame_slot, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    inline void resetLandmarkDepths() {
        for (auto& item : features_) {
            item.second.estimated_depth = INVALID_DEPTH;
        }
    }

    void triangulate(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeOldestFrameObservations(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeFrameObservations(
        int frame_slot, const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void replaceRetainedHost(
        int old_host_slot, int new_host_slot, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void removeNewestFrameObservations(int frame_slot);

    void removeOutliers(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    const FeatureInputStats& lastInputStats() const { return last_input_stats_; }

    void logInputStats(bool is_keyframe) const;

    bool hasLatestKeyframe() const { return !latest_keyframe_features_.empty(); }
    void acceptKeyframe(const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    void reset();

    std::vector<Feature*> collectLandmarks();

    std::vector<MarginalizedFeatureObservation> collectMarginalizedObservations(
        int host_slot, int target_slot);

    std::vector<Eigen::Vector3d> getPointCloud(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) const;

    std::vector<SFMFeature> collectSFMFeatures(const State& state) const;

    std::unordered_map<int, Feature>& features() { return features_; }

private:
    double reproj_err_thres_;

    double parallax_thres_;

    int tracked_times_thres_;

    int min_tracked_pts_;

    double min_translation_;

    double min_depth_, max_depth_;

    FeatureInputStats last_input_stats_;

    std::unordered_map<int, cv::Point2f> latest_keyframe_features_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
