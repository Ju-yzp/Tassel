#ifndef TASSEL_CORE_FEATURE_MANAGER_H_
#define TASSEL_CORE_FEATURE_MANAGER_H_

// 标准库
#include <Eigen/Core>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Tassel
#include "feature.h"

namespace tassel_core {

struct State;
struct SFMFeature;

struct HostLandmark {
    int feature_id = -1;
    cv::Point2f host_pixel;
    Eigen::Vector3d host_uv = Eigen::Vector3d::Zero();
    double host_depth = 0.0;
};

class FeatureManager {
public:
    FeatureManager(
        double reproj_err_thres, int tracked_times_thres, double min_translation,
        double keyframe_new_feature_ratio, double min_depth = MIN_DISTANCE,
        double max_depth = MAX_DISTANCE);

    bool addFeatureFrame(
        int frame_index, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    void triangulate(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeFrameObservations(
        int frame_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void replaceRetainedHost(
        int old_host_index, int new_host_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void removeNewestFrameObservations(int frame_index);

    void removeOutliers(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    bool hasLatestKeyframe() const { return !latest_keyframe_feature_ids_.empty(); }

    void reset();

    std::vector<Feature*> collectLandmarks();

    std::vector<MarginalizedFeatureObservation> collectMarginalizedObservations(
        int host_frame_index, int target_frame_index);

    std::vector<HostLandmark> exportHostLandmarks(int host_frame_index, const State& state) const;

    std::vector<SFMFeature> collectSFMFeatures(const State& state) const;

    std::unordered_map<int, Feature>& features() { return features_; }

private:
    double reproj_err_thres_;

    int tracked_times_thres_;

    double min_translation_;

    double keyframe_new_feature_ratio_;

    double min_depth_, max_depth_;

    std::unordered_set<int> latest_keyframe_feature_ids_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
