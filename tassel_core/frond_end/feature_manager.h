#ifndef TASSEL_CORE_FEATURE_MANAGER_H_
#define TASSEL_CORE_FEATURE_MANAGER_H_

// cpp
#include <Eigen/Core>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// tassel
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

struct OutlierStats {
    size_t checked_count = 0;
    size_t removed_count = 0;
    double average_error = 0.0;
    double maximum_error = 0.0;
    double removed_average_error = 0.0;
};

class FeatureManager {
public:
    FeatureManager(
        double reproj_err_thres, double parallax_thres, int tracked_times_thres,
        int min_tracked_pts, double min_translation, double min_depth = MIN_DISTANCE,
        double max_depth = MAX_DISTANCE);

    bool checkParallax(
        tassel_utils::FrameId frame_id,
        const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    inline void invalidateDepths() {
        for (auto& item : features_) {
            item.second.estimated_depth = INVALID_DEPTH;
        }
    }

    void triangulate(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeOldest(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeFrame(
        tassel_utils::FrameId frame_id, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void replaceHost(
        tassel_utils::FrameId old_host_frame_id, tassel_utils::FrameId new_host_frame_id,
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeNewest(tassel_utils::FrameId frame_id);

    OutlierStats removeOutliers(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    const FeatureInputStats& lastInputStats() const { return last_input_stats_; }

    bool hasLatestKeyframe() const { return latest_keyframe_id_ != tassel_utils::kInvalidFrameId; }
    tassel_utils::FrameId latestKeyframeId() const { return latest_keyframe_id_; }
    bool isKeyframe(tassel_utils::FrameId frame_id) const {
        return keyframe_ids_.count(frame_id) > 0;
    }
    void retireKeyframe(tassel_utils::FrameId frame_id);
    void acceptKeyframe(
        tassel_utils::FrameId frame_id,
        const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    void reset();

    std::vector<Feature*> collectLandmarks();

    std::vector<MarginalizedFeatureObservation> collectMarginalizedObservations(
        tassel_utils::FrameId host_frame_id, tassel_utils::FrameId target_frame_id);

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

    tassel_utils::FrameId latest_keyframe_id_ = tassel_utils::kInvalidFrameId;
    std::unordered_set<tassel_utils::FrameId> keyframe_ids_;
    std::unordered_set<int> latest_keyframe_feature_ids_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
