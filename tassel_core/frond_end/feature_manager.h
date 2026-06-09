#ifndef TASSEL_CORE_FEATURE_MANAGER_H_
#define TASSEL_CORE_FEATURE_MANAGER_H_

// cpp
#include <Eigen/Core>
#include <unordered_map>
#include <vector>

// tassel
#include "feature.h"

namespace tassel_core {

struct State;

class FeatureManager {
public:
    FeatureManager(
        double reprojection_error_thres, double pnp_reprojection_error_thres, double parallax_thres,
        int tracked_times_thres, int min_tracked_pts_num, int min_pnp_pt_num,
        double min_pnp_inliers_ratio, double min_translation, double min_depth = MIN_DISTANCE,
        double max_depth = MAX_DISTANCE);

    bool checkKeyFrameByParallax(
        size_t frame_count, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    void triangulate(
        const State& state, const Eigen::Matrix3d& ric1 = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& tic1 = Eigen::Vector3d::Zero());

    bool initPoseByPNP(
        int frame_count, std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps,
        Eigen::Matrix3d ric, Eigen::Vector3d tic);

    void removeOldest(const State& state);

    void removeNewest(size_t frame_count);

    void removeOutliers(const State& state);

    void reset();

    std::vector<Feature*> collectOptimizedFeatures();

    std::vector<Feature*> collectMarginalizationFeatures();

    std::vector<Eigen::Vector3d> getPointCloud(const State& state) const;

    std::unordered_map<int, Feature>& testFeatures() { return features_; }

    void removeMarginalizedFeatures();

private:
    double reprojection_error_thres_;
    double pnp_reprojection_error_thres_;

    double parallax_thres_;

    int tracked_times_thres_;

    int min_tracked_pts_num_;

    int min_pnp_pt_num_;

    double min_pnp_inliers_ratio_;

    double min_translation_;

    double min_depth_, max_depth_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
