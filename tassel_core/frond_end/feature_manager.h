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
struct SFMFeature;

class FeatureManager {
public:
    FeatureManager(
        double reproj_err_thres, double pnp_reproj_err_thres, double parallax_thres,
        int tracked_times_thres, int min_tracked_pts, int min_pnp_pts, double min_pnp_inliers_ratio,
        double min_translation, double min_depth = MIN_DISTANCE, double max_depth = MAX_DISTANCE);

    bool checkParallax(
        size_t frame_count, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    inline void invalidateDepths() {
        for (auto& [id, feature] : features_) {
            feature.estimated_depth = INVALID_DEPTH;
        }
    }

    void triangulate(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
        const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1);

    void removeOldest(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeNewest(size_t frame_count);

    void removeOutliers(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void reset();

    std::vector<Feature*> collectLandmarks();

    std::vector<Feature*> collectMargFeatures();

    std::vector<Eigen::Vector3d> getPointCloud(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) const;

    void removeMargFeatures();

    void reset(int parallax_thres);

    std::vector<SFMFeature> collectSFMFeatures(int frame_num) const;

    std::unordered_map<int, Feature>& features() { return features_; }

private:
    double reproj_err_thres_;

    double pnp_reproj_err_thres_;

    double parallax_thres_;

    int tracked_times_thres_;

    int min_tracked_pts_;

    int min_pnp_pts_;

    double min_pnp_inliers_ratio_;

    double min_translation_;

    double min_depth_, max_depth_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
