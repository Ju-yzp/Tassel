#ifndef TASSEL_CORE_FEATURE_MANAGER_H_
#define TASSEL_CORE_FEATURE_MANAGER_H_

// cpp
#include <memory>
#include <unordered_map>

// tassel
#include "feature/feature.h"

// #include "tassel_utils/camera_base.h"

namespace tassel_core {

struct State;

class FeatureManager {
public:
    FeatureManager(
        double reprojection_error_thres, double parallax_thres, int tracked_times_thres,
        int max_pnp_needed_num, int min_pnp_pt_num, double min_pnp_inliers_ratio,
        double min_translation, double min_depth = MIN_DISTANCE, double max_depth = MAX_DISTANCE);

    bool checkKeyFrameByParallax(
        size_t frame_count, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    void triangulate(
        std::shared_ptr<State> state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
        const Eigen::Matrix3d& ric1 = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& tic1 = Eigen::Vector3d::Zero());

    // 仅适用于 VO 模式
    void initPoseByPNP(
        std::shared_ptr<State> state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeOldest(
        std::shared_ptr<State> state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeNewest(size_t frame_count);

    void removeOutliers(
        std::shared_ptr<State> state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void reset();

    // 收集具有有效深度和足够观测数的特征用于优化。
    // 估计器使用这些特征构建视觉因子并填充 depth_ptrs。
    std::vector<Feature*> collectOptimizationFeatures();

    // 收集起始帧为 0、深度有效且观测数 ≥2 的特征，
    // 供估计器用于边缘化。
    std::vector<std::pair<int, const Feature*>> collectMarginalizationFeatures() const;

    std::vector<Eigen::Vector3d> get_pointcloud(
        std::shared_ptr<State> state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

private:
    double reprojection_error_thres_;

    double parallax_thres_;

    int tracked_times_thres_;

    int min_pnp_pt_num_;

    double min_pnp_inliers_ratio_;

    double min_translation_;

    double min_depth_, max_depth_;

    int max_pnp_needed_num_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
