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

struct HostLandmark {
    int feature_id = -1;
    cv::Point2f host_pixel;
    Eigen::Vector3d host_uv = Eigen::Vector3d::Zero();
    double host_depth = 0.0;
};

struct ObservedLandmark {
    int feature_id = -1;
    cv::Point2f pixel;
    Eigen::Vector3d world_point = Eigen::Vector3d::Zero();
};

class FeatureManager {
public:
    FeatureManager(
        double reproj_err_thres, int min_landmark_observations, double parallax_threshold,
        double keyframe_min_connection_ratio, double min_depth, double max_depth);

    bool addFeatureFrame(
        int frame_index, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    bool replaceInitializationCandidate(
        int accepted_frame_index, int candidate_frame_index,
        const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    void triangulate(const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void removeFrameObservations(
        int frame_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void replaceRetainedHost(
        int old_host_index, int new_host_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void removeNewestFrameObservations(int frame_index);

    std::vector<int> removeOutliers(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    bool hasLatestKeyframe() const { return !latest_keyframe_observations_.empty(); }

    void reset();

    std::vector<Feature*> collectLandmarks();

    // target_frame_index < 0 选择宿主帧后的全部观测，否则只选择该目标帧的观测。
    // 返回指针仅在本次特征管理器不修改 features_ 的边缘化流程内有效。
    std::vector<Feature*> collectMarginalizedFeatures(
        int host_frame_index, int target_frame_index = -1);

    std::vector<HostLandmark> exportHostLandmarks(int host_frame_index, const State& state) const;

    std::unordered_map<int, Eigen::Vector3d> exportObservedWorldLandmarks(
        int observed_frame_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic) const;

    std::vector<ObservedLandmark> exportObservedLandmarks(
        int observed_frame_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic) const;

    std::vector<SFMFeature> collectSFMFeatures(const State& state, int first_frame_index = 0) const;

    std::unordered_map<int, Feature>& features() { return features_; }

private:
    bool shouldCreateKeyframe(const std::unordered_map<int, FeaturePerFrame>& feature_frame) const;

    double reproj_err_thres_;

    int min_landmark_observations_;

    double parallax_threshold_;

    double keyframe_min_connection_ratio_;

    double min_depth_, max_depth_;

    std::unordered_map<int, cv::Point2f> latest_keyframe_observations_;

    std::unordered_map<int, Feature> features_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FEATURE_MANAGER_H_
