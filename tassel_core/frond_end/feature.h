#ifndef TASSEL_CORE_FEATURE_H_
#define TASSEL_CORE_FEATURE_H_

#include <Eigen/Core>
#include <vector>

#include <opencv2/core.hpp>

namespace tassel_core {
struct State;

struct FeaturePerFrame {
    void setObservation(Eigen::Vector2d uv, cv::Point2f pt) {
        this->uv << uv(0), uv(1), 1.0;
        this->pt = pt;
    }

    cv::Point2f pt;
    Eigen::Vector3d uv = Eigen::Vector3d::Zero();
    double sync_delay = 0.0;
};

struct Feature {
    static constexpr double InvalidDepth = -1.0;

    Feature(int host_frame_index, size_t max_capacity);

    int observationFrameIndex(size_t observation_index) const {
        return host_frame_index + static_cast<int>(observation_index);
    }

    void monoTriangulate(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
        double min_depth);

    void removeFrame(
        int frame_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    bool transferHost(
        int new_host_index, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void removeFrameObservation(int frame_index);

    int host_frame_index;
    double estimated_depth;
    std::vector<FeaturePerFrame> observations;
};

struct MarginalizedFeatureObservation {
    Feature* feature = nullptr;
    std::vector<int> target_frame_indices;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FEATURE_H_
