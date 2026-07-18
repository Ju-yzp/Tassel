#ifndef TASSEL_CORE_FEATURE_H_
#define TASSEL_CORE_FEATURE_H_

#include <Eigen/Core>
#include <vector>

#include <opencv2/core.hpp>
#include "tassel_utils/types.h"

namespace tassel_core {
struct State;

struct FeaturePerFrame {
    FeaturePerFrame()
        : frame_id(tassel_utils::kInvalidFrameId),
          is_stereo(false),
          pt(cv::Point2f()),
          pt_r(cv::Point2f()),
          uv(Eigen::Vector3d::Zero()),
          uv_r(Eigen::Vector3d::Zero()),
          applied_delay(0.0){};
    void setLeft(Eigen::Vector2d uv, cv::Point2f pt) {
        this->uv << uv(0), uv(1), 1.0;
        this->pt = pt;
    }

    void setRight(Eigen::Vector2d uv, cv::Point2f pt) {
        this->uv_r << uv(0), uv(1), 1.0;
        this->pt_r = pt;
        is_stereo = true;
    }
    tassel_utils::FrameId frame_id;
    bool is_stereo;
    cv::Point2f pt, pt_r;
    Eigen::Vector3d uv, uv_r;
    double applied_delay;
};

inline constexpr double INVALID_DEPTH = -1.0;
inline constexpr double MIN_DISTANCE = 0.1;
inline constexpr double MAX_DISTANCE = 3.0;

struct Feature {
    Feature(tassel_utils::FrameId host_frame_id, size_t max_capacity);

    void monoTriangulate(
        const State& state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
        double min_translation, double min_depth, double max_depth);

    void removeFrame(
        tassel_utils::FrameId frame_id, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    bool transferHost(
        tassel_utils::FrameId new_host_frame_id, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void removeFrameObservation(tassel_utils::FrameId frame_id);

    tassel_utils::FrameId host_frame_id;
    double estimated_depth;
    std::vector<FeaturePerFrame> observations;
    // 已进入过边缘化先验；宿主滑动并继承深度后允许继续边缘化。
    bool has_been_marginalized = false;
};

struct MarginalizedFeatureObservation {
    Feature* feature = nullptr;
    tassel_utils::FrameId target_frame_id = tassel_utils::kInvalidFrameId;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FEATURE_H_
