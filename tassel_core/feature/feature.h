#ifndef TASSEL_CORE_FEATURE_H_
#define TASSEL_CORE_FEATURE_H_

#include <Eigen/Core>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace tassel_core {
struct FeaturePerFrame {
    FeaturePerFrame()
        : is_stereo(false),
          pt(cv::Point2f()),
          pt_r(cv::Point2f()),
          uv(Eigen::Vector3d::Zero()),
          uv_r(Eigen::Vector3d::Zero()){};
    void setLeft(Eigen::Vector2d uv, cv::Point2f pt) {
        this->uv << uv(0), uv(1), 1.0;
        this->pt = pt;
    }

    void setRight(Eigen::Vector2d uv, cv::Point2f pt) {
        this->uv_r << uv(0), uv(1), 1.0;
        this->pt_r = pt;
        is_stereo = true;
    }
    bool is_stereo;
    cv::Point2f pt, pt_r;
    Eigen::Vector3d uv, uv_r;
};

const double INVALID_DEPTH = -1.0;
const double MIN_DISTANCE = 0.1;
const double MAX_DISTANCE = 3.0;

struct Feature {
    //     Feature(size_t max_capacity);

    //     Feature(size_t start_frame_id, size_t max_capacity);

    //     Feature();

    //     void stereoTriangulate(
    //         const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1,
    //         const Eigen::Vector3d& tic1, double min_depth, double max_depth);

    //     void monoTriangulate(
    //         std::shared_ptr<State> state, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    //         double min_depth, double max_depth);

    //     void removeOldest(
    //         const Eigen::Matrix3d& prev_r, const Eigen::Vector3d& prev_t, const Eigen::Matrix3d&
    //         cur_r, const Eigen::Vector3d& cur_t, const Eigen::Matrix3d& ric, const
    //         Eigen::Vector3d& tic);

    //     void removeNewest(size_t frame_count);

    size_t start_frame_id;
    double estimated_depth;
    std::vector<FeaturePerFrame> observations;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FEATURE_H_
