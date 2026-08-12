#ifndef TASSEL_TOOLS_RTABMAP_RTABMAP_BACKEND_H_
#define TASSEL_TOOLS_RTABMAP_RTABMAP_BACKEND_H_

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "tassel_utils/types.h"

namespace tassel_core {
struct ObservedLandmark;
}

namespace tassel_tools {

struct RtabmapBackendOptions {
    std::string database_path;
    size_t queue_capacity = 3;
    double linear_variance = 0.01;
    double angular_variance = 0.01;
};

class RtabmapBackend {
public:
    RtabmapBackend(
        const cv::Mat& camera_matrix, const cv::Mat& distortion, const cv::Size& image_size,
        bool equidistant, const Sophus::SE3d& imu_to_camera, RtabmapBackendOptions options = {});
    ~RtabmapBackend();

    RtabmapBackend(const RtabmapBackend&) = delete;
    RtabmapBackend& operator=(const RtabmapBackend&) = delete;

    void submit(
        tassel_utils::FrameId frame_id, const cv::Mat& image, const Sophus::SE3d& world_to_imu,
        const std::vector<tassel_core::ObservedLandmark>& landmarks);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_RTABMAP_RTABMAP_BACKEND_H_
