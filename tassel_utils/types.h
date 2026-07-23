#ifndef TASSEL_UTILS_TYPES_H_
#define TASSEL_UTILS_TYPES_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

namespace tassel_utils {

inline Eigen::Vector3d G{0.0, 0.0, 9.8};

using FrameId = std::int64_t;
inline constexpr FrameId kInvalidFrameId = std::numeric_limits<FrameId>::min();

inline FrameId secondsToFrameId(double timestamp) {
    return static_cast<FrameId>(std::llround(timestamp * 1e9));
}

inline double frameIdToSeconds(FrameId frame_id) { return static_cast<double>(frame_id) * 1e-9; }

using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct IMUMeasurement {
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
    double timestamp;
    inline double get_timestamp() const { return timestamp; }
    inline void set_timestamp(double ts) { timestamp = ts; }
};

struct StereoObservation {
    cv::Mat left_img;
    cv::Mat right_img;
    FrameId timestamp = kInvalidFrameId;
    inline double get_timestamp() const { return frameIdToSeconds(timestamp); }
    inline void set_timestamp(double ts) { timestamp = secondsToFrameId(ts); }
};

enum class IntegratorType {
    MidPoint,
    Euler,
};

// 估计器输出回调类型
using PoseCallback = std::function<void(const Eigen::Matrix3d&, const Eigen::Vector3d&)>;
using PointCloudCallback = std::function<void(const std::vector<Eigen::Vector3d>&)>;
using ResidualCallback = std::function<void(double, double)>;

}  // namespace tassel_utils
#endif
