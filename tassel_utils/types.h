#ifndef TASSEL_UTILS_TYPES_H_
#define TASSEL_UTILS_TYPES_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <functional>
#include <opencv2/core.hpp>
#include <vector>

#include "tassel_utils/constants.h"

namespace tassel_utils {

inline Eigen::Vector3d G{0.0, 0.0, 9.8};

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
    double timestamp;
    inline double get_timestamp() const { return timestamp; }
    inline void set_timestamp(double ts) { timestamp = ts; }
};

// Callback types for estimator output
using PoseCallback = std::function<void(const Eigen::Matrix3d&, const Eigen::Vector3d&)>;
using PointCloudCallback = std::function<void(const std::vector<Eigen::Vector3d>&)>;
using ResidualCallback = std::function<void(double, double)>;

}  // namespace tassel_utils
#endif
