#ifndef TASSEL_UTILS_TYPES_H_
#define TASSEL_UTILS_TYPES_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <functional>
#include <opencv2/core.hpp>
#include <vector>

namespace tassel_utils {
const int O_P = 0;
const int O_R = 3;
const int O_V = 6;
const int O_BA = 9;
const int O_BG = 12;
const Eigen::Vector3d G{0.0, 0.0, 9.8};

struct IMUMeasurement {
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
    double timestamp;
    double get_timestamp() const { return timestamp; }
    void set_timestamp(double ts) { timestamp = ts; }
};

struct StereoObservation {
    cv::Mat left_img;
    cv::Mat right_img;
    double timestamp;
    double get_timestamp() const { return timestamp; }
    void set_timestamp(double ts) { timestamp = ts; }
};

// Callback types for estimator output
using PoseCallback = std::function<void(const Eigen::Matrix3d&, const Eigen::Vector3d&)>;
using PointCloudCallback = std::function<void(const std::vector<Eigen::Vector3d>&)>;
using ResidualCallback = std::function<void(double, double)>;

}  // namespace tassel_utils
#endif
