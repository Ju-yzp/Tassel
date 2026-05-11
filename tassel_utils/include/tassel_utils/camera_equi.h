#ifndef TASSEL_UTILS_CAMERA_EQUI_H_
#define TASSEL_UTILS_CAMERA_EQUI_H_

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "tassel_utils/camera_base.h"

namespace tassel_utils {

class CameraEqui : public CameraBase {
public:
    CameraEqui(const cv::Mat& k, const cv::Mat& dist_coeffs, int width, int height)
        : CameraBase(k, dist_coeffs, width, height) {
        k_.convertTo(k_, CV_64F);
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
    }

    Eigen::Vector2d undistort(const Eigen::Vector2d& pt) override {
        std::vector<cv::Point2f> pts = {
            cv::Point2f(static_cast<float>(pt.x()), static_cast<float>(pt.y()))};
        std::vector<cv::Point2f> out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(pts, out, k_, dist_coeffs_, I, k_);
        return Eigen::Vector2d(out[0].x, out[0].y);
    }

    Eigen::Vector2d distort(const Eigen::Vector2d& uv_norm) override {
        double x = uv_norm(0);
        double y = uv_norm(1);
        double r = std::sqrt(x * x + y * y);
        double theta = std::atan(r);

        double k1 = dist_coeffs_.at<double>(0);
        double k2 = dist_coeffs_.at<double>(1);
        double k3 = dist_coeffs_.at<double>(2);
        double k4 = dist_coeffs_.at<double>(3);

        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta4 * theta2;
        double theta8 = theta4 * theta4;
        double theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

        double inv_r = r > 1e-8 ? 1.0 / r : 1.0;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1.0;

        double u = k_.at<double>(0, 0) * x * cdist + k_.at<double>(0, 2);
        double v = k_.at<double>(1, 1) * y * cdist + k_.at<double>(1, 2);
        return Eigen::Vector2d(u, v);
    }
};

}  // namespace tassel_utils

#endif  // TASSEL_UTILS_CAMERA_EQUI_H_
