#ifndef TASSEL_CORE_CAM_CAMERA_RAD_TAN_H_
#define TASSEL_CORE_CAM_CAMERA_RAD_TAN_H_

#include <Eigen/Core>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "cam/camera_base.h"

namespace tassel_core {

class CameraRadTan : public CameraBase {
public:
    CameraRadTan(const cv::Mat& k, const cv::Mat& dist_coeffs, int width, int height)
        : CameraBase(k, dist_coeffs, width, height) {
        bool cond1 = (k_.rows == 3 && k_.cols == 3);
        bool cond2 = (dist_coeffs_.rows == 1 || dist_coeffs_.cols == 1);
        bool cond3 =
            (dist_coeffs_.rows * dist_coeffs_.cols == 4 ||
             dist_coeffs_.rows * dist_coeffs_.cols == 5);
        if (!(cond1 && cond2 && cond3)) {
            throw std::invalid_argument("Invalid camera parameters");
        }
        k_.convertTo(k_, CV_64F);
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
    }

    Eigen::Vector2d undistort(const Eigen::Vector2d& pt) const override {
        std::vector<cv::Point2f> pts = {
            cv::Point2f(static_cast<float>(pt.x()), static_cast<float>(pt.y()))};
        std::vector<cv::Point2f> out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::undistortPoints(pts, out, k_, dist_coeffs_, I, I);
        return Eigen::Vector2d(out[0].x, out[0].y);
    }

    std::vector<Eigen::Vector2d> undistort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<cv::Point2f> cv_pts;
        cv_pts.reserve(pts.size());
        for (const auto& p : pts)
            cv_pts.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::undistortPoints(cv_pts, cv_out, k_, dist_coeffs_, I, I);
        std::vector<Eigen::Vector2d> out;
        out.reserve(cv_out.size());
        for (const auto& p : cv_out) out.emplace_back(p.x, p.y);
        return out;
    }

    Eigen::Vector2d distort(const Eigen::Vector2d& uv_norm) const override {
        double x = uv_norm(0);
        double y = uv_norm(1);
        double r2 = x * x + y * y;
        double r4 = r2 * r2;

        double k1 = dist_coeffs_.at<double>(0);
        double k2 = dist_coeffs_.at<double>(1);
        double p1 = dist_coeffs_.at<double>(2);
        double p2 = dist_coeffs_.at<double>(3);
        double k3 = dist_coeffs_.total() > 4 ? dist_coeffs_.at<double>(4) : 0.0;

        double radial = 1 + k1 * r2 + k2 * r4 + k3 * r4 * r2;
        double x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        double fx = k_.at<double>(0, 0);
        double fy = k_.at<double>(1, 1);
        double cx = k_.at<double>(0, 2);
        double cy = k_.at<double>(1, 2);

        return Eigen::Vector2d(fx * x_dist + cx, fy * y_dist + cy);
    }

    std::vector<Eigen::Vector2d> distort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<Eigen::Vector2d> out;
        out.reserve(pts.size());
        for (const auto& p : pts) out.emplace_back(distort(p));
        return out;
    }
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_CAM_CAMERA_RAD_TAN_H_
