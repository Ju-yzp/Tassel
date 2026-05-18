#ifndef TASSEL_CORE_CAM_CAMERA_EQUI_H_
#define TASSEL_CORE_CAM_CAMERA_EQUI_H_

#include <Eigen/Core>

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "cam/camera_base.h"

namespace tassel_core {

class CameraEqui : public CameraBase {
public:
    CameraEqui(const cv::Mat& k, const cv::Mat& dist_coeffs, int width, int height)
        : CameraBase(k, dist_coeffs, width, height) {
        bool cond1 = (k_.rows == 3 && k_.cols == 3);
        bool cond2 = (dist_coeffs_.rows == 1 || dist_coeffs_.cols == 1);
        bool cond3 = (dist_coeffs_.rows * dist_coeffs_.cols == 4);
        if (!(cond1 && cond2 && cond3)) {
            throw std::invalid_argument("Invalid camera parameters");
        }
        k_.convertTo(k_, CV_64F);
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);

        // Store in OpenVINS format: [fx, fy, cx, cy, k1, k2, k3, k4]
        camera_values = Eigen::MatrixXd::Zero(8, 1);
        camera_values(0) = k_.at<double>(0, 0);         // fx
        camera_values(1) = k_.at<double>(1, 1);         // fy
        camera_values(2) = k_.at<double>(0, 2);         // cx
        camera_values(3) = k_.at<double>(1, 2);         // cy
        camera_values(4) = dist_coeffs_.at<double>(0);  // k1
        camera_values(5) = dist_coeffs_.at<double>(1);  // k2
        camera_values(6) = dist_coeffs_.at<double>(2);  // k3
        camera_values(7) = dist_coeffs_.at<double>(3);  // k4
    }

    Eigen::Vector2d undistort(const Eigen::Vector2d& pt) const override {
        std::vector<cv::Point2f> pts = {
            cv::Point2f(static_cast<float>(pt.x()), static_cast<float>(pt.y()))};
        std::vector<cv::Point2f> out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(pts, out, k_, dist_coeffs_, I, k_);
        return Eigen::Vector2d(out[0].x, out[0].y);
    }

    std::vector<Eigen::Vector2d> undistort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<cv::Point2f> cv_pts;
        cv_pts.reserve(pts.size());
        for (const auto& p : pts)
            cv_pts.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(cv_pts, cv_out, k_, dist_coeffs_, I, k_);
        std::vector<Eigen::Vector2d> out;
        out.reserve(cv_out.size());
        for (const auto& p : cv_out) out.emplace_back(p.x, p.y);
        return out;
    }

    Eigen::Vector2d distort(const Eigen::Vector2d& uv_norm) const override {
        double x = uv_norm(0);
        double y = uv_norm(1);
        double r = std::sqrt(x * x + y * y);
        double theta = std::atan(r);

        double k1 = camera_values(4);
        double k2 = camera_values(5);
        double k3 = camera_values(6);
        double k4 = camera_values(7);

        double theta_d = theta + k1 * std::pow(theta, 3) + k2 * std::pow(theta, 5) +
                         k3 * std::pow(theta, 7) + k4 * std::pow(theta, 9);

        double inv_r = r > 1e-8 ? 1.0 / r : 1.0;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1.0;

        double fx = camera_values(0);
        double fy = camera_values(1);
        double cx = camera_values(2);
        double cy = camera_values(3);

        return Eigen::Vector2d(fx * x * cdist + cx, fy * y * cdist + cy);
    }

    std::vector<Eigen::Vector2d> distort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<Eigen::Vector2d> out;
        out.reserve(pts.size());
        for (const auto& p : pts) out.emplace_back(distort(p));
        return out;
    }

    void get_jacobian(
        Eigen::Vector2d uv_norm, Eigen::MatrixXd& H_dz_dzn, Eigen::MatrixXd& H_dz_dzeta) override {
        double x = uv_norm(0);
        double y = uv_norm(1);
        double r = std::sqrt(x * x + y * y);
        double theta = std::atan(r);

        double fx = camera_values(0);
        double fy = camera_values(1);
        double k1 = camera_values(4);
        double k2 = camera_values(5);
        double k3 = camera_values(6);
        double k4 = camera_values(7);

        double theta_d = theta + k1 * std::pow(theta, 3) + k2 * std::pow(theta, 5) +
                         k3 * std::pow(theta, 7) + k4 * std::pow(theta, 9);

        double inv_r = r > 1e-8 ? 1.0 / r : 1.0;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1.0;

        // Chain rule: H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) *
        // dr_dxyn)

        // duv_dxy = diag(fx, fy)  [2x2]
        Eigen::Matrix2d duv_dxy = Eigen::Matrix2d::Zero();
        duv_dxy(0, 0) = fx;
        duv_dxy(1, 1) = fy;

        // dxy_dxyn = diag(theta_d * inv_r, theta_d * inv_r)  [2x2]
        Eigen::Matrix2d dxy_dxyn = Eigen::Matrix2d::Zero();
        dxy_dxyn(0, 0) = theta_d * inv_r;
        dxy_dxyn(1, 1) = theta_d * inv_r;

        // dxy_dr = [-x * theta_d / r^2, -y * theta_d / r^2]^T  [2x1]
        Eigen::Vector2d dxy_dr;
        dxy_dr << -x * theta_d * inv_r * inv_r, -y * theta_d * inv_r * inv_r;

        // dr_dxyn = [x/r, y/r]  [1x2]
        Eigen::RowVector2d dr_dxyn;
        dr_dxyn << x * inv_r, y * inv_r;

        // dxy_dthd = [x/r, y/r]^T  [2x1]
        Eigen::Vector2d dxy_dthd;
        dxy_dthd << x * inv_r, y * inv_r;

        // dthd_dth = 1 + 3*k1*θ² + 5*k2*θ⁴ + 7*k3*θ⁶ + 9*k4*θ⁸  [scalar]
        double dthd_dth = 1 + 3 * k1 * std::pow(theta, 2) + 5 * k2 * std::pow(theta, 4) +
                          7 * k3 * std::pow(theta, 6) + 9 * k4 * std::pow(theta, 8);

        // dth_dr = 1 / (1 + r²)  [scalar]
        double dth_dr = 1 / (r * r + 1);

        // Total Jacobian w.r.t. normalized coordinates
        H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
        H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);

        // Jacobian w.r.t. intrinsics [fx, fy, cx, cy, k1, k2, k3, k4] (2x8)
        H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
        H_dz_dzeta(0, 0) = x * cdist;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = fx * x * inv_r * std::pow(theta, 3);
        H_dz_dzeta(0, 5) = fx * x * inv_r * std::pow(theta, 5);
        H_dz_dzeta(0, 6) = fx * x * inv_r * std::pow(theta, 7);
        H_dz_dzeta(0, 7) = fx * x * inv_r * std::pow(theta, 9);
        H_dz_dzeta(1, 1) = y * cdist;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = fy * y * inv_r * std::pow(theta, 3);
        H_dz_dzeta(1, 5) = fy * y * inv_r * std::pow(theta, 5);
        H_dz_dzeta(1, 6) = fy * y * inv_r * std::pow(theta, 7);
        H_dz_dzeta(1, 7) = fy * y * inv_r * std::pow(theta, 9);
    }

private:
    Eigen::MatrixXd camera_values;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_CAM_CAMERA_EQUI_H_
