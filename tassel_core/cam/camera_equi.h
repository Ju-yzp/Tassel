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
        validate_and_store();
    }

    CameraEqui(
        const Eigen::Ref<const Eigen::Matrix3d>& K,
        const Eigen::Ref<const Eigen::VectorXd>& dist_coeffs, int width, int height)
        : CameraBase(K, dist_coeffs, width, height) {
        validate_and_store();
    }

    Eigen::Vector2d undistort(const Eigen::Vector2d& pt) const override {
        std::vector<cv::Point2f> pts = {
            cv::Point2f(static_cast<float>(pt.x()), static_cast<float>(pt.y()))};
        std::vector<cv::Point2f> out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(pts, out, k_, dist_coeffs_, I, I);
        return Eigen::Vector2d(out[0].x, out[0].y);
    }

    std::vector<Eigen::Vector2d> undistort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<cv::Point2f> cv_pts;
        cv_pts.reserve(pts.size());
        for (const auto& p : pts) {
            cv_pts.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
        }
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(cv_pts, cv_out, k_, dist_coeffs_, I, I);
        std::vector<Eigen::Vector2d> out;
        out.reserve(cv_out.size());
        for (const auto& p : cv_out) {
            out.emplace_back(p.x, p.y);
        }
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

        const double theta2 = theta * theta;
        const double theta3 = theta2 * theta;
        const double theta5 = theta3 * theta2;
        const double theta7 = theta5 * theta2;
        const double theta9 = theta7 * theta2;
        double theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        double inv_r = r > 1e-8 ? 1.0 / r : 1.0;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1.0;

        double fx = camera_values(0);
        double fy = camera_values(1);
        double cx = camera_values(2);
        double cy = camera_values(3);

        return Eigen::Vector2d(fx * x * cdist + cx, fy * y * cdist + cy);
    }

    void distortWithJacobian(
        const Eigen::Vector2d& uv_norm, Eigen::Vector2d& pixel,
        Eigen::Matrix2d& pixel_jacobian) const override {
        const double x = uv_norm.x();
        const double y = uv_norm.y();
        const double r = std::sqrt(x * x + y * y);
        if (r <= 1e-8) {
            pixel = Eigen::Vector2d(camera_values(2), camera_values(3));
            pixel_jacobian.setZero();
            pixel_jacobian(0, 0) = camera_values(0);
            pixel_jacobian(1, 1) = camera_values(1);
            return;
        }

        const double theta = std::atan(r);
        const double theta2 = theta * theta;
        const double theta3 = theta2 * theta;
        const double theta4 = theta2 * theta2;
        const double theta5 = theta3 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta7 = theta5 * theta2;
        const double theta8 = theta6 * theta2;
        const double theta9 = theta7 * theta2;
        const double theta_d = theta + camera_values(4) * theta3 + camera_values(5) * theta5 +
                               camera_values(6) * theta7 + camera_values(7) * theta9;
        const double dtheta_d_dtheta =
            1.0 + 3.0 * camera_values(4) * theta2 + 5.0 * camera_values(5) * theta4 +
            7.0 * camera_values(6) * theta6 + 9.0 * camera_values(7) * theta8;
        const double inv_r = 1.0 / r;
        const double scale = theta_d * inv_r;
        const double scale_derivative = (dtheta_d_dtheta / (1.0 + r * r) - scale) * inv_r * inv_r;

        pixel.x() = camera_values(0) * x * scale + camera_values(2);
        pixel.y() = camera_values(1) * y * scale + camera_values(3);
        pixel_jacobian(0, 0) = camera_values(0) * (scale + x * x * scale_derivative);
        pixel_jacobian(0, 1) = camera_values(0) * x * y * scale_derivative;
        pixel_jacobian(1, 0) = camera_values(1) * x * y * scale_derivative;
        pixel_jacobian(1, 1) = camera_values(1) * (scale + y * y * scale_derivative);
    }

    std::vector<Eigen::Vector2d> distort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<Eigen::Vector2d> out;
        out.reserve(pts.size());
        for (const auto& p : pts) {
            out.emplace_back(distort(p));
        }
        return out;
    }

    void get_jacobian_dzn(
        const Eigen::Vector2d& uv_norm, Eigen::Matrix2d& H_dz_dzn) const override {
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

        if (r <= 1e-8) {
            H_dz_dzn = Eigen::Matrix2d::Zero();
            H_dz_dzn(0, 0) = fx;
            H_dz_dzn(1, 1) = fy;
            return;
        }

        const double theta2 = theta * theta;
        const double theta3 = theta2 * theta;
        const double theta5 = theta3 * theta2;
        const double theta7 = theta5 * theta2;
        const double theta9 = theta7 * theta2;
        double theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        double inv_r = r > 1e-8 ? 1.0 / r : 1.0;

        Eigen::Matrix2d duv_dxy = Eigen::Matrix2d::Zero();
        duv_dxy(0, 0) = fx;
        duv_dxy(1, 1) = fy;

        Eigen::Matrix2d dxy_dxyn = Eigen::Matrix2d::Zero();
        dxy_dxyn(0, 0) = theta_d * inv_r;
        dxy_dxyn(1, 1) = theta_d * inv_r;

        Eigen::Vector2d dxy_dr;
        dxy_dr << -x * theta_d * inv_r * inv_r, -y * theta_d * inv_r * inv_r;

        Eigen::RowVector2d dr_dxyn;
        dr_dxyn << x * inv_r, y * inv_r;

        Eigen::Vector2d dxy_dthd;
        dxy_dthd << x * inv_r, y * inv_r;

        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta6 * theta2;
        double dthd_dth = 1 + 3 * k1 * theta2 + 5 * k2 * theta4 + 7 * k3 * theta6 + 9 * k4 * theta8;

        double dth_dr = 1 / (r * r + 1);

        H_dz_dzn.setZero();
        H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);
    }

    void get_jacobian_dzeta(Eigen::Vector2d uv_norm, Eigen::MatrixXd& H_dz_dzeta) const override {
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

        const double theta2 = theta * theta;
        const double theta3 = theta2 * theta;
        const double theta5 = theta3 * theta2;
        const double theta7 = theta5 * theta2;
        const double theta9 = theta7 * theta2;
        double theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        double inv_r = r > 1e-8 ? 1.0 / r : 1.0;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1.0;

        H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
        H_dz_dzeta(0, 0) = x * cdist;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = fx * x * inv_r * theta3;
        H_dz_dzeta(0, 5) = fx * x * inv_r * theta5;
        H_dz_dzeta(0, 6) = fx * x * inv_r * theta7;
        H_dz_dzeta(0, 7) = fx * x * inv_r * theta9;
        H_dz_dzeta(1, 1) = y * cdist;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = fy * y * inv_r * theta3;
        H_dz_dzeta(1, 5) = fy * y * inv_r * theta5;
        H_dz_dzeta(1, 6) = fy * y * inv_r * theta7;
        H_dz_dzeta(1, 7) = fy * y * inv_r * theta9;
    }

private:
    void validate_and_store() {
        bool cond1 = (k_.rows == 3 && k_.cols == 3);
        bool cond2 = (dist_coeffs_.rows == 1 || dist_coeffs_.cols == 1);
        bool cond3 = (dist_coeffs_.rows * dist_coeffs_.cols == 4);
        if (!(cond1 && cond2 && cond3)) {
            throw std::invalid_argument("Invalid camera parameters");
        }
        k_.convertTo(k_, CV_64F);
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);

        camera_values.setZero();
        camera_values(0) = k_.at<double>(0, 0);         // fx
        camera_values(1) = k_.at<double>(1, 1);         // fy
        camera_values(2) = k_.at<double>(0, 2);         // cx
        camera_values(3) = k_.at<double>(1, 2);         // cy
        camera_values(4) = dist_coeffs_.at<double>(0);  // k1
        camera_values(5) = dist_coeffs_.at<double>(1);  // k2
        camera_values(6) = dist_coeffs_.at<double>(2);  // k3
        camera_values(7) = dist_coeffs_.at<double>(3);  // k4
    }

    Eigen::Matrix<double, 8, 1> camera_values;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_CAM_CAMERA_EQUI_H_
