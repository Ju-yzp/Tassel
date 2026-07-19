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
        validate_and_store();
    }

    CameraRadTan(
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
        cv::undistortPoints(pts, out, k_, dist_coeffs_, I, I);
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
        cv::undistortPoints(cv_pts, cv_out, k_, dist_coeffs_, I, I);
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
        double r2 = x * x + y * y;
        double r4 = r2 * r2;

        double k1 = camera_values(4);
        double k2 = camera_values(5);
        double p1 = camera_values(6);
        double p2 = camera_values(7);

        double radial = 1 + k1 * r2 + k2 * r4;
        double x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        double fx = camera_values(0);
        double fy = camera_values(1);
        double cx = camera_values(2);
        double cy = camera_values(3);

        return Eigen::Vector2d(fx * x_dist + cx, fy * y_dist + cy);
    }

    std::vector<Eigen::Vector2d> distort(const std::vector<Eigen::Vector2d>& pts) const override {
        std::vector<Eigen::Vector2d> out;
        out.reserve(pts.size());
        for (const auto& p : pts) {
            out.emplace_back(distort(p));
        }
        return out;
    }

    void get_jacobian_dzn(Eigen::Vector2d uv_norm, Eigen::MatrixXd& H_dz_dzn) const override {
        double x = uv_norm(0);
        double y = uv_norm(1);
        double x_2 = x * x;
        double y_2 = y * y;
        double x_y = x * y;
        double r_2 = x_2 + y_2;
        double r_4 = r_2 * r_2;

        double fx = camera_values(0);
        double fy = camera_values(1);
        double k1 = camera_values(4);
        double k2 = camera_values(5);
        double p1 = camera_values(6);
        double p2 = camera_values(7);

        H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
        H_dz_dzn(0, 0) = fx * ((1 + k1 * r_2 + k2 * r_4) + (2 * k1 * x_2 + 4 * k2 * x_2 * r_2) +
                               2 * p1 * y + (2 * p2 * x + 4 * p2 * x));
        H_dz_dzn(0, 1) = fx * (2 * k1 * x_y + 4 * k2 * x_y * r_2 + 2 * p1 * x + 2 * p2 * y);
        H_dz_dzn(1, 0) = fy * (2 * k1 * x_y + 4 * k2 * x_y * r_2 + 2 * p1 * x + 2 * p2 * y);
        H_dz_dzn(1, 1) = fy * ((1 + k1 * r_2 + k2 * r_4) + (2 * k1 * y_2 + 4 * k2 * y_2 * r_2) +
                               2 * p2 * x + (2 * p1 * y + 4 * p1 * y));
    }

    void get_jacobian_dzeta(Eigen::Vector2d uv_norm, Eigen::MatrixXd& H_dz_dzeta) const override {
        double x = uv_norm(0);
        double y = uv_norm(1);
        double x_2 = x * x;
        double y_2 = y * y;
        double x_y = x * y;
        double r_2 = x_2 + y_2;
        double r_4 = r_2 * r_2;

        double fx = camera_values(0);
        double fy = camera_values(1);
        double k1 = camera_values(4);
        double k2 = camera_values(5);
        double p1 = camera_values(6);
        double p2 = camera_values(7);

        double radial = 1 + k1 * r_2 + k2 * r_4;
        double x1 = x * radial + 2 * p1 * x_y + p2 * (r_2 + 2 * x_2);
        double y1 = y * radial + p1 * (r_2 + 2 * y_2) + 2 * p2 * x_y;

        H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
        H_dz_dzeta(0, 0) = x1;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = fx * x * r_2;
        H_dz_dzeta(0, 5) = fx * x * r_4;
        H_dz_dzeta(0, 6) = fx * 2 * x_y;
        H_dz_dzeta(0, 7) = fx * (r_2 + 2 * x_2);
        H_dz_dzeta(1, 1) = y1;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = fy * y * r_2;
        H_dz_dzeta(1, 5) = fy * y * r_4;
        H_dz_dzeta(1, 6) = fy * (r_2 + 2 * y_2);
        H_dz_dzeta(1, 7) = fy * 2 * x_y;
    }

private:
    void validate_and_store() {
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

        camera_values = Eigen::MatrixXd::Zero(8, 1);
        camera_values(0) = k_.at<double>(0, 0);         // fx
        camera_values(1) = k_.at<double>(1, 1);         // fy
        camera_values(2) = k_.at<double>(0, 2);         // cx
        camera_values(3) = k_.at<double>(1, 2);         // cy
        camera_values(4) = dist_coeffs_.at<double>(0);  // k1
        camera_values(5) = dist_coeffs_.at<double>(1);  // k2
        camera_values(6) = dist_coeffs_.at<double>(2);  // p1
        camera_values(7) = dist_coeffs_.at<double>(3);  // p2
    }

    Eigen::MatrixXd camera_values;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_CAM_CAMERA_RAD_TAN_H_
