#include "pnp_verifier.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <cmath>
#include <stdexcept>

namespace tassel_loop {
namespace {

constexpr double kVarianceScale = 2.1981;
constexpr double kMinimumVariance = 1e-12;

double robustVariance(std::vector<double> errors, int quantile_divisor) {
    std::sort(errors.begin(), errors.end());
    return std::max(
        kVarianceScale * errors[errors.size() / static_cast<size_t>(quantile_divisor)],
        kMinimumVariance);
}

}  // namespace

PnpVerifier::PnpVerifier(
    int min_inliers, double min_inlier_ratio, double inlier_threshold, int max_iterations,
    double confidence, int variance_quantile_divisor, double max_translation_variance)
    : min_inliers_(min_inliers),
      min_inlier_ratio_(min_inlier_ratio),
      inlier_threshold_(inlier_threshold),
      max_iterations_(max_iterations),
      confidence_(confidence),
      variance_quantile_divisor_(variance_quantile_divisor),
      max_translation_variance_(max_translation_variance) {
    if (min_inliers_ < 6 || min_inlier_ratio_ <= 0.0 || min_inlier_ratio_ > 1.0 ||
        inlier_threshold_ <= 0.0 || max_iterations_ <= 0 || confidence_ <= 0.0 ||
        confidence_ >= 1.0 || variance_quantile_divisor_ <= 1 || max_translation_variance_ < 0.0) {
        throw std::invalid_argument("Invalid PnP verification options");
    }
}

PnpVerification PnpVerifier::verify(
    const std::vector<Eigen::Vector3d>& host_points,
    const std::vector<Eigen::Vector2d>& current_points) const {
    if (host_points.size() != current_points.size()) {
        throw std::invalid_argument("PnP verification point counts differ");
    }
    PnpVerification output;
    output.match_count = static_cast<int>(host_points.size());
    if (host_points.size() < static_cast<size_t>(min_inliers_)) {
        return output;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(host_points.size());
    image_points.reserve(current_points.size());
    for (size_t index = 0; index < host_points.size(); ++index) {
        if (!host_points[index].allFinite() || !current_points[index].allFinite()) {
            return output;
        }
        object_points.emplace_back(
            host_points[index].x(), host_points[index].y(), host_points[index].z());
        image_points.emplace_back(current_points[index].x(), current_points[index].y());
    }

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat current_r_host;
    cv::Mat current_t_host;
    cv::Mat inliers;
    const bool solved = cv::solvePnPRansac(
        object_points, image_points, camera_matrix, cv::noArray(), current_r_host, current_t_host,
        false, max_iterations_, inlier_threshold_, confidence_, inliers, cv::SOLVEPNP_EPNP);
    if (!solved || inliers.rows < min_inliers_) {
        return output;
    }

    std::vector<cv::Point3d> inlier_object_points;
    std::vector<cv::Point2d> inlier_image_points;
    inlier_object_points.reserve(inliers.rows);
    inlier_image_points.reserve(inliers.rows);
    for (int row = 0; row < inliers.rows; ++row) {
        const int index = inliers.at<int>(row);
        inlier_object_points.push_back(object_points[index]);
        inlier_image_points.push_back(image_points[index]);
    }
    cv::solvePnPRefineLM(
        inlier_object_points, inlier_image_points, camera_matrix, cv::noArray(), current_r_host,
        current_t_host);

    cv::Mat current_R_host_cv;
    cv::Rodrigues(current_r_host, current_R_host_cv);
    Eigen::Matrix3d current_R_host;
    Eigen::Vector3d current_t_host_eigen;
    cv::cv2eigen(current_R_host_cv, current_R_host);
    cv::cv2eigen(current_t_host, current_t_host_eigen);
    output.host_R_current = current_R_host.transpose();
    output.host_t_current = -output.host_R_current * current_t_host_eigen;
    output.inlier_count = inliers.rows;
    output.inlier_ratio = static_cast<double>(output.inlier_count) / output.match_count;

    double error_sum = 0.0;
    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    translation_errors.reserve(inliers.rows);
    rotation_errors.reserve(inliers.rows);
    for (int row = 0; row < inliers.rows; ++row) {
        const int index = inliers.at<int>(row);
        const Eigen::Vector3d point_current =
            current_R_host * host_points[index] + current_t_host_eigen;
        if (point_current.z() <= 0.0) {
            return PnpVerification{};
        }
        const Eigen::Vector2d projection = point_current.head<2>() / point_current.z();
        error_sum += (projection - current_points[index]).norm();

        const Eigen::Vector3d ray(current_points[index].x(), current_points[index].y(), 1.0);
        const Eigen::Vector3d reconstructed_current = ray * point_current.z() * 1.1;
        const Eigen::Vector3d reconstructed_host =
            output.host_R_current * (reconstructed_current - current_t_host_eigen);
        const Eigen::Vector3d difference = host_points[index] - reconstructed_host;
        translation_errors.push_back(difference.squaredNorm());
        const double cosine = std::clamp(
            host_points[index].normalized().dot(reconstructed_host.normalized()), -1.0, 1.0);
        const double angle = std::acos(cosine);
        rotation_errors.push_back(angle);
    }
    output.mean_reprojection_error = error_sum / output.inlier_count;
    output.translation_variance =
        robustVariance(std::move(translation_errors), variance_quantile_divisor_);
    output.rotation_variance =
        robustVariance(std::move(rotation_errors), variance_quantile_divisor_);
    output.accepted = output.inlier_ratio >= min_inlier_ratio_ &&
                      output.mean_reprojection_error <= inlier_threshold_ &&
                      (max_translation_variance_ == 0.0 ||
                       output.translation_variance <= max_translation_variance_) &&
                      output.host_R_current.allFinite() && output.host_t_current.allFinite();
    return output;
}

}  // namespace tassel_loop
