#include "geometric_verifier.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <stdexcept>

namespace tassel_loop {

GeometricVerifier::GeometricVerifier(
    int min_inliers, double min_inlier_ratio, double ransac_threshold)
    : min_inliers_(min_inliers),
      min_inlier_ratio_(min_inlier_ratio),
      ransac_threshold_(ransac_threshold) {
    if (min_inliers_ < 5 || min_inlier_ratio_ <= 0.0 || min_inlier_ratio_ > 1.0 ||
        ransac_threshold_ <= 0.0) {
        throw std::invalid_argument("Invalid geometric verification options");
    }
}

GeometricVerification GeometricVerifier::verify(
    const std::vector<Eigen::Vector2d>& candidate_points,
    const std::vector<Eigen::Vector2d>& current_points) const {
    if (candidate_points.size() != current_points.size()) {
        throw std::invalid_argument("Geometric verification point counts differ");
    }
    GeometricVerification output;
    output.match_count = static_cast<int>(candidate_points.size());
    if (candidate_points.size() < 5) {
        return output;
    }

    std::vector<cv::Point2d> candidate_cv;
    std::vector<cv::Point2d> current_cv;
    candidate_cv.reserve(candidate_points.size());
    current_cv.reserve(current_points.size());
    for (size_t i = 0; i < candidate_points.size(); ++i) {
        candidate_cv.emplace_back(candidate_points[i].x(), candidate_points[i].y());
        current_cv.emplace_back(current_points[i].x(), current_points[i].y());
    }

    cv::Mat mask;
    const cv::Mat essential = cv::findEssentialMat(
        candidate_cv, current_cv, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.999, ransac_threshold_,
        mask);
    if (essential.empty()) {
        return output;
    }
    cv::Mat current_R_candidate;
    cv::Mat current_t_candidate;
    output.inlier_count = cv::recoverPose(
        essential, candidate_cv, current_cv, current_R_candidate, current_t_candidate, 1.0,
        cv::Point2d(0.0, 0.0), mask);
    cv::cv2eigen(current_R_candidate, output.candidate_R_current);
    Eigen::Vector3d current_t_candidate_eigen;
    cv::cv2eigen(current_t_candidate, current_t_candidate_eigen);
    output.candidate_R_current.transposeInPlace();
    output.candidate_t_current =
        (-output.candidate_R_current * current_t_candidate_eigen).normalized();
    const double ratio = static_cast<double>(output.inlier_count) / output.match_count;
    output.accepted = output.inlier_count >= min_inliers_ && ratio >= min_inlier_ratio_;
    return output;
}

}  // namespace tassel_loop
