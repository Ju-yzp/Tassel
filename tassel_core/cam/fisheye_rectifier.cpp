#include "cam/fisheye_rectifier.h"

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace tassel_core {

FisheyeRectifier::FisheyeRectifier(
    const cv::Mat& camera_matrix, const cv::Mat& distortion, cv::Size image_size,
    double mask_radius)
    : source_size_(image_size) {
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3 || distortion.total() != 4 ||
        image_size.width <= 0 || image_size.height <= 0 || !std::isfinite(mask_radius) ||
        mask_radius <= 0.0) {
        throw std::invalid_argument("Invalid fisheye rectification configuration");
    }
    const double max_radius = 0.5 * std::min(image_size.width, image_size.height);
    if (mask_radius > max_radius) {
        throw std::invalid_argument("Fisheye mask radius exceeds the output image");
    }

    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, distortion, image_size, identity, rectified_camera_matrix_, 0.0,
        image_size, 1.0);
    cv::fisheye::initUndistortRectifyMap(
        camera_matrix, distortion, identity, rectified_camera_matrix_, image_size, CV_32FC1,
        map_x_, map_y_);

    cv::Mat full_mask = cv::Mat::zeros(image_size, CV_8UC1);
    const cv::Point center(
        cvRound(rectified_camera_matrix_.at<double>(0, 2)),
        cvRound(rectified_camera_matrix_.at<double>(1, 2)));
    cv::circle(full_mask, center, cvRound(mask_radius), cv::Scalar(255), -1);
    const cv::Rect roi = cv::boundingRect(full_mask);
    if (roi.empty()) {
        throw std::logic_error("Fisheye valid region is empty");
    }

    // remap 输出采用有效圆的外接矩形，虚拟内参必须同步平移到裁剪后的像素坐标系。
    map_x_ = map_x_(roi).clone();
    map_y_ = map_y_(roi).clone();
    valid_mask_ = full_mask(roi).clone();
    rectified_camera_matrix_.at<double>(0, 2) -= roi.x;
    rectified_camera_matrix_.at<double>(1, 2) -= roi.y;
    output_size_ = roi.size();
}

cv::Mat FisheyeRectifier::rectify(const cv::Mat& gray) const {
    if (gray.empty() || gray.type() != CV_8UC1 || gray.size() != source_size_) {
        throw std::invalid_argument("Fisheye rectifier expects a fixed-size grayscale image");
    }
    cv::Mat output;
    cv::remap(gray, output, map_x_, map_y_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::bitwise_and(output, valid_mask_, output);
    return output;
}

}  // namespace tassel_core
