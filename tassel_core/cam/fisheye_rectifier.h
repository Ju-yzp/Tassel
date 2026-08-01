#ifndef TASSEL_CORE_CAM_FISHEYE_RECTIFIER_H_
#define TASSEL_CORE_CAM_FISHEYE_RECTIFIER_H_

#include <opencv2/core.hpp>

namespace tassel_core {

class FisheyeRectifier {
public:
    FisheyeRectifier(
        const cv::Mat& camera_matrix, const cv::Mat& distortion, cv::Size image_size,
        double mask_radius);

    cv::Mat rectify(const cv::Mat& gray) const;

    const cv::Mat& cameraMatrix() const { return rectified_camera_matrix_; }
    const cv::Mat& validMask() const { return valid_mask_; }
    cv::Size outputSize() const { return output_size_; }

private:
    cv::Size source_size_;
    cv::Size output_size_;
    cv::Mat rectified_camera_matrix_;
    cv::Mat map_x_;
    cv::Mat map_y_;
    cv::Mat valid_mask_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_CAM_FISHEYE_RECTIFIER_H_
