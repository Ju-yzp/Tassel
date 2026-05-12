#ifndef TASSEL_UTILS_CAMERA_BASE_H_
#define TASSEL_UTILS_CAMERA_BASE_H_

#include <Eigen/Core>

#include <opencv2/core.hpp>

namespace tassel_utils {
class CameraBase {
public:
    explicit CameraBase(const cv::Mat& k, const cv::Mat& dist_coeffs, int width, int height)
        : k_(k), dist_coeffs_(dist_coeffs), width_(width), height_(height) {
        bool cond1 = (k_.rows == 3 && k_.cols == 3);
        bool cond2 = (dist_coeffs_.rows == 1 || dist_coeffs_.cols == 1);
        bool cond3 =
            (dist_coeffs_.rows * dist_coeffs_.cols == 4 ||
             dist_coeffs_.rows * dist_coeffs_.cols == 5);
        if (!(cond1 && cond2 && cond3)) {
            throw std::invalid_argument("Invalid camera parameters");
        }
        if (height_ <= 0 || width_ <= 0) {
            throw std::invalid_argument("Invalid image dimensions");
        }
    }

    virtual Eigen::Vector2d undistort(const Eigen::Vector2d&) = 0;

    virtual Eigen::Vector2d distort(const Eigen::Vector2d&) = 0;

    int get_height() const { return height_; }

    int get_width() const { return width_; }

protected:
    cv::Mat k_;
    cv::Mat dist_coeffs_;

private:
    int width_, height_;
};
}  // namespace tassel_utils
#endif  // TASSEL_UTILS_CAMERA_BASE_H_
