#ifndef TASSEL_CORE_CAM_CAMERA_BASE_H_
#define TASSEL_CORE_CAM_CAMERA_BASE_H_

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

namespace tassel_core {

class CameraBase {
public:
    CameraBase(const cv::Mat& k, const cv::Mat& dist_coeffs, int width, int height)
        : k_(k), dist_coeffs_(dist_coeffs), width_(width), height_(height) {
        if (height_ <= 0 || width_ <= 0) {
            throw std::invalid_argument("Invalid image dimensions");
        }
    }

    CameraBase(
        const Eigen::Ref<const Eigen::Matrix3d>& K,
        const Eigen::Ref<const Eigen::VectorXd>& dist_coeffs, int width, int height)
        : k_(Eigen33ToCvMat(K)),
          dist_coeffs_(EigenVecToCvMat(dist_coeffs)),
          width_(width),
          height_(height) {
        if (height_ <= 0 || width_ <= 0) {
            throw std::invalid_argument("Invalid image dimensions");
        }
    }

    virtual ~CameraBase() = default;

    virtual Eigen::Vector2d undistort(const Eigen::Vector2d& pt) const = 0;

    virtual Eigen::Vector2d distort(const Eigen::Vector2d& uv_norm) const = 0;

    virtual std::vector<Eigen::Vector2d> undistort(const std::vector<Eigen::Vector2d>& pts) const {
        std::vector<Eigen::Vector2d> out;
        out.reserve(pts.size());
        for (const auto& p : pts) out.emplace_back(undistort(p));
        return out;
    }

    virtual std::vector<Eigen::Vector2d> distort(
        const std::vector<Eigen::Vector2d>& uv_norm) const {
        std::vector<Eigen::Vector2d> out;
        out.reserve(uv_norm.size());
        for (const auto& p : uv_norm) out.emplace_back(distort(p));
        return out;
    }

    virtual void get_jacobian_dzn(Eigen::Vector2d uv_norm, Eigen::MatrixXd& H_dz_dzn) const = 0;

    virtual void get_jacobian_dzeta(Eigen::Vector2d uv_norm, Eigen::MatrixXd& H_dz_dzeta) const = 0;

    int get_height() const { return height_; }
    int get_width() const { return width_; }

protected:
    cv::Mat k_;
    cv::Mat dist_coeffs_;

private:
    static cv::Mat Eigen33ToCvMat(const Eigen::Ref<const Eigen::Matrix3d>& m) {
        cv::Mat out(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) out.at<double>(i, j) = m(i, j);
        return out;
    }

    static cv::Mat EigenVecToCvMat(const Eigen::Ref<const Eigen::VectorXd>& v) {
        cv::Mat out(1, static_cast<int>(v.size()), CV_64F);
        for (int i = 0; i < v.size(); ++i) out.at<double>(0, i) = v(i);
        return out;
    }

    int width_, height_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_CAM_CAMERA_BASE_H_
