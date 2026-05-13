#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <random>

#include "cam/camera_base.h"
#include "cam/camera_equi.h"
#include "cam/camera_rad_tan.h"

namespace {

// ── shared test camera matrices ────────────────────────────────────────────

cv::Mat radtan_K =
    (cv::Mat_<double>(3, 3) << 455.510864, 0.000000, 328.529851, 0.000000, 455.426715, 225.596721,
     0.0, 0.0, 1.0);
cv::Mat radtan_D = (cv::Mat_<double>(1, 5) << 0.010831, -0.007841, 0.000166, 0.000512, 0.000000);

cv::Mat equi_K = (cv::Mat_<double>(3, 3) << 300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);
cv::Mat equi_D = (cv::Mat_<double>(1, 4) << 0.1, 0.01, 0.001, 0.0001);

const int kWidth = 640;
const int kHeight = 480;
}  // namespace

// ── CameraBase ─────────────────────────────────────────────────────────────

TEST(CameraBase, ThrowsOnInvalidIntrinsicsSize) {
    cv::Mat bad_K(2, 2, CV_64F);
    EXPECT_THROW(
        tassel_core::CameraRadTan(bad_K, radtan_D, kWidth, kHeight), std::invalid_argument);
}

TEST(CameraBase, ThrowsOnInvalidDistortionSize) {
    cv::Mat bad_D(1, 3, CV_64F);  // must be 4 or 5
    EXPECT_THROW(
        tassel_core::CameraRadTan(radtan_K, bad_D, kWidth, kHeight), std::invalid_argument);
}

TEST(CameraBase, ThrowsOnNonPositiveDimensions) {
    EXPECT_THROW(tassel_core::CameraRadTan(radtan_K, radtan_D, 0, kWidth), std::invalid_argument);
    EXPECT_THROW(tassel_core::CameraRadTan(radtan_K, radtan_D, kHeight, -1), std::invalid_argument);
}

TEST(CameraBase, GettersReturnCorrectValues) {
    tassel_core::CameraRadTan cam(radtan_K, radtan_D, kWidth, kHeight);
    EXPECT_EQ(cam.get_width(), kWidth);
    EXPECT_EQ(cam.get_height(), kHeight);
}

// ── CameraRadTan ───────────────────────────────────────────────────────────

class CameraRadTanTest : public ::testing::Test {
protected:
    void SetUp() override {
        cam_ = std::make_unique<tassel_core::CameraRadTan>(radtan_K, radtan_D, kWidth, kHeight);
    }

    // normalize pixel coords: K^-1 * pixel
    Eigen::Vector2d normalize(const Eigen::Vector2d& pixel) {
        double fx = radtan_K.at<double>(0, 0);
        double fy = radtan_K.at<double>(1, 1);
        double cx = radtan_K.at<double>(0, 2);
        double cy = radtan_K.at<double>(1, 2);
        return Eigen::Vector2d((pixel(0) - cx) / fx, (pixel(1) - cy) / fy);
    }

    std::unique_ptr<tassel_core::CameraBase> cam_;
};

TEST_F(CameraRadTanTest, UndistortMatchesOpenCV) {
    // pixel → undistort (returns undistorted pixels) → normalize → compare with opencv
    std::vector<Eigen::Vector2d> pixels = {
        {350, 200},
        {150, 300},
        {500, 400},
        {320, 225},
    };
    for (const auto& p : pixels) {
        Eigen::Vector2d uv_pixel = cam_->undistort(p);
        Eigen::Vector2d uv_norm = normalize(uv_pixel);

        // OpenCV undistortPoints with P=cv::noArray() returns normalized coords
        std::vector<cv::Point2f> cv_in = {
            cv::Point2f(static_cast<float>(p(0)), static_cast<float>(p(1)))};
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::undistortPoints(cv_in, cv_out, radtan_K, radtan_D, I);
        EXPECT_NEAR(uv_norm(0), cv_out[0].x, 1e-6);
        EXPECT_NEAR(uv_norm(1), cv_out[0].y, 1e-6);
    }
}

TEST_F(CameraRadTanTest, PixelRoundTrip) {
    // distorted pixel → undistort → undsitorted pixel → normalize → distort → back to pixel ≈
    // original
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(0, kWidth);
    std::uniform_real_distribution<double> dy(0, kHeight);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector2d pixel_raw(dx(rng), dy(rng));
        Eigen::Vector2d pixel_undist = cam_->undistort(pixel_raw);
        Eigen::Vector2d norm_undist = normalize(pixel_undist);
        Eigen::Vector2d pixel_redist = cam_->distort(norm_undist);
        EXPECT_NEAR(pixel_raw(0), pixel_redist(0), 0.5);
        EXPECT_NEAR(pixel_raw(1), pixel_redist(1), 0.5);
    }
}

// ── CameraEqui ─────────────────────────────────────────────────────────────

class CameraEquiTest : public ::testing::Test {
protected:
    void SetUp() override {
        cam_ = std::make_unique<tassel_core::CameraEqui>(equi_K, equi_D, kWidth, kHeight);
    }

    Eigen::Vector2d normalize(const Eigen::Vector2d& pixel) {
        double fx = equi_K.at<double>(0, 0);
        double fy = equi_K.at<double>(1, 1);
        double cx = equi_K.at<double>(0, 2);
        double cy = equi_K.at<double>(1, 2);
        return Eigen::Vector2d((pixel(0) - cx) / fx, (pixel(1) - cy) / fy);
    }

    std::unique_ptr<tassel_core::CameraBase> cam_;
};

TEST_F(CameraEquiTest, UndistortMatchesOpenCV) {
    std::vector<Eigen::Vector2d> pixels = {
        {350, 200},
        {150, 300},
        {500, 400},
        {320, 240},
    };
    for (const auto& p : pixels) {
        Eigen::Vector2d uv_pixel = cam_->undistort(p);
        Eigen::Vector2d uv_norm = normalize(uv_pixel);

        std::vector<cv::Point2f> cv_in = {
            cv::Point2f(static_cast<float>(p(0)), static_cast<float>(p(1)))};
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(cv_in, cv_out, equi_K, equi_D, I);
        EXPECT_NEAR(uv_norm(0), cv_out[0].x, 1e-6);
        EXPECT_NEAR(uv_norm(1), cv_out[0].y, 1e-6);
    }
}

TEST_F(CameraEquiTest, PixelRoundTrip) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(0, kWidth);
    std::uniform_real_distribution<double> dy(0, kHeight);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector2d pixel_raw(dx(rng), dy(rng));
        Eigen::Vector2d pixel_undist = cam_->undistort(pixel_raw);
        Eigen::Vector2d norm_undist = normalize(pixel_undist);
        Eigen::Vector2d pixel_redist = cam_->distort(norm_undist);
        EXPECT_NEAR(pixel_raw(0), pixel_redist(0), 0.5);
        EXPECT_NEAR(pixel_raw(1), pixel_redist(1), 0.5);
    }
}

TEST_F(CameraEquiTest, ZeroDistortionRoundTrip) {
    // 零畸变 equi: 小 r 时 atan(r)/r ≈ 1，靠近中心时模型近似为针孔。
    cv::Mat zero_D = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 0.0);
    tassel_core::CameraEqui cam_zero(equi_K, zero_D, kWidth, kHeight);

    std::vector<Eigen::Vector2d> norms = {
        {0.001, 0.0}, {-0.001, 0.0}, {0.0, 0.001}, {0.0, -0.001}, {0.0007, 0.0007},
    };
    for (const auto& uv_norm : norms) {
        Eigen::Vector2d uv_dist = cam_zero.distort(uv_norm);
        Eigen::Vector2d uv_norm_back = normalize(uv_dist);
        EXPECT_NEAR(uv_norm(0), uv_norm_back(0), 1e-6);
        EXPECT_NEAR(uv_norm(1), uv_norm_back(1), 1e-6);
    }
}

TEST(CameraEqui, FourCoefDistortionIsValid) {
    EXPECT_NO_THROW(tassel_core::CameraEqui(equi_K, equi_D, kWidth, kHeight));
}

// ── Polymorphism ───────────────────────────────────────────────────────────

TEST(CameraPolymorphism, BothModelsWorkThroughBasePtr) {
    auto radtan = std::make_unique<tassel_core::CameraRadTan>(radtan_K, radtan_D, kWidth, kHeight);
    auto equi = std::make_unique<tassel_core::CameraEqui>(equi_K, equi_D, kWidth, kHeight);

    // undistort some points through base pointer
    std::vector<Eigen::Vector2d> pixels = {{350, 200}, {150, 300}, {500, 400}};

    for (const auto& p : pixels) {
        auto r1 = radtan->undistort(p);
        auto r2 = radtan->undistort(p);
        EXPECT_EQ(r1, r2);  // deterministic

        auto e1 = equi->undistort(p);
        auto e2 = equi->undistort(p);
        EXPECT_EQ(e1, e2);
    }
}
