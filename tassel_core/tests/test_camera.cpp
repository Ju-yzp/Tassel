#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <random>

#include "cam/camera_equi.h"
#include "cam/camera_factory.h"
#include "cam/camera_rad_tan.h"

namespace {

cv::Mat radtan_K =
    (cv::Mat_<double>(3, 3) << 455.510864, 0.000000, 328.529851, 0.000000, 455.426715, 225.596721,
     0.0, 0.0, 1.0);
cv::Mat radtan_D = (cv::Mat_<double>(1, 5) << 0.010831, -0.007841, 0.000166, 0.000512, 0.000000);

cv::Mat equi_K = (cv::Mat_<double>(3, 3) << 300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);
cv::Mat equi_D = (cv::Mat_<double>(1, 4) << 0.1, 0.01, 0.001, 0.0001);

const int kWidth = 640;
const int kHeight = 480;

// ── Numerical differentiation helpers for Jacobian verification ────────────

Eigen::Matrix2d numerical_dzn(
    tassel_core::CameraBase* cam, const Eigen::Vector2d& uv_norm, double eps = 1e-6) {
    Eigen::Matrix2d J;
    for (int i = 0; i < 2; ++i) {
        Eigen::Vector2d d = Eigen::Vector2d::Zero();
        d(i) = eps;
        J.col(i) = (cam->distort(uv_norm + d) - cam->distort(uv_norm - d)) / (2.0 * eps);
    }
    return J;
}

template <typename CamType>
Eigen::Matrix<double, 2, 8> numerical_dzeta(
    const cv::Mat& K, const cv::Mat& D, int w, int h, const Eigen::Vector2d& uv_norm,
    double eps = 1e-6) {
    Eigen::Matrix<double, 2, 8> J;
    for (int p = 0; p < 8; ++p) {
        auto eval = [&](double delta) -> Eigen::Vector2d {
            cv::Mat Kp = K.clone();
            cv::Mat Dp = D.clone();
            switch (p) {
                case 0:
                    Kp.at<double>(0, 0) += delta;
                    break;  // fx
                case 1:
                    Kp.at<double>(1, 1) += delta;
                    break;  // fy
                case 2:
                    Kp.at<double>(0, 2) += delta;
                    break;  // cx
                case 3:
                    Kp.at<double>(1, 2) += delta;
                    break;  // cy
                case 4:
                    Dp.at<double>(0) += delta;
                    break;  // k1
                case 5:
                    Dp.at<double>(1) += delta;
                    break;  // k2
                case 6:
                    Dp.at<double>(2) += delta;
                    break;  // p1/k3
                case 7:
                    Dp.at<double>(3) += delta;
                    break;  // p2/k4
            }
            CamType cam(Kp, Dp, w, h);
            return cam.distort(uv_norm);
        };
        J.col(p) = (eval(+eps) - eval(-eps)) / (2.0 * eps);
    }
    return J;
}

}  // namespace

// ── Validation tests ───────────────────────────────────────────────────────

TEST(CameraRadTan, ThrowsOnInvalidIntrinsicsSize) {
    cv::Mat bad_K(2, 2, CV_64F);
    EXPECT_THROW(
        tassel_core::CameraRadTan(bad_K, radtan_D, kWidth, kHeight), std::invalid_argument);
}

TEST(CameraRadTan, ThrowsOnInvalidDistortionSize) {
    cv::Mat bad_D(1, 3, CV_64F);
    EXPECT_THROW(
        tassel_core::CameraRadTan(radtan_K, bad_D, kWidth, kHeight), std::invalid_argument);
}

TEST(CameraRadTan, ThrowsOnNonPositiveDimensions) {
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

    Eigen::Vector2d normalize(const Eigen::Vector2d& pixel) {
        double fx = radtan_K.at<double>(0, 0);
        double fy = radtan_K.at<double>(1, 1);
        double cx = radtan_K.at<double>(0, 2);
        double cy = radtan_K.at<double>(1, 2);
        return Eigen::Vector2d((pixel(0) - cx) / fx, (pixel(1) - cy) / fy);
    }

    tassel_core::Camera cam_;
};

TEST_F(CameraRadTanTest, UndistortMatchesOpenCV) {
    std::vector<Eigen::Vector2d> pixels = {
        {350, 200},
        {150, 300},
        {500, 400},
        {320, 225},
    };
    auto uv_norm = cam_->undistort(pixels);
    for (size_t i = 0; i < pixels.size(); ++i) {
        std::vector<cv::Point2f> cv_in = {
            cv::Point2f(static_cast<float>(pixels[i](0)), static_cast<float>(pixels[i](1)))};
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::undistortPoints(cv_in, cv_out, radtan_K, radtan_D, I);
        EXPECT_NEAR(uv_norm[i](0), cv_out[0].x, 1e-6);
        EXPECT_NEAR(uv_norm[i](1), cv_out[0].y, 1e-6);
    }
}

TEST_F(CameraRadTanTest, PixelRoundTrip) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(0, kWidth);
    std::uniform_real_distribution<double> dy(0, kHeight);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector2d pixel_raw(dx(rng), dy(rng));
        Eigen::Vector2d norm_undist = cam_->undistort(pixel_raw);
        Eigen::Vector2d pixel_redist = cam_->distort(norm_undist);
        EXPECT_NEAR(pixel_raw(0), pixel_redist(0), 0.5);
        EXPECT_NEAR(pixel_raw(1), pixel_redist(1), 0.5);
    }
}

TEST_F(CameraRadTanTest, JacobianDZN) {
    const double eps = 1e-6;
    const double tol = 1e-4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rnd(-1.2, 1.2);

    for (int k = 0; k < 20; ++k) {
        Eigen::Vector2d uv(rnd(rng), rnd(rng));
        Eigen::MatrixXd H_analytic, H_zeta;
        cam_->get_jacobian(uv, H_analytic, H_zeta);
        Eigen::Matrix2d H_num = numerical_dzn(cam_.get(), uv, eps);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_NEAR(H_analytic(i, j), H_num(i, j), tol)
                    << "uv=(" << uv(0) << ", " << uv(1) << ") elem(" << i << "," << j << ")";
    }
}

TEST_F(CameraRadTanTest, JacobianDZeta) {
    const double eps = 1e-5;
    const double tol = 1e-4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rnd(-1.2, 1.2);

    for (int k = 0; k < 15; ++k) {
        Eigen::Vector2d uv(rnd(rng), rnd(rng));
        Eigen::MatrixXd H_dzn, H_analytic;
        cam_->get_jacobian(uv, H_dzn, H_analytic);
        Eigen::Matrix<double, 2, 8> H_num = numerical_dzeta<tassel_core::CameraRadTan>(
            radtan_K, radtan_D, kWidth, kHeight, uv, eps);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 8; ++j)
                EXPECT_NEAR(H_analytic(i, j), H_num(i, j), tol)
                    << "uv=(" << uv(0) << ", " << uv(1) << ") elem(" << i << "," << j << ")";
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

    tassel_core::Camera cam_;
};

TEST_F(CameraEquiTest, UndistortMatchesOpenCV) {
    std::vector<Eigen::Vector2d> pixels = {
        {350, 200},
        {150, 300},
        {500, 400},
        {320, 240},
    };
    auto uv_norm = cam_->undistort(pixels);
    for (size_t i = 0; i < pixels.size(); ++i) {
        std::vector<cv::Point2f> cv_in = {
            cv::Point2f(static_cast<float>(pixels[i](0)), static_cast<float>(pixels[i](1)))};
        std::vector<cv::Point2f> cv_out;
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        cv::fisheye::undistortPoints(cv_in, cv_out, equi_K, equi_D, I);
        EXPECT_NEAR(uv_norm[i](0), cv_out[0].x, 1e-6);
        EXPECT_NEAR(uv_norm[i](1), cv_out[0].y, 1e-6);
    }
}

TEST_F(CameraEquiTest, PixelRoundTrip) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(0, kWidth);
    std::uniform_real_distribution<double> dy(0, kHeight);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector2d pixel_raw(dx(rng), dy(rng));
        Eigen::Vector2d norm_undist = cam_->undistort(pixel_raw);
        Eigen::Vector2d pixel_redist = cam_->distort(norm_undist);
        EXPECT_NEAR(pixel_raw(0), pixel_redist(0), 0.5);
        EXPECT_NEAR(pixel_raw(1), pixel_redist(1), 0.5);
    }
}

TEST_F(CameraEquiTest, ZeroDistortionRoundTrip) {
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

TEST_F(CameraEquiTest, JacobianDZN) {
    const double eps = 1e-6;
    const double tol = 1e-4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rnd(-1.2, 1.2);

    for (int k = 0; k < 20; ++k) {
        Eigen::Vector2d uv(rnd(rng), rnd(rng));
        if (uv.norm() < 0.01) continue;  // skip singularity at r=0
        Eigen::MatrixXd H_analytic, H_zeta;
        cam_->get_jacobian(uv, H_analytic, H_zeta);
        Eigen::Matrix2d H_num = numerical_dzn(cam_.get(), uv, eps);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_NEAR(H_analytic(i, j), H_num(i, j), tol)
                    << "uv=(" << uv(0) << ", " << uv(1) << ") elem(" << i << "," << j << ")";
    }
}

TEST_F(CameraEquiTest, JacobianDZeta) {
    const double eps = 1e-5;
    const double tol = 1e-4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rnd(-1.2, 1.2);

    for (int k = 0; k < 15; ++k) {
        Eigen::Vector2d uv(rnd(rng), rnd(rng));
        if (uv.norm() < 0.01) continue;  // skip singularity at r=0
        Eigen::MatrixXd H_dzn, H_analytic;
        cam_->get_jacobian(uv, H_dzn, H_analytic);
        Eigen::Matrix<double, 2, 8> H_num =
            numerical_dzeta<tassel_core::CameraEqui>(equi_K, equi_D, kWidth, kHeight, uv, eps);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 8; ++j)
                EXPECT_NEAR(H_analytic(i, j), H_num(i, j), tol)
                    << "uv=(" << uv(0) << ", " << uv(1) << ") elem(" << i << "," << j << ")";
    }
}

TEST(CameraEqui, FourCoefDistortionIsValid) {
    EXPECT_NO_THROW(tassel_core::CameraEqui(equi_K, equi_D, kWidth, kHeight));
}

// ── Polymorphism through base pointer ─────────────────────────────────────

TEST(CameraPolymorphism, BothModelsWorkThroughBasePtr) {
    tassel_core::Camera radtan =
        std::make_unique<tassel_core::CameraRadTan>(radtan_K, radtan_D, kWidth, kHeight);
    tassel_core::Camera equi =
        std::make_unique<tassel_core::CameraEqui>(equi_K, equi_D, kWidth, kHeight);

    std::vector<Eigen::Vector2d> pixels = {{350, 200}, {150, 300}, {500, 400}};

    for (const auto& p : pixels) {
        auto r1 = radtan->undistort(p);
        auto r2 = radtan->undistort(p);
        EXPECT_EQ(r1, r2);

        auto e1 = equi->undistort(p);
        auto e2 = equi->undistort(p);
        EXPECT_EQ(e1, e2);
    }
}

// ── Factory ────────────────────────────────────────────────────────────────

TEST(CameraFactory, CreatesFromString) {
    auto cam1 = tassel_core::CameraFactory::create("radtan", radtan_K, radtan_D, kWidth, kHeight);
    auto cam2 = tassel_core::CameraFactory::create("equi", equi_K, equi_D, kWidth, kHeight);

    EXPECT_EQ(cam1->get_width(), kWidth);
    EXPECT_EQ(cam1->get_height(), kHeight);
    EXPECT_EQ(cam2->get_width(), kWidth);
    EXPECT_EQ(cam2->get_height(), kHeight);
}

TEST(CameraFactory, ThrowsOnUnknownModel) {
    EXPECT_THROW(
        tassel_core::CameraFactory::create("unknown", radtan_K, radtan_D, kWidth, kHeight),
        std::invalid_argument);
}
