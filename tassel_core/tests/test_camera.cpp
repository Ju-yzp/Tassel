// =============================================================================
// test_camera.cpp
//
// Purpose:
//   验证相机模型的参数检查、畸变/去畸变一致性和解析雅各比。
//
// Test design:
//   使用固定 RadTan 与 Equidistant 内参/畸变参数构造相机; 用 OpenCV 结果作为
//   去畸变参考, 用中心差分作为 distort 对归一化坐标和相机参数的雅各比参考。
//
// Pass criteria:
//   相机工厂能正确创建对象, 非法参数会被拒绝, 投影往返误差与解析雅各比误差
//   均落在测试容差内。
// =============================================================================

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
const std::vector<Eigen::Vector2d> kPixels = {{350, 200}, {150, 300}, {500, 400}, {320, 240}};

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

void expectVectorNear(const Eigen::Vector2d& a, const Eigen::Vector2d& b, double tol) {
    EXPECT_NEAR(a(0), b(0), tol);
    EXPECT_NEAR(a(1), b(1), tol);
}

void expectMatrixNear(
    const Eigen::MatrixXd& analytic, const Eigen::MatrixXd& numeric, double tol,
    const Eigen::Vector2d& uv) {
    ASSERT_EQ(analytic.rows(), numeric.rows());
    ASSERT_EQ(analytic.cols(), numeric.cols());
    for (int r = 0; r < analytic.rows(); ++r) {
        for (int c = 0; c < analytic.cols(); ++c) {
            EXPECT_NEAR(analytic(r, c), numeric(r, c), tol)
                << "uv=(" << uv(0) << ", " << uv(1) << ") elem(" << r << "," << c << ")";
        }
    }
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

template <typename UndistortFn>
void expectUndistortMatchesOpenCV(
    tassel_core::CameraBase* cam, const cv::Mat& K, const cv::Mat& D,
    const std::vector<Eigen::Vector2d>& pixels, UndistortFn&& undistort) {
    auto uv_norm = cam->undistort(pixels);
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    for (size_t i = 0; i < pixels.size(); ++i) {
        std::vector<cv::Point2f> cv_in = {
            cv::Point2f(static_cast<float>(pixels[i](0)), static_cast<float>(pixels[i](1)))};
        std::vector<cv::Point2f> cv_out;
        undistort(cv_in, cv_out, K, D, I);
        expectVectorNear(uv_norm[i], Eigen::Vector2d(cv_out[0].x, cv_out[0].y), 1e-6);
    }
}

void expectPixelRoundTrip(tassel_core::CameraBase* cam) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(0, kWidth);
    std::uniform_real_distribution<double> dy(0, kHeight);

    for (int i = 0; i < 50; ++i) {
        Eigen::Vector2d pixel_raw(dx(rng), dy(rng));
        expectVectorNear(pixel_raw, cam->distort(cam->undistort(pixel_raw)), 0.5);
    }
}

void expectDznMatchesFiniteDiff(tassel_core::CameraBase* cam, bool skip_center = false) {
    constexpr double kEps = 1e-6;
    constexpr double kTol = 1e-4;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rnd(-1.2, 1.2);

    for (int k = 0; k < 20; ++k) {
        Eigen::Vector2d uv(rnd(rng), rnd(rng));
        if (skip_center && uv.norm() < 0.01) continue;
        Eigen::MatrixXd H_analytic;
        cam->get_jacobian_dzn(uv, H_analytic);
        expectMatrixNear(H_analytic, numerical_dzn(cam, uv, kEps), kTol, uv);
    }
}

template <typename CamType>
void expectDzetaMatchesFiniteDiff(
    tassel_core::CameraBase* cam, const cv::Mat& K, const cv::Mat& D, bool skip_center = false) {
    constexpr double kEps = 1e-5;
    constexpr double kTol = 1e-4;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rnd(-1.2, 1.2);

    for (int k = 0; k < 15; ++k) {
        Eigen::Vector2d uv(rnd(rng), rnd(rng));
        if (skip_center && uv.norm() < 0.01) continue;
        Eigen::MatrixXd H_analytic;
        cam->get_jacobian_dzeta(uv, H_analytic);
        expectMatrixNear(
            H_analytic, numerical_dzeta<CamType>(K, D, kWidth, kHeight, uv, kEps), kTol, uv);
    }
}

}  // namespace

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

    tassel_core::Camera cam_;
};

TEST_F(CameraRadTanTest, UndistortMatchesOpenCV) {
    expectUndistortMatchesOpenCV(
        cam_.get(), radtan_K, radtan_D, kPixels,
        [](auto& cv_in, auto& cv_out, const auto& K, const auto& D, const auto& I) {
            cv::undistortPoints(cv_in, cv_out, K, D, I);
        });
}

TEST_F(CameraRadTanTest, PixelRoundTrip) { expectPixelRoundTrip(cam_.get()); }

TEST_F(CameraRadTanTest, JacobianDZN) { expectDznMatchesFiniteDiff(cam_.get()); }

TEST_F(CameraRadTanTest, JacobianDZeta) {
    expectDzetaMatchesFiniteDiff<tassel_core::CameraRadTan>(cam_.get(), radtan_K, radtan_D);
}

// ── CameraEqui ─────────────────────────────────────────────────────────────

class CameraEquiTest : public ::testing::Test {
protected:
    void SetUp() override {
        cam_ = std::make_unique<tassel_core::CameraEqui>(equi_K, equi_D, kWidth, kHeight);
    }

    tassel_core::Camera cam_;
};

TEST_F(CameraEquiTest, UndistortMatchesOpenCV) {
    expectUndistortMatchesOpenCV(
        cam_.get(), equi_K, equi_D, kPixels,
        [](auto& cv_in, auto& cv_out, const auto& K, const auto& D, const auto& I) {
            cv::fisheye::undistortPoints(cv_in, cv_out, K, D, I);
        });
}

TEST_F(CameraEquiTest, PixelRoundTrip) { expectPixelRoundTrip(cam_.get()); }

TEST_F(CameraEquiTest, ZeroDistortionRoundTrip) {
    cv::Mat zero_D = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 0.0);
    tassel_core::CameraEqui cam_zero(equi_K, zero_D, kWidth, kHeight);

    std::vector<Eigen::Vector2d> norms = {
        {0.001, 0.0}, {-0.001, 0.0}, {0.0, 0.001}, {0.0, -0.001}, {0.0007, 0.0007},
    };
    for (const auto& uv_norm : norms) {
        Eigen::Vector2d uv_dist = cam_zero.distort(uv_norm);
        Eigen::Vector2d uv_norm_back(
            (uv_dist(0) - equi_K.at<double>(0, 2)) / equi_K.at<double>(0, 0),
            (uv_dist(1) - equi_K.at<double>(1, 2)) / equi_K.at<double>(1, 1));
        EXPECT_NEAR(uv_norm(0), uv_norm_back(0), 1e-6);
        EXPECT_NEAR(uv_norm(1), uv_norm_back(1), 1e-6);
    }
}

TEST_F(CameraEquiTest, JacobianDZN) { expectDznMatchesFiniteDiff(cam_.get(), true); }

TEST_F(CameraEquiTest, JacobianDZeta) {
    expectDzetaMatchesFiniteDiff<tassel_core::CameraEqui>(cam_.get(), equi_K, equi_D, true);
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
