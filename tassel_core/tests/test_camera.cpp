// =============================================================================
// test_camera.cpp
//
// 测试思路：
//   1. 先验证 distort(undistort(pixel)) 能还原原始像素，建立畸变与去畸变互逆契约；
//   2. 互逆契约成立后，以 distort 为被测函数，用中心差分验证归一化坐标雅可比；
//   3. 重建仅有一个参数发生扰动的相机，用中心差分验证相机参数雅可比。
//
// OpenCV 已负责去畸变的具体迭代实现，这里不重复验证其内部算法。固定样本覆盖光轴、
// 普通视场和图像边缘，使测试保持确定性，并能定位雅可比的具体行列。
// =============================================================================

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "cam/camera_base.h"
#include "cam/camera_equi.h"
#include "cam/camera_rad_tan.h"

namespace {

using tassel_core::CameraBase;
using tassel_core::CameraEqui;
using tassel_core::CameraRadTan;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr double kDifferenceStep = 1e-5;
constexpr double kJacobianTolerance = 1e-4;
constexpr double kRoundTripTolerance = 0.5;

enum class CameraModel { RadTan, Equidistant };

struct CameraCase {
    CameraModel model;
    const char* name;
    cv::Mat intrinsics;
    cv::Mat distortion;
};

const std::array<Eigen::Vector2d, 5> kPixels = {
    Eigen::Vector2d(320.0, 240.0), Eigen::Vector2d(350.0, 200.0), Eigen::Vector2d(150.0, 300.0),
    Eigen::Vector2d(500.0, 400.0), Eigen::Vector2d(30.0, 40.0)};

const std::array<Eigen::Vector2d, 6> kNormalizedPoints = {
    Eigen::Vector2d::Zero(),   Eigen::Vector2d(0.05, -0.08), Eigen::Vector2d(-0.3, 0.2),
    Eigen::Vector2d(0.6, 0.5), Eigen::Vector2d(-0.9, 0.7),   Eigen::Vector2d(1.1, -1.0)};

std::unique_ptr<CameraBase> createCamera(
    CameraModel model, const cv::Mat& intrinsics, const cv::Mat& distortion) {
    switch (model) {
        case CameraModel::RadTan:
            return std::make_unique<CameraRadTan>(intrinsics, distortion, kWidth, kHeight);
        case CameraModel::Equidistant:
            return std::make_unique<CameraEqui>(intrinsics, distortion, kWidth, kHeight);
    }
    throw std::logic_error("Unknown camera model in test");
}

class CameraModelVerifier {
public:
    explicit CameraModelVerifier(const CameraCase& camera_case)
        : case_(camera_case),
          camera_(createCamera(case_.model, case_.intrinsics, case_.distortion)) {}

    void verify() const {
        // 后续数值微分以 distort 为参考，因此必须先确认它和 undistort 描述同一模型。
        ASSERT_NO_FATAL_FAILURE(verifyDistortionRoundTrip());
        ASSERT_NO_FATAL_FAILURE(verifyCoordinateJacobian());
        ASSERT_NO_FATAL_FAILURE(verifyParameterJacobian());
    }

private:
    void verifyDistortionRoundTrip() const {
        for (const Eigen::Vector2d& pixel : kPixels) {
            const Eigen::Vector2d restored = camera_->distort(camera_->undistort(pixel));
            EXPECT_LE((restored - pixel).cwiseAbs().maxCoeff(), kRoundTripTolerance)
                << "model=" << case_.name << ", pixel=" << pixel.transpose();
        }
    }

    void verifyCoordinateJacobian() const {
        for (const Eigen::Vector2d& point : kNormalizedPoints) {
            const Eigen::Matrix2d numeric = numericalCoordinateJacobian(point);
            Eigen::Matrix2d analytic;
            camera_->get_jacobian_dzn(point, analytic);
            expectMatrixNear(analytic, numeric, point, "dzn");

            Eigen::Vector2d fused_pixel;
            Eigen::Matrix2d fused;
            camera_->distortWithJacobian(point, fused_pixel, fused);
            EXPECT_TRUE(fused_pixel.isApprox(camera_->distort(point), 1e-12));
            expectMatrixNear(fused, numeric, point, "fused dzn");
        }
    }

    void verifyParameterJacobian() const {
        for (const Eigen::Vector2d& point : kNormalizedPoints) {
            Eigen::MatrixXd analytic;
            camera_->get_jacobian_dzeta(point, analytic);
            expectMatrixNear(analytic, numericalParameterJacobian(point), point, "dzeta");
        }
    }

    Eigen::Matrix2d numericalCoordinateJacobian(const Eigen::Vector2d& point) const {
        Eigen::Matrix2d numeric;
        for (int col = 0; col < 2; ++col) {
            Eigen::Vector2d delta = Eigen::Vector2d::Zero();
            delta(col) = kDifferenceStep;
            numeric.col(col) = (camera_->distort(point + delta) - camera_->distort(point - delta)) /
                               (2.0 * kDifferenceStep);
        }
        return numeric;
    }

    Eigen::Matrix<double, 2, 8> numericalParameterJacobian(const Eigen::Vector2d& point) const {
        Eigen::Matrix<double, 2, 8> numeric;
        for (int col = 0; col < 8; ++col) {
            // 参数列固定为 [fx, fy, cx, cy, d0, d1, d2, d3]，与 dzeta 的列布局一致。
            const Eigen::Vector2d positive =
                distortWithParameterOffset(point, col, kDifferenceStep);
            const Eigen::Vector2d negative =
                distortWithParameterOffset(point, col, -kDifferenceStep);
            numeric.col(col) = (positive - negative) / (2.0 * kDifferenceStep);
        }
        return numeric;
    }

    Eigen::Vector2d distortWithParameterOffset(
        const Eigen::Vector2d& point, int parameter, double offset) const {
        cv::Mat intrinsics = case_.intrinsics.clone();
        cv::Mat distortion = case_.distortion.clone();
        if (parameter < 4) {
            constexpr std::array<std::array<int, 2>, 4> kIntrinsicIndices = {
                std::array<int, 2>{0, 0}, {1, 1}, {0, 2}, {1, 2}};
            const auto& index = kIntrinsicIndices[static_cast<size_t>(parameter)];
            intrinsics.at<double>(index[0], index[1]) += offset;
        } else {
            distortion.at<double>(parameter - 4) += offset;
        }
        return createCamera(case_.model, intrinsics, distortion)->distort(point);
    }

    template <typename DerivedA, typename DerivedB>
    void expectMatrixNear(
        const Eigen::MatrixBase<DerivedA>& analytic, const Eigen::MatrixBase<DerivedB>& numeric,
        const Eigen::Vector2d& point, const char* jacobian_name) const {
        ASSERT_EQ(analytic.rows(), numeric.rows());
        ASSERT_EQ(analytic.cols(), numeric.cols());
        for (int row = 0; row < analytic.rows(); ++row) {
            for (int col = 0; col < analytic.cols(); ++col) {
                EXPECT_NEAR(analytic(row, col), numeric(row, col), kJacobianTolerance)
                    << "model=" << case_.name << ", jacobian=" << jacobian_name
                    << ", point=" << point.transpose() << ", element=(" << row << ", " << col
                    << ")";
            }
        }
    }

    const CameraCase& case_;
    std::unique_ptr<CameraBase> camera_;
};

class CameraModelTest : public ::testing::TestWithParam<CameraCase> {};

TEST_P(CameraModelTest, DistortionAndJacobiansAreConsistent) {
    CameraModelVerifier(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(
    CameraModels, CameraModelTest,
    ::testing::Values(
        CameraCase{
            CameraModel::RadTan, "radtan",
            (cv::Mat_<double>(3, 3) << 455.510864, 0.0, 328.529851, 0.0, 455.426715, 225.596721,
             0.0, 0.0, 1.0),
            (cv::Mat_<double>(1, 5) << 0.010831, -0.007841, 0.000166, 0.000512, 0.0)},
        CameraCase{
            CameraModel::Equidistant, "equi",
            (cv::Mat_<double>(3, 3) << 300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0),
            (cv::Mat_<double>(1, 4) << 0.1, 0.01, 0.001, 0.0001)}),
    [](const ::testing::TestParamInfo<CameraCase>& info) { return std::string(info.param.name); });

}  // namespace
