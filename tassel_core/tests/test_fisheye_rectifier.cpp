#include <gtest/gtest.h>

#include "cam/fisheye_rectifier.h"

namespace tassel_core {
namespace {

TEST(FisheyeRectifierTest, ProducesPinholeImageWithCircularValidRegion) {
    const cv::Size size(200, 120);
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 90.0, 0.0, 100.0, 0.0, 90.0, 60.0, 0.0, 0.0, 1.0);
    const cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.03, -0.008, 0.001, -0.0005);
    FisheyeRectifier rectifier(camera_matrix, distortion, size, 50.0);
    const cv::Mat input(size, CV_8UC1, cv::Scalar(200));

    const cv::Mat output = rectifier.rectify(input);

    EXPECT_EQ(output.type(), CV_8UC1);
    EXPECT_EQ(output.size(), cv::Size(101, 101));
    EXPECT_GT(output.at<uchar>(50, 50), 0);
    EXPECT_EQ(output.at<uchar>(0, 0), 0);
    EXPECT_NEAR(rectifier.cameraMatrix().at<double>(0, 2), 50.0, 1.0);
    EXPECT_NEAR(rectifier.cameraMatrix().at<double>(1, 2), 50.0, 1.0);
    EXPECT_TRUE(cv::checkRange(rectifier.cameraMatrix()));
}

TEST(FisheyeRectifierTest, RejectsWrongInputContract) {
    const cv::Size size(200, 120);
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 90.0, 0.0, 100.0, 0.0, 90.0, 60.0, 0.0, 0.0, 1.0);
    const cv::Mat distortion = cv::Mat::zeros(1, 4, CV_64F);
    FisheyeRectifier rectifier(camera_matrix, distortion, size, 50.0);

    EXPECT_THROW(rectifier.rectify(cv::Mat(size, CV_8UC3)), std::invalid_argument);
}

}  // namespace
}  // namespace tassel_core
