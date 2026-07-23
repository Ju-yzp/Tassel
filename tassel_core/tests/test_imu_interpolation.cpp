#include "estimator/imu_interpolation.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

tassel_utils::IMUMeasurement makeMeasurement(double timestamp) {
    tassel_utils::IMUMeasurement measurement;
    measurement.timestamp = timestamp;
    measurement.gyro = Eigen::Vector3d(
        timestamp * timestamp + 2.0 * timestamp + 1.0, -2.0 * timestamp * timestamp + 3.0,
        0.5 * timestamp * timestamp - timestamp);
    measurement.acc = Eigen::Vector3d(
        -timestamp * timestamp + 4.0 * timestamp, 3.0 * timestamp * timestamp - 2.0,
        2.0 * timestamp * timestamp + timestamp + 5.0);
    return measurement;
}

}  // namespace

TEST(ImuInterpolationTest, InterpolatesBodyGyroAndAccelerationTogether) {
    const std::vector<tassel_utils::IMUMeasurement> measurements = {
        makeMeasurement(0.0), makeMeasurement(0.01), makeMeasurement(0.02), makeMeasurement(0.03)};
    constexpr double kQueryTime = 0.015;
    Eigen::Vector3d body_gyro;
    Eigen::Vector3d body_acc;

    tassel_core::interpolateBodyImu(measurements, kQueryTime, body_gyro, body_acc);

    EXPECT_TRUE(body_gyro.isApprox(makeMeasurement(kQueryTime).gyro, 1e-12));
    EXPECT_TRUE(body_acc.isApprox(makeMeasurement(kQueryTime).acc, 1e-12));
}

TEST(ImuInterpolationTest, UsesDocumentedFallbacks) {
    Eigen::Vector3d body_gyro = Eigen::Vector3d::Ones();
    Eigen::Vector3d body_acc = Eigen::Vector3d::Ones();
    tassel_core::interpolateBodyImu({}, 0.0, body_gyro, body_acc);
    EXPECT_TRUE(body_gyro.isZero());
    EXPECT_TRUE(body_acc.isZero());

    std::vector<tassel_utils::IMUMeasurement> measurements = {
        makeMeasurement(0.0), makeMeasurement(0.01)};
    tassel_core::interpolateBodyImu(measurements, 0.005, body_gyro, body_acc);
    EXPECT_TRUE(body_gyro.isApprox(measurements.back().gyro));
    EXPECT_TRUE(body_acc.isApprox(measurements.back().acc));

    measurements = {makeMeasurement(0.0), makeMeasurement(0.0), makeMeasurement(0.01)};
    tassel_core::interpolateBodyImu(measurements, 0.005, body_gyro, body_acc);
    EXPECT_TRUE(body_gyro.isApprox(measurements.back().gyro));
    EXPECT_TRUE(body_acc.isApprox(measurements.back().acc));
}
