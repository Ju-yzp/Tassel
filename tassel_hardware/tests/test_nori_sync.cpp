#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "nori/nori_sync.h"

namespace tassel_hardware {
namespace {

tassel_utils::IMUMeasurement imu(double timestamp, double value) {
    return {Eigen::Vector3d::Constant(value), Eigen::Vector3d::Constant(-value), timestamp};
}

TEST(NoriSyncTest, SharesInterpolatedBoundaryBetweenAdjacentFrames) {
    NoriSync sync(0.0);
    sync.push({tassel_utils::secondsToFrameId(0.015), {}}, {imu(0.010, 1.0), imu(0.020, 2.0)});

    NoriSyncedFrame first;
    ASSERT_TRUE(sync.waitPop(first));
    ASSERT_EQ(first.imu.size(), 2u);
    EXPECT_DOUBLE_EQ(first.imu.back().timestamp, 0.015);
    EXPECT_NEAR(first.imu.back().acc.x(), 1.5, 1e-12);

    sync.push({tassel_utils::secondsToFrameId(0.025), {}}, {imu(0.030, 3.0)});
    NoriSyncedFrame second;
    ASSERT_TRUE(sync.waitPop(second));
    ASSERT_EQ(second.imu.size(), 3u);
    EXPECT_DOUBLE_EQ(second.imu.front().timestamp, first.imu.back().timestamp);
    EXPECT_DOUBLE_EQ(second.imu.back().timestamp, 0.025);
    EXPECT_NEAR(second.imu.back().acc.x(), 2.5, 1e-12);
}

TEST(NoriSyncTest, RejectsNonFiniteDelay) {
    EXPECT_THROW(NoriSync(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

}  // namespace
}  // namespace tassel_hardware
