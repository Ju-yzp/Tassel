#include <gtest/gtest.h>

#include "behavior/low_speed_ba_behavior.h"

namespace tassel_core {
namespace {

TEST(LowSpeedBaBehaviorTest, UsesWindowConsensusAndExitHysteresis) {
    LowSpeedBaBehavior behavior(0.05, 0.08, 3, 2);
    const std::vector<Eigen::Vector3d> low(4, 0.02 * Eigen::Vector3d::UnitX());
    std::vector<Eigen::Vector3d> mixed = low;
    mixed.back() = 0.04 * Eigen::Vector3d::UnitX();

    EXPECT_EQ(behavior.update(low), BaUpdateMode::Normal);
    EXPECT_FALSE(behavior.holdsBa());
    EXPECT_TRUE(behavior.needsSchmidtCovariance());
    EXPECT_EQ(behavior.update(mixed), BaUpdateMode::Normal);
    EXPECT_EQ(behavior.update(low), BaUpdateMode::LowSpeedHold);
    EXPECT_TRUE(behavior.holdsBa());

    mixed.back() = 0.05 * Eigen::Vector3d::UnitX();
    EXPECT_EQ(behavior.update(mixed), BaUpdateMode::LowSpeedHold);
    mixed.back() = 0.09 * Eigen::Vector3d::UnitX();
    EXPECT_EQ(behavior.update(mixed), BaUpdateMode::LowSpeedHold);
    EXPECT_EQ(behavior.update(mixed), BaUpdateMode::Normal);
    EXPECT_FALSE(behavior.needsSchmidtCovariance());
}

TEST(LowSpeedBaBehaviorTest, InterruptedEntryMustRestartConsensus) {
    LowSpeedBaBehavior behavior(0.05, 0.08, 2, 1);
    const std::vector<Eigen::Vector3d> low(3, Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> moving = low;
    moving[1] = 0.06 * Eigen::Vector3d::UnitX();

    EXPECT_EQ(behavior.update(low), BaUpdateMode::Normal);
    EXPECT_EQ(behavior.update(moving), BaUpdateMode::Normal);
    EXPECT_EQ(behavior.update(low), BaUpdateMode::Normal);
    EXPECT_EQ(behavior.update(low), BaUpdateMode::LowSpeedHold);
}

}  // namespace
}  // namespace tassel_core
