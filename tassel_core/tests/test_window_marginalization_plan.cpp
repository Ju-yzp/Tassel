#include <gtest/gtest.h>

#include "estimator/window_marginalization_plan.h"

namespace tassel_core {
namespace {

TEST(WindowMarginalizationPlanTest, DescribesAllHostTransitions) {
    const auto promote = WindowMarginalizationPlan::create(false, false);
    EXPECT_EQ(promote.transition, HostTransition::kPromote);
    EXPECT_EQ(promote.imu_start_slot, 0);
    EXPECT_EQ(promote.imu_block_count, 2);
    EXPECT_EQ(promote.retained_pose_source_slot, 0);
    EXPECT_EQ(promote.removed_measurement_slot, 1);

    const auto keep = WindowMarginalizationPlan::create(true, false);
    EXPECT_EQ(keep.transition, HostTransition::kKeep);
    EXPECT_EQ(keep.imu_start_slot, 1);
    EXPECT_EQ(keep.imu_block_count, 1);
    EXPECT_EQ(keep.retained_pose_source_slot, 0);
    EXPECT_EQ(keep.removed_measurement_slot, 1);

    const auto replace = WindowMarginalizationPlan::create(true, true);
    EXPECT_EQ(replace.transition, HostTransition::kReplace);
    EXPECT_EQ(replace.imu_start_slot, 1);
    EXPECT_EQ(replace.imu_block_count, 1);
    EXPECT_EQ(replace.retained_pose_source_slot, 1);
    EXPECT_EQ(replace.removed_measurement_slot, 0);
}

TEST(WindowMarginalizationPlanTest, MovesEliminatedColumnsBeforeKeptColumns) {
    Eigen::MatrixXd columns(1, 46);
    for (int i = 0; i < columns.cols(); ++i) columns(0, i) = i;

    const auto keep = WindowMarginalizationPlan::create(true, false);
    const Eigen::MatrixXd keep_order = keep.marginalColumnsFirst(columns);
    for (int i = 0; i < 9; ++i) EXPECT_EQ(keep_order(0, i), i + 6);
    for (int i = 0; i < 15; ++i) EXPECT_EQ(keep_order(0, i + 9), i + 15);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(keep_order(0, i + 24), i);

    const auto replace = WindowMarginalizationPlan::create(true, true);
    const Eigen::MatrixXd replace_order = replace.marginalColumnsFirst(columns);
    for (int i = 0; i < 15; ++i) EXPECT_EQ(replace_order(0, i), i);
    for (int i = 0; i < 9; ++i) EXPECT_EQ(replace_order(0, i + 15), i + 21);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(replace_order(0, i + 24), i + 15);

    EXPECT_EQ(keep_order(0, 45), 45);
    EXPECT_EQ(replace_order(0, 45), 45);
}

}  // namespace
}  // namespace tassel_core
