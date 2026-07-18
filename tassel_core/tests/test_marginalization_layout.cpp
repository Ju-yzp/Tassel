#include <gtest/gtest.h>

#include "estimator/marginalization_layout.h"

namespace tassel_core {
namespace {

TEST(MarginalizationLayoutTest, DescribesEachRetainedHostAction) {
    const MarginalizationLayout create(RetainedHostAction::kCreate);
    EXPECT_FALSE(create.replacesRetainedHost());
    EXPECT_TRUE(create.eliminatesSlot0Motion());
    EXPECT_EQ(create.eliminatedParameterSize(), 24);
    EXPECT_EQ(create.firstImuFactorSlot(), 0);
    EXPECT_EQ(create.imuFactorCount(), 2);
    EXPECT_EQ(create.nextRetainedHostSourceSlot(), 0);

    const MarginalizationLayout keep(RetainedHostAction::kKeep);
    EXPECT_FALSE(keep.replacesRetainedHost());
    EXPECT_FALSE(keep.eliminatesSlot0Motion());
    EXPECT_EQ(keep.eliminatedParameterSize(), 15);
    EXPECT_EQ(keep.firstImuFactorSlot(), 1);
    EXPECT_EQ(keep.imuFactorCount(), 1);
    EXPECT_EQ(keep.nextRetainedHostSourceSlot(), 0);

    const MarginalizationLayout replace(RetainedHostAction::kReplace);
    EXPECT_TRUE(replace.replacesRetainedHost());
    EXPECT_FALSE(replace.eliminatesSlot0Motion());
    EXPECT_EQ(replace.eliminatedParameterSize(), 24);
    EXPECT_EQ(replace.firstImuFactorSlot(), 1);
    EXPECT_EQ(replace.imuFactorCount(), 1);
    EXPECT_EQ(replace.nextRetainedHostSourceSlot(), 1);
}

TEST(MarginalizationLayoutTest, MovesEliminatedColumnsBeforeRetainedColumns) {
    Eigen::MatrixXd columns(1, 46);
    for (int i = 0; i < columns.cols(); ++i) {
        columns(0, i) = i;
    }

    const MarginalizationLayout create(RetainedHostAction::kCreate);
    const Eigen::MatrixXd create_order = create.reorderForElimination(columns);
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(create_order(0, i), i + 6);
    }
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(create_order(0, i + 9), i + 15);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(create_order(0, i + 24), i);
    }

    const MarginalizationLayout keep(RetainedHostAction::kKeep);
    const Eigen::MatrixXd keep_order = keep.reorderForElimination(columns);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(keep_order(0, i), i + 15);
    }
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(keep_order(0, i + 15), i);
    }

    const MarginalizationLayout replace(RetainedHostAction::kReplace);
    const Eigen::MatrixXd replace_order = replace.reorderForElimination(columns);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(replace_order(0, i), i);
    }
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(replace_order(0, i + 15), i + 21);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(replace_order(0, i + 24), i + 15);
    }

    EXPECT_EQ(create_order(0, 45), 45);
    EXPECT_EQ(keep_order(0, 45), 45);
    EXPECT_EQ(replace_order(0, 45), 45);
}

}  // namespace
}  // namespace tassel_core
