#include <gtest/gtest.h>

#include "marg/marg_helper.h"

namespace tassel_core {
namespace {

TEST(MargHelperTest, MovesMarginalizedColumnsBeforeRetainedColumns) {
    Eigen::MatrixXd columns(1, 46);
    for (int i = 0; i < columns.cols(); ++i) {
        columns(0, i) = i;
    }

    const Eigen::MatrixXd create_order =
        MargHelper::reorderForMarginalization(columns, RetainedHostAction::InitializeRetainedSlot);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(create_order(0, i), i);
    }
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(create_order(0, i + 15), i + 21);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(create_order(0, i + 24), i + 15);
    }

    const Eigen::MatrixXd keep_order =
        MargHelper::reorderForMarginalization(columns, RetainedHostAction::MarginalizeOldestFrame);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(keep_order(0, i), i + 15);
        EXPECT_EQ(keep_order(0, i + 15), i);
    }

    const Eigen::MatrixXd replace_order =
        MargHelper::reorderForMarginalization(columns, RetainedHostAction::ReplaceRetainedSlot);
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
