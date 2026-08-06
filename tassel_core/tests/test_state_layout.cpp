#include <gtest/gtest.h>

#include <vector>

#include "marg/state_layout.h"

namespace tassel_core {
namespace {

TEST(PriorStateLayoutTest, MapsPoseOnlyHostPriorIntoFullWindow) {
    PriorStateLayout layout(3, 6 + 2 * 15 + 1);

    EXPECT_EQ(layout.kind(), PriorLayoutKind::PoseOnlyHost);
    EXPECT_EQ(layout.poseColumn(0), 0);
    EXPECT_FALSE(layout.hasSpeedBias(0));
    EXPECT_EQ(layout.poseColumn(1), 6);
    EXPECT_EQ(layout.speedBiasColumn(1), 12);
    EXPECT_EQ(layout.baColumns(1), (std::array<int, 3>{15, 16, 17}));
    EXPECT_EQ(layout.delayColumn(), 36);

    const std::vector<int> mapping = layout.compactToWindowColumns(4);
    EXPECT_EQ(mapping[0], 0);
    EXPECT_EQ(mapping[5], 5);
    EXPECT_EQ(mapping[6], 15);
    EXPECT_EQ(mapping[20], 29);
    EXPECT_EQ(mapping[21], 30);
    EXPECT_EQ(mapping[35], 44);
    EXPECT_EQ(mapping[36], 60);
}

TEST(PriorStateLayoutTest, BuildsEveryMarginalizationPermutation) {
    constexpr int window_states = 4;
    constexpr int total_columns = window_states * 15 + 1;
    for (RetainedHostAction action :
         {RetainedHostAction::InitializeRetainedSlot, RetainedHostAction::MarginalizeOldestFrame,
          RetainedHostAction::ReplaceRetainedSlot}) {
        const std::vector<int> order = marginalizationColumnOrder(window_states, action);
        ASSERT_EQ(order.size(), static_cast<size_t>(total_columns));
        std::vector<bool> seen(total_columns, false);
        for (int column : order) {
            ASSERT_GE(column, 0);
            ASSERT_LT(column, total_columns);
            EXPECT_FALSE(seen[static_cast<size_t>(column)]);
            seen[static_cast<size_t>(column)] = true;
        }
    }

    const std::vector<int> oldest =
        marginalizationColumnOrder(window_states, RetainedHostAction::MarginalizeOldestFrame);
    EXPECT_EQ(oldest.front(), 15);
    EXPECT_EQ(oldest[14], 29);
    EXPECT_EQ(oldest[15], 0);
}

TEST(PriorStateLayoutTest, RejectsMissingSpeedBiasBlock) {
    PriorStateLayout layout(2, 6 + 15 + 1);
    EXPECT_THROW(layout.baColumns(0), std::logic_error);
    EXPECT_THROW(layout.compactToWindowColumns(1), std::invalid_argument);
}

}  // namespace
}  // namespace tassel_core
