#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "solver/variable_layout.h"

namespace tassel_core {
namespace {

TEST(VariableLayoutTest, PreservesDeclaredJointLayoutAndSeparatesRoles) {
    const VariableKey pose{20, VariableKind::Pose};
    const VariableKey ba{20, VariableKind::AccelBias};
    const VariableKey velocity{10, VariableKind::Velocity};
    const VariableLayout layout({
        {pose, 6, VariableRole::Active},
        {ba, 3, VariableRole::Schmidt},
        {velocity, 3, VariableRole::Fixed},
    });

    EXPECT_EQ(layout.totalSize(), 12);
    EXPECT_EQ(layout.block(pose).offset, 0);
    EXPECT_EQ(layout.block(ba).offset, 6);
    EXPECT_EQ(layout.block(velocity).offset, 9);
    EXPECT_EQ(layout.columns(VariableRole::Active), (std::vector<int>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(layout.columns(VariableRole::Schmidt), (std::vector<int>{6, 7, 8}));
    EXPECT_EQ(layout.columns(VariableRole::Fixed), (std::vector<int>{9, 10, 11}));
}

TEST(VariableLayoutTest, RejectsInvalidContracts) {
    const VariableKey pose{1, VariableKind::Pose};
    EXPECT_THROW(VariableLayout({{pose, 0, VariableRole::Active}}), std::invalid_argument);
    EXPECT_THROW(
        VariableLayout(
            {{pose, 6, VariableRole::Active}, {pose, 6, VariableRole::Schmidt}}),
        std::invalid_argument);

    const VariableLayout layout({{pose, 6, VariableRole::Active}});
    EXPECT_THROW(layout.block({2, VariableKind::Pose}), std::out_of_range);
}

}  // namespace
}  // namespace tassel_core
