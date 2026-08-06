#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

#include "solver/parameter_block_hold_callback.h"

namespace tassel_core {
namespace {

TEST(ParameterBlockHoldCallbackTest, RestoresEveryHeldSubBlock) {
    std::array<double, 5> first{1.0, 2.0, 3.0, 4.0, 5.0};
    std::array<double, 2> second{6.0, 7.0};
    ParameterBlockHoldCallback callback({{first.data() + 1, 3}, {second.data(), 2}});

    first = {10.0, 20.0, 30.0, 40.0, 50.0};
    second = {60.0, 70.0};
    callback.restore();

    EXPECT_EQ(first, (std::array<double, 5>{10.0, 2.0, 3.0, 4.0, 50.0}));
    EXPECT_EQ(second, (std::array<double, 2>{6.0, 7.0}));
}

TEST(ParameterBlockHoldCallbackTest, RejectsInvalidBlock) {
    EXPECT_THROW(ParameterBlockHoldCallback({{nullptr, 3}}), std::invalid_argument);
    double value = 0.0;
    EXPECT_THROW(ParameterBlockHoldCallback({{&value, 0}}), std::invalid_argument);
}

}  // namespace
}  // namespace tassel_core
