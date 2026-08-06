#include "parameters/parameters.h"

#include <gtest/gtest.h>

namespace {

using tassel_tools::parseTrustRegionStrategy;
using tassel_tools::TrustRegionStrategy;

TEST(TrustRegionStrategyTest, ParsesSupportedStrategies) {
    EXPECT_EQ(
        parseTrustRegionStrategy(" levenberg_marquardt "), TrustRegionStrategy::LevenbergMarquardt);
    EXPECT_EQ(parseTrustRegionStrategy("DOGLEG"), TrustRegionStrategy::Dogleg);
}

TEST(TrustRegionStrategyTest, RejectsUnsupportedStrategy) {
    EXPECT_THROW(parseTrustRegionStrategy(""), std::runtime_error);
    EXPECT_THROW(parseTrustRegionStrategy("gauss_newton"), std::runtime_error);
}

}  // namespace
