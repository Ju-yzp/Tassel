#include "estimator/estimator.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace {

TEST(EstimatorKeyframeTest, ReportsLatestDecisionAndClearsItOnReset) {
    const std::filesystem::path config =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "config" /
        "euroc.yaml";
    tassel_tools::Parameters params(config.string());
    auto state = std::make_shared<tassel_core::State>(static_cast<int>(params.max_frame_count));
    auto feature_manager = std::make_shared<tassel_core::FeatureManager>(
        params.reproj_err_thres, params.tracked_times_thres, params.min_translation,
        params.min_depth, params.max_depth);
    tassel_core::Estimator estimator(params, state, feature_manager);

    EXPECT_FALSE(estimator.lastMeasurementWasKeyframe());
    estimator.processMeasurement(1, std::unordered_map<int, tassel_core::FeaturePerFrame>{});
    EXPECT_TRUE(estimator.lastMeasurementWasKeyframe());

    estimator.reset();
    EXPECT_FALSE(estimator.lastMeasurementWasKeyframe());
}

}  // namespace
