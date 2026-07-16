#include <gtest/gtest.h>

#include "frond_end/feature_manager.h"

namespace tassel_core {
namespace {

FeaturePerFrame makeObservation() {
    FeaturePerFrame observation;
    observation.uv = Eigen::Vector3d(0.1, -0.1, 1.0);
    observation.pt = cv::Point2f(10.0f, 20.0f);
    return observation;
}

TEST(FeatureManagerTest, OnlyMarginalizationMarksInheritedLandmark) {
    FeatureManager manager(2.0, 10.0, 3, 20, 0.1);
    Feature feature(0, 10);
    feature.estimated_depth = 2.0;
    feature.observations = {makeObservation(), makeObservation(), makeObservation()};
    manager.features().emplace(1, std::move(feature));

    auto landmarks = manager.collectLandmarks();
    ASSERT_EQ(landmarks.size(), 1);
    EXPECT_FALSE(landmarks[0]->has_been_marginalized);

    landmarks[0]->observations.pop_back();
    EXPECT_TRUE(manager.collectLandmarks().empty());
    EXPECT_TRUE(manager.collectMargFeatures().empty());

    landmarks[0]->observations.push_back(makeObservation());
    auto marginalized = manager.collectMargFeatures();
    ASSERT_EQ(marginalized.size(), 1);
    EXPECT_TRUE(marginalized[0]->has_been_marginalized);

    marginalized[0]->observations.pop_back();
    EXPECT_EQ(manager.collectLandmarks().size(), 1);
    EXPECT_EQ(manager.collectMargFeatures().size(), 1);
}

}  // namespace
}  // namespace tassel_core
