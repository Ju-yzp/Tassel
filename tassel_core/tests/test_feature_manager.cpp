#include <gtest/gtest.h>

#include <limits>

#include "cam/camera_rad_tan.h"
#include "frond_end/feature_manager.h"
#include "state/state.h"

namespace tassel_core {
namespace {

FeaturePerFrame observation(double x = 0.0, double delay = 0.0) {
    FeaturePerFrame result;
    result.setLeft(Eigen::Vector2d(x, 0.0), cv::Point2f(x, 0.0f));
    result.sync_delay = delay;
    return result;
}

FeatureManager manager() { return FeatureManager(3.0, 10.0, 2, 1, 0.0, 0.1, 100.0); }

TEST(FeatureManagerTest, MarginalizationUsesContinuousTargetSlot) {
    auto fm = manager();
    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    auto marginalized = fm.collectMarginalizedObservations(0, 1);
    ASSERT_EQ(marginalized.size(), 1u);
    EXPECT_EQ(marginalized[0].target_slot, 1);
    EXPECT_TRUE(fm.features().at(1).has_been_marginalized);
}

TEST(FeatureManagerTest, TransfersDepthWhenOldestHostLeaves) {
    auto fm = manager();
    State state(3);
    state.newest_slot = 2;
    state.frames[0].P = Eigen::Vector3d::Zero();
    state.frames[1].P = Eigen::Vector3d(0.1, 0.0, 0.0);
    state.frames[2].P = Eigen::Vector3d(0.2, 0.0, 0.0);

    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.05), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    fm.removeOldestFrameObservations(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Feature& transferred = fm.features().at(1);
    EXPECT_EQ(transferred.start_slot, 0);
    ASSERT_EQ(transferred.observations.size(), 2u);
    EXPECT_GT(transferred.estimated_depth, 0.0);
}

TEST(FeatureManagerTest, DepthTransferIncludesGravityAndFrameDelay) {
    State state(2);
    state.newest_slot = 1;
    state.delay_time = 0.1;
    state.frames[0].sync_delay = 0.0;
    state.frames[1].sync_delay = 0.05;
    Feature feature(0, 2);
    feature.estimated_depth = 3.0;
    feature.observations = {observation(0.0, 0.0), observation(0.0, 0.05)};

    ASSERT_TRUE(
        feature.transferHost(1, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
    EXPECT_TRUE(std::isfinite(feature.estimated_depth));
    EXPECT_GT(feature.estimated_depth, 0.0);
}

TEST(FeatureManagerTest, RemovingMiddleSlotCompactsFeatureStart) {
    auto fm = manager();
    State state(4);
    state.newest_slot = 3;
    Feature feature(2, 2);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    fm.removeFrameObservations(1, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    EXPECT_EQ(fm.features().at(1).start_slot, 1);
}

TEST(FeatureManagerTest, MeasuresConnectionToLatestKeyframeSnapshot) {
    auto fm = manager();
    std::unordered_map<int, FeaturePerFrame> keyframe = {
        {1, observation(0.0)}, {2, observation(1.0)}};
    fm.checkParallax(0, keyframe);
    fm.acceptKeyframe(keyframe);

    std::unordered_map<int, FeaturePerFrame> current = {
        {1, observation(2.0)}, {3, observation(3.0)}};
    fm.checkParallax(1, current);
    EXPECT_EQ(fm.lastInputStats().connected_to_keyframe_count, 1u);
    EXPECT_DOUBLE_EQ(fm.lastInputStats().current_keyframe_connection_ratio, 0.5);
}

TEST(FeatureManagerTest, RejectsReappearingFeatureAfterObservationGap) {
    auto fm = manager();
    std::unordered_map<int, FeaturePerFrame> first = {{1, observation()}};
    fm.checkParallax(0, first);

    std::unordered_map<int, FeaturePerFrame> reappearing = {{1, observation(1.0)}};
    EXPECT_THROW(fm.checkParallax(2, reappearing), std::logic_error);
}

TEST(FeatureManagerTest, ReplacingHostKeepsConnectedLandmarkAtSlotZero) {
    auto fm = manager();
    State state(4);
    state.newest_slot = 3;
    state.frames[1].P = Eigen::Vector3d(0.1, 0.0, 0.0);
    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.05), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    fm.replaceRetainedHost(0, 1, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Feature& transferred = fm.features().at(1);
    EXPECT_EQ(transferred.start_slot, 0);
    EXPECT_EQ(transferred.observations.size(), 2u);
    EXPECT_GT(transferred.estimated_depth, 0.0);
}

TEST(FeatureManagerTest, ResetClearsKeyframeSnapshotAndFeatures) {
    auto fm = manager();
    std::unordered_map<int, FeaturePerFrame> frame = {{1, observation()}};
    fm.checkParallax(0, frame);
    fm.acceptKeyframe(frame);
    fm.reset();
    EXPECT_FALSE(fm.hasLatestKeyframe());
    EXPECT_TRUE(fm.features().empty());
}

TEST(FeatureManagerTest, RemovesLandmarkUsingDirectPixelReprojectionError) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0);
    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    CameraRadTan camera(K, D, 100, 80);

    State state(2);
    state.newest_slot = 1;
    state.camera = &camera;

    FeaturePerFrame host;
    host.setLeft(Eigen::Vector2d::Zero(), cv::Point2f(50.0f, 40.0f));
    FeaturePerFrame matching = host;
    FeaturePerFrame outlier = host;
    outlier.pt.x += 10.0f;

    Feature good(0, 2);
    good.estimated_depth = 2.0;
    good.observations = {host, matching};
    Feature bad(0, 2);
    bad.estimated_depth = 2.0;
    bad.observations = {host, outlier};
    Feature invalid(0, 2);
    invalid.estimated_depth = std::numeric_limits<double>::quiet_NaN();
    invalid.observations = {host, matching};

    auto fm = manager();
    fm.features().emplace(1, std::move(good));
    fm.features().emplace(2, std::move(bad));
    fm.features().emplace(3, std::move(invalid));
    fm.removeOutliers(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    EXPECT_TRUE(fm.features().contains(1));
    EXPECT_FALSE(fm.features().contains(2));
    EXPECT_FALSE(fm.features().contains(3));
}

}  // namespace
}  // namespace tassel_core
