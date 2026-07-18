#include <gtest/gtest.h>

#include "frond_end/feature_manager.h"
#include "state/state.h"

namespace tassel_core {
namespace {

FeaturePerFrame makeObservation(tassel_utils::FrameId frame_id) {
    FeaturePerFrame observation;
    observation.frame_id = frame_id;
    observation.uv = Eigen::Vector3d(0.1, -0.1, 1.0);
    observation.pt = cv::Point2f(10.0f, 20.0f);
    return observation;
}

TEST(FeatureManagerTest, OnlyMarginalizationMarksInheritedLandmark) {
    FeatureManager manager(2.0, 10.0, 3, 20, 0.1);
    constexpr tassel_utils::FrameId host_id = 100;
    Feature feature(host_id, 10);
    feature.estimated_depth = 2.0;
    feature.observations = {makeObservation(host_id), makeObservation(200), makeObservation(300)};
    manager.features().emplace(1, std::move(feature));

    auto landmarks = manager.collectLandmarks();
    ASSERT_EQ(landmarks.size(), 1);
    EXPECT_FALSE(landmarks[0]->has_been_marginalized);

    landmarks[0]->observations.pop_back();
    EXPECT_TRUE(manager.collectLandmarks().empty());
    EXPECT_TRUE(manager.collectMarginalizedObservations(host_id, 200).empty());

    landmarks[0]->observations.push_back(makeObservation(300));
    auto marginalized = manager.collectMarginalizedObservations(host_id, 200);
    ASSERT_EQ(marginalized.size(), 1);
    EXPECT_EQ(marginalized[0].feature, landmarks[0]);
    EXPECT_EQ(marginalized[0].target_frame_id, 200);
    EXPECT_TRUE(marginalized[0].feature->has_been_marginalized);

    marginalized[0].feature->observations.pop_back();
    EXPECT_EQ(manager.collectLandmarks().size(), 1);
    EXPECT_EQ(manager.collectMarginalizedObservations(host_id, 200).size(), 1);
    EXPECT_TRUE(manager.collectMarginalizedObservations(host_id, 300).empty());
}

TEST(FeatureManagerTest, TransfersDepthAcrossNonConsecutiveFrameIds) {
    FeatureManager manager(2.0, 10.0, 2, 20, 0.1);
    State state(3);
    state.cur_frame_count = 2;
    state.frame_ids = {100, 300, 500};
    state.acc_vec.assign(3, Eigen::Vector3d::Zero());
    state.gyro_vec.assign(3, Eigen::Vector3d::Zero());

    Feature feature(100, 10);
    feature.estimated_depth = 2.0;
    feature.observations = {makeObservation(100), makeObservation(500)};
    manager.features().emplace(1, std::move(feature));

    manager.removeOldest(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    const auto& transferred = manager.features().at(1);
    EXPECT_EQ(transferred.host_frame_id, 500);
    ASSERT_EQ(transferred.observations.size(), 1);
    EXPECT_EQ(transferred.observations.front().frame_id, 500);
    EXPECT_NEAR(transferred.estimated_depth, 2.0, 1e-12);
}

TEST(FeatureManagerTest, DepthTransferIncludesGravityForDifferentFrameDelays) {
    State state(2);
    state.cur_frame_count = 1;
    state.frame_ids = {100, 200};
    state.acc_vec.assign(2, Eigen::Vector3d::Zero());
    state.gyro_vec.assign(2, Eigen::Vector3d::Zero());
    state.delay_time = 0.1;

    Feature feature(100, 2);
    feature.estimated_depth = 2.0;
    auto old_observation = makeObservation(100);
    old_observation.uv = Eigen::Vector3d(0.0, 0.0, 1.0);
    old_observation.applied_delay = 0.0;
    auto new_observation = makeObservation(200);
    new_observation.uv = Eigen::Vector3d(0.0, 0.0, 1.0);
    new_observation.applied_delay = 0.05;
    feature.observations = {old_observation, new_observation};

    ASSERT_TRUE(
        feature.transferHost(200, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));

    const double dt_old = 0.1;
    const double dt_new = 0.05;
    const double expected_depth =
        2.0 - 0.5 * tassel_utils::G.z() * (dt_old * dt_old - dt_new * dt_new);
    EXPECT_NEAR(feature.estimated_depth, expected_depth, 1e-12);
}

TEST(FeatureManagerTest, RemovesOldestFromTwoFrameWindow) {
    FeatureManager manager(2.0, 10.0, 2, 20, 0.1);
    State state(2);
    state.cur_frame_count = 1;
    state.frame_ids = {100, 200};
    state.acc_vec.assign(2, Eigen::Vector3d::Zero());
    state.gyro_vec.assign(2, Eigen::Vector3d::Zero());

    Feature feature(100, 2);
    feature.estimated_depth = 2.0;
    feature.observations = {makeObservation(100), makeObservation(200)};
    manager.features().emplace(1, std::move(feature));

    manager.removeOldest(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    ASSERT_EQ(manager.features().size(), 1);
    EXPECT_EQ(manager.features().at(1).host_frame_id, 200);
    EXPECT_EQ(manager.features().at(1).observations.size(), 1);
}

TEST(FeatureManagerTest, MeasuresConnectionToLatestKeyframe) {
    FeatureManager manager(2.0, 10.0, 2, 20, 0.1);
    std::unordered_map<int, FeaturePerFrame> keyframe;
    for (int id : {1, 2, 3, 4}) keyframe.emplace(id, makeObservation(100));
    manager.checkParallax(100, keyframe);
    manager.acceptKeyframe(100, keyframe);

    std::unordered_map<int, FeaturePerFrame> current;
    for (int id : {1, 2, 5, 6}) current.emplace(id, makeObservation(200));
    manager.checkParallax(200, current);

    const auto& stats = manager.lastInputStats();
    EXPECT_EQ(stats.connected_to_keyframe_count, 2);
    EXPECT_DOUBLE_EQ(stats.current_keyframe_connection_ratio, 0.5);
    EXPECT_DOUBLE_EQ(stats.keyframe_feature_retention_ratio, 0.5);

    manager.reset();
    EXPECT_FALSE(manager.hasLatestKeyframe());
}

TEST(FeatureManagerTest, RemovingLatestKeyframeClearsKeyframeReference) {
    FeatureManager manager(2.0, 10.0, 2, 20, 0.1);
    std::unordered_map<int, FeaturePerFrame> keyframe;
    keyframe.emplace(1, makeObservation(100));
    manager.checkParallax(100, keyframe);
    manager.acceptKeyframe(100, keyframe);

    State state(2);
    state.cur_frame_count = 1;
    state.frame_ids = {100, 200};
    manager.removeFrame(100, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    EXPECT_FALSE(manager.hasLatestKeyframe());
    EXPECT_FALSE(manager.isKeyframe(100));
}

TEST(FeatureManagerTest, ReplacesHostOnlyForLandmarksConnectedToNewKeyframe) {
    FeatureManager manager(2.0, 10.0, 2, 20, 0.1);
    State state(4);
    state.cur_frame_count = 3;
    state.frame_ids = {100, 200, 300, 400};
    state.acc_vec.assign(4, Eigen::Vector3d::Zero());
    state.gyro_vec.assign(4, Eigen::Vector3d::Zero());

    Feature connected(100, 10);
    connected.estimated_depth = 2.0;
    connected.observations = {makeObservation(100), makeObservation(200), makeObservation(300)};
    manager.features().emplace(1, std::move(connected));

    Feature disconnected(100, 10);
    disconnected.estimated_depth = 2.0;
    disconnected.observations = {makeObservation(100), makeObservation(300)};
    manager.features().emplace(2, std::move(disconnected));

    manager.replaceHost(100, 200, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    ASSERT_EQ(manager.features().size(), 1);
    const auto& transferred = manager.features().at(1);
    EXPECT_EQ(transferred.host_frame_id, 200);
    EXPECT_EQ(transferred.observations.front().frame_id, 200);
    EXPECT_NEAR(transferred.estimated_depth, 2.0, 1e-12);
    for (const auto& observation : transferred.observations) {
        EXPECT_NE(observation.frame_id, 100);
    }
}

}  // namespace
}  // namespace tassel_core
