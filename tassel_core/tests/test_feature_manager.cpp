#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <sophus/so3.hpp>

#include "cam/camera_rad_tan.h"
#include "frond_end/feature_manager.h"
#include "frond_end/reprojection.h"
#include "initial/initial_sfm.h"
#include "state/state.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
namespace {

FeaturePerFrame observation(double x = 0.0, double delay = 0.0) {
    FeaturePerFrame result;
    result.setObservation(Eigen::Vector2d(x, 0.0), cv::Point2f(x, 0.0f));
    result.sync_delay = delay;
    return result;
}

FeatureManager manager() { return FeatureManager(3.0, 2, 1e9, 0.75, 0.1, 100.0); }

bool containsId(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

TEST(ReprojectionTest, SplitTransformMatchesComposedTransform) {
    FrameState host;
    host.P = Eigen::Vector3d(0.2, -0.1, 0.3);
    host.V = Eigen::Vector3d(0.4, 0.1, -0.2);
    host.gyro = Eigen::Vector3d(0.1, -0.2, 0.3);
    host.acc = Eigen::Vector3d(0.2, 0.3, 9.7);
    FrameState target;
    target.P = Eigen::Vector3d(0.4, 0.0, 0.2);
    target.V = Eigen::Vector3d(0.3, -0.1, 0.0);
    target.gyro = Eigen::Vector3d(-0.1, 0.2, 0.1);
    target.acc = Eigen::Vector3d(0.1, -0.2, 9.6);

    const Eigen::Vector3d host_uv(0.1, -0.05, 1.0);
    const Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d tic(0.05, 0.0, 0.0);
    Eigen::Vector3d composed_point;
    ASSERT_TRUE(reprojectToTargetCamera(
        host, target, host_uv, 2.0, 0.01, 0.015, 0.02, ric, tic, composed_point));

    Eigen::Vector3d world_point;
    Eigen::Vector3d split_point;
    ASSERT_TRUE(hostPointToWorld(host, host_uv, 2.0, 0.01, 0.02, ric, tic, world_point));
    ASSERT_TRUE(worldPointToTargetCamera(target, world_point, 0.015, 0.02, ric, tic, split_point));
    EXPECT_TRUE(split_point.isApprox(composed_point, 1e-12));
}

TEST(TriangulationTest, ConditionsOnlyTheNonNullSubspace) {
    Eigen::Matrix<double, 3, 4> first = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> second = first;
    second(0, 3) = -1.0;
    const std::vector<Eigen::Matrix<double, 3, 4>> poses = {first, second};
    const std::vector<Eigen::Vector2d> observations = {
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(-0.2, 0.0)};

    double condition = 0.0;
    const Eigen::Vector4d point =
        tassel_utils::triangulateMultiView(poses, observations, &condition);

    EXPECT_TRUE(std::isfinite(condition));
    EXPECT_LT(condition, 1e6);
    EXPECT_NEAR(tassel_utils::dehomogenize(point).z(), 5.0, 1e-12);
}

TEST(TriangulationTest, RejectsDegenerateNonNullSubspace) {
    const Eigen::Matrix<double, 3, 4> pose = Eigen::Matrix<double, 3, 4>::Identity();
    const std::vector<Eigen::Matrix<double, 3, 4>> poses = {pose, pose};
    const std::vector<Eigen::Vector2d> observations(2, Eigen::Vector2d::Zero());

    double condition = 0.0;
    tassel_utils::triangulateMultiView(poses, observations, &condition);

    EXPECT_TRUE(std::isinf(condition));
}

TEST(FeatureManagerTest, CollectsFeatureForSpecifiedMarginalizationTarget) {
    auto fm = manager();
    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    auto marginalized = fm.collectMarginalizedFeatures(0, 1);
    ASSERT_EQ(marginalized.size(), 1u);
    EXPECT_EQ(marginalized[0].first, 1);
    EXPECT_EQ(marginalized[0].second.host_frame_index, 0);
    EXPECT_DOUBLE_EQ(marginalized[0].second.estimated_depth, 2.0);
    ASSERT_EQ(marginalized[0].second.observations.size(), 2u);
    EXPECT_DOUBLE_EQ(marginalized[0].second.observations[1].uv.x(), 0.1);
}

TEST(FeatureManagerTest, CollectsFeatureForAllMarginalizationTargets) {
    auto fm = manager();
    Feature feature(1, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1), observation(0.2)};
    fm.features().emplace(1, std::move(feature));

    auto marginalized = fm.collectMarginalizedFeatures(1);

    ASSERT_EQ(marginalized.size(), 1u);
    EXPECT_EQ(marginalized[0].first, 1);
    EXPECT_EQ(marginalized[0].second.host_frame_index, 1);
    EXPECT_DOUBLE_EQ(marginalized[0].second.estimated_depth, 2.0);
    EXPECT_EQ(marginalized[0].second.observations.size(), 3u);
}

TEST(FeatureManagerTest, CollectsIndependentLandmarkSnapshots) {
    auto fm = manager();
    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(7, std::move(feature));

    auto landmarks = fm.collectLandmarks();

    ASSERT_EQ(landmarks.size(), 1u);
    EXPECT_EQ(landmarks[0].first, 7);
    landmarks[0].second.estimated_depth = 3.0;
    EXPECT_DOUBLE_EQ(fm.features().at(7).estimated_depth, 2.0);
}

TEST(FeatureManagerTest, UpdatesFeatureDepthsById) {
    auto fm = manager();
    fm.features().emplace(3, Feature(0, 2));
    fm.features().emplace(5, Feature(0, 2));

    fm.updateFeatureDepths({{3, 1.5}, {5, Feature::InvalidDepth}});

    EXPECT_DOUBLE_EQ(fm.features().at(3).estimated_depth, 1.5);
    EXPECT_DOUBLE_EQ(fm.features().at(5).estimated_depth, Feature::InvalidDepth);
    EXPECT_THROW(fm.updateFeatureDepths({{9, 2.0}}), std::out_of_range);
}

TEST(FeatureManagerTest, TransfersDepthWhenOldestHostLeaves) {
    auto fm = manager();
    State state(3);
    state.latest_frame_index = 2;
    state.frames[0].P = Eigen::Vector3d::Zero();
    state.frames[1].P = Eigen::Vector3d(0.1, 0.0, 0.0);
    state.frames[2].P = Eigen::Vector3d(0.2, 0.0, 0.0);

    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.05), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    fm.removeFrameObservations(0, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Feature& transferred = fm.features().at(1);
    EXPECT_EQ(transferred.host_frame_index, 0);
    ASSERT_EQ(transferred.observations.size(), 2u);
    EXPECT_GT(transferred.estimated_depth, 0.0);
}

TEST(FeatureManagerTest, DepthTransferIncludesGravityAndFrameDelay) {
    State state(2);
    state.latest_frame_index = 1;
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

TEST(FeatureManagerTest, RemovingMiddleFrameCompactsFeatureHostIndex) {
    auto fm = manager();
    State state(4);
    state.latest_frame_index = 3;
    Feature feature(2, 2);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    fm.removeFrameObservations(1, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    EXPECT_EQ(fm.features().at(1).host_frame_index, 1);
}

TEST(FeatureManagerTest, KeepsFrameNonKeyframeWhenConnectionsAreSufficient) {
    FeatureManager fm(3.0, 2, 1e9, 0.5, 0.1, 100.0);
    EXPECT_TRUE(fm.addFeatureFrame(0, {{1, observation(0.0)}}));
    EXPECT_FALSE(fm.addFeatureFrame(1, {{1, observation(20.0)}}));

    EXPECT_FALSE(fm.addFeatureFrame(2, {{1, observation(40.0)}}));
}

TEST(FeatureManagerTest, ReplacesInitializationCandidateUntilConnectionsAreInsufficient) {
    FeatureManager fm(3.0, 2, 1e9, 0.5, 0.1, 100.0);
    EXPECT_TRUE(fm.addFeatureFrame(
        1, {{1, observation(0.0)}, {2, observation(0.0)}, {3, observation(0.0)}}));

    EXPECT_FALSE(fm.replaceInitializationCandidate(
        1, 2,
        {{1, observation(20.0)},
         {2, observation(20.0)},
         {3, observation(20.0)},
         {4, observation(20.0)}}));
    ASSERT_EQ(fm.features().at(1).observations.size(), 2u);
    EXPECT_DOUBLE_EQ(fm.features().at(1).observations.back().pt.x, 20.0);

    EXPECT_TRUE(
        fm.replaceInitializationCandidate(1, 2, {{1, observation(40.0)}, {5, observation(40.0)}}));
    ASSERT_EQ(fm.features().at(1).observations.size(), 2u);
    EXPECT_DOUBLE_EQ(fm.features().at(1).observations.back().pt.x, 40.0);
    EXPECT_FALSE(fm.features().contains(4));
    EXPECT_TRUE(fm.features().contains(5));
}

TEST(FeatureManagerTest, AcceptsInitializationCandidateWhenParallaxIsSufficient) {
    FeatureManager fm(3.0, 2, 10.0, 0.5, 0.1, 100.0);
    EXPECT_TRUE(fm.addFeatureFrame(
        1, {{1, observation(0.0)}, {2, observation(0.0)}, {3, observation(0.0)}}));

    EXPECT_TRUE(fm.replaceInitializationCandidate(
        1, 2, {{1, observation(12.0)}, {2, observation(12.0)}, {3, observation(12.0)}}));
}

TEST(FeatureManagerTest, AcceptsInitializationCandidateWhenKeyframeConnectionsAreLost) {
    FeatureManager fm(3.0, 2, 1e9, 0.75, 0.1, 100.0);
    EXPECT_TRUE(fm.addFeatureFrame(
        1, {{1, observation()}, {2, observation()}, {3, observation()}, {4, observation()}}));

    EXPECT_TRUE(fm.replaceInitializationCandidate(1, 2, {{1, observation()}, {2, observation()}}));
}

TEST(FeatureManagerTest, CreatesKeyframeWhenPreviousKeyframeConnectionsAreLost) {
    FeatureManager fm(3.0, 2, 1e9, 0.75, 0.1, 100.0);
    EXPECT_TRUE(fm.addFeatureFrame(
        0, {{1, observation()}, {2, observation()}, {3, observation()}, {4, observation()}}));

    EXPECT_TRUE(fm.addFeatureFrame(1, {{1, observation()}, {2, observation()}}));
}

TEST(FeatureManagerTest, RejectsReappearingFeatureAfterObservationGap) {
    auto fm = manager();
    std::unordered_map<int, FeaturePerFrame> first = {{1, observation()}};
    fm.addFeatureFrame(0, first);

    std::unordered_map<int, FeaturePerFrame> reappearing = {{1, observation(1.0)}};
    EXPECT_THROW(fm.addFeatureFrame(2, reappearing), std::logic_error);
}

TEST(FeatureManagerTest, RejectsTriangulationObservationOutsideActiveWindow) {
    auto fm = manager();
    State state(2);
    state.latest_frame_index = 0;
    Feature feature(0, 2);
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    EXPECT_THROW(
        fm.triangulate(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()),
        std::logic_error);
}

TEST(FeatureManagerTest, TriangulatesDepthBeyondExportLimit) {
    FeatureManager fm(3.0, 2, 1e9, 0.75, 0.1, 10.0);
    State state(3);
    state.latest_frame_index = 2;
    state.frames[1].P = Eigen::Vector3d(1.0, 0.0, 0.0);
    state.frames[2].P = Eigen::Vector3d(2.0, 0.0, 0.0);

    Feature feature(0, 3);
    feature.observations = {observation(0.0), observation(-0.05), observation(-0.1001)};
    fm.features().emplace(1, std::move(feature));

    fm.triangulate(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    EXPECT_NEAR(fm.features().at(1).estimated_depth, 20.0, 0.1);
}

TEST(FeatureManagerTest, TriangulationUsesThirdOrderFrameCompensation) {
    State state(3);
    state.latest_frame_index = 2;
    state.delay_time = 0.12;
    state.frames[0].P = Eigen::Vector3d(0.0, 0.0, 0.0);
    state.frames[1].P = Eigen::Vector3d(0.35, 0.02, -0.01);
    state.frames[2].P = Eigen::Vector3d(0.72, -0.01, 0.03);
    for (int i = 0; i < 3; ++i) {
        auto& frame = state.frames[i];
        frame.V = Eigen::Vector3d(0.4 + 0.1 * i, -0.15, 0.08);
        frame.gyro = Eigen::Vector3d(0.7, -0.4 + 0.1 * i, 0.5);
        frame.acc = Eigen::Vector3d(1.2, -0.6, 9.3);
        frame.Bg = Eigen::Vector3d(0.02, -0.01, 0.03);
        frame.Ba = Eigen::Vector3d(0.08, -0.04, 0.05);
        frame.sync_delay = 0.01 + 0.02 * i;
    }

    const Eigen::Matrix3d ric = Sophus::SO3d::exp(Eigen::Vector3d(0.03, -0.02, 0.01)).matrix();
    const Eigen::Vector3d tic(0.06, -0.01, 0.02);
    const Eigen::Vector3d host_uv(0.1, -0.06, 1.0);
    constexpr double depth = 3.5;
    Feature feature(0, 3);
    feature.observations.push_back(observation(host_uv.x(), state.frames[0].sync_delay));
    feature.observations.back().uv = host_uv;
    for (int target_index = 1; target_index <= 2; ++target_index) {
        Eigen::Vector3d target_point;
        ASSERT_TRUE(reprojectToTargetCamera(
            state.frames[0], state.frames[target_index], host_uv, depth, state.frames[0].sync_delay,
            state.frames[target_index].sync_delay, state.delay_time, ric, tic, target_point));
        FeaturePerFrame target_observation;
        const Eigen::Vector2d target_uv = target_point.head<2>() / target_point.z();
        target_observation.setObservation(target_uv, cv::Point2f(target_uv.x(), target_uv.y()));
        target_observation.sync_delay = state.frames[target_index].sync_delay;
        feature.observations.push_back(target_observation);
    }

    feature.monoTriangulate(state, ric, tic, 0.1);

    EXPECT_NEAR(feature.estimated_depth, depth, 1e-9);
}

TEST(FeatureManagerTest, RejectsTriangulatedDepthBelowThreshold) {
    FeatureManager fm(3.0, 2, 1e9, 0.75, 0.3, 10.0);
    State state(2);
    state.latest_frame_index = 1;
    state.frames[1].P = Eigen::Vector3d(0.1, 0.0, 0.0);
    Feature feature(0, 2);
    feature.observations = {observation(0.0), observation(-0.5)};
    fm.features().emplace(1, std::move(feature));

    fm.triangulate(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    EXPECT_EQ(fm.features().at(1).estimated_depth, Feature::InvalidDepth);
}

TEST(FeatureManagerTest, RejectsSfmObservationOutsideActiveWindow) {
    auto fm = manager();
    State state(2);
    state.latest_frame_index = 0;
    Feature feature(0, 2);
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    EXPECT_THROW(fm.collectSFMFeatures(state), std::logic_error);
}

TEST(FeatureManagerTest, ReplacingHostKeepsConnectedLandmarkAtIndexZero) {
    auto fm = manager();
    State state(4);
    state.latest_frame_index = 3;
    state.frames[1].P = Eigen::Vector3d(0.1, 0.0, 0.0);
    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.05), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    fm.replaceRetainedHost(0, 1, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Feature& transferred = fm.features().at(1);
    EXPECT_EQ(transferred.host_frame_index, 0);
    EXPECT_EQ(transferred.observations.size(), 2u);
    EXPECT_GT(transferred.estimated_depth, 0.0);
}

TEST(FeatureManagerTest, ResetClearsKeyframeSnapshotAndFeatures) {
    auto fm = manager();
    std::unordered_map<int, FeaturePerFrame> frame = {{1, observation()}};
    fm.addFeatureFrame(0, frame);
    fm.reset();
    EXPECT_FALSE(fm.hasLatestKeyframe());
    EXPECT_TRUE(fm.features().empty());
}

TEST(FeatureManagerTest, RemovesLandmarkUsingDirectPixelReprojectionError) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0);
    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    CameraRadTan camera(K, D, 100, 80);

    State state(2);
    state.latest_frame_index = 1;
    state.camera = &camera;

    FeaturePerFrame host;
    host.setObservation(Eigen::Vector2d::Zero(), cv::Point2f(50.0f, 40.0f));
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
    const std::vector<int> removed_ids =
        fm.removeOutliers(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    EXPECT_TRUE(fm.features().contains(1));
    EXPECT_FALSE(fm.features().contains(2));
    EXPECT_FALSE(fm.features().contains(3));
    EXPECT_EQ(removed_ids.size(), 2u);
    EXPECT_TRUE(containsId(removed_ids, 2));
    EXPECT_TRUE(containsId(removed_ids, 3));
}

TEST(FeatureManagerTest, RemovesReprojectionOutlierBeforeLaterFrames) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0);
    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    CameraRadTan camera(K, D, 100, 80);
    State state(2);
    state.latest_frame_index = 1;
    state.camera = &camera;

    FeaturePerFrame matching;
    matching.setObservation(Eigen::Vector2d::Zero(), cv::Point2f(50.0f, 40.0f));
    FeaturePerFrame outlier = matching;
    outlier.pt.x += 10.0f;

    auto fm = manager();
    EXPECT_TRUE(fm.addFeatureFrame(0, {{1, matching}, {2, matching}}));
    EXPECT_FALSE(fm.addFeatureFrame(1, {{1, matching}, {2, outlier}}));
    fm.features().at(1).estimated_depth = 2.0;
    fm.features().at(2).estimated_depth = 2.0;
    const std::vector<int> removed_ids =
        fm.removeOutliers(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    ASSERT_FALSE(fm.features().contains(2));
    ASSERT_EQ(removed_ids.size(), 1u);
    EXPECT_EQ(removed_ids.front(), 2);

    EXPECT_FALSE(fm.addFeatureFrame(2, {{1, matching}}));
    EXPECT_TRUE(fm.features().contains(1));
    EXPECT_FALSE(fm.features().contains(2));
}

TEST(FeatureManagerTest, ExportsValidLandmarksForRequestedHostAsIndependentValues) {
    auto fm = manager();
    State state(3);
    state.latest_frame_index = 2;
    state.frames[1].R = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    state.frames[1].P = Eigen::Vector3d(4.0, 5.0, 6.0);
    Feature valid(1, 2);
    valid.estimated_depth = 2.0;
    valid.observations = {observation(0.25)};
    valid.observations.front().pt = cv::Point2f(120.0f, 80.0f);
    fm.features().emplace(7, std::move(valid));

    Feature other_host(0, 2);
    other_host.estimated_depth = 2.0;
    other_host.observations = {observation(0.5)};
    fm.features().emplace(8, std::move(other_host));

    Feature invalid_depth(1, 2);
    invalid_depth.estimated_depth = std::numeric_limits<double>::quiet_NaN();
    invalid_depth.observations = {observation(0.75)};
    fm.features().emplace(9, std::move(invalid_depth));

    std::vector<HostLandmark> landmarks = fm.exportHostLandmarks(1, state);
    ASSERT_EQ(landmarks.size(), 1u);
    EXPECT_EQ(landmarks.front().feature_id, 7);
    EXPECT_EQ(landmarks.front().host_pixel, cv::Point2f(120.0f, 80.0f));
    EXPECT_TRUE(landmarks.front().host_uv.isApprox(Eigen::Vector3d(0.25, 0.0, 1.0)));
    EXPECT_DOUBLE_EQ(landmarks.front().host_depth, 2.0);

    fm.features().clear();
    EXPECT_EQ(landmarks.front().feature_id, 7);
    EXPECT_TRUE(landmarks.front().host_uv.isApprox(Eigen::Vector3d(0.25, 0.0, 1.0)));
    EXPECT_DOUBLE_EQ(landmarks.front().host_depth, 2.0);
}

TEST(FeatureManagerTest, ExportsObservedHostDepthAsWorldLandmark) {
    auto fm = manager();
    State state(2);
    state.latest_frame_index = 1;
    state.frames[0].R = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    state.frames[0].P = Eigen::Vector3d(4.0, 5.0, 6.0);
    Feature feature(0, 2);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(0.25), observation(0.3)};
    fm.features().emplace(7, std::move(feature));

    Feature not_observed(0, 1);
    not_observed.estimated_depth = 2.0;
    not_observed.observations = {observation(0.5)};
    fm.features().emplace(8, std::move(not_observed));

    const auto landmarks = fm.exportObservedWorldLandmarks(
        1, state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    ASSERT_EQ(landmarks.size(), 1u);
    EXPECT_TRUE(landmarks.at(7).isApprox(Eigen::Vector3d(4.0, 5.5, 8.0)));
}

TEST(FeatureManagerTest, MapsReservedSlotIndicesToCompactSfmIndices) {
    auto fm = manager();
    State state(4);
    state.latest_frame_index = 3;

    Feature feature(1, 3);
    feature.observations = {observation(0.1), observation(0.2), observation(0.3)};
    fm.features().emplace(7, std::move(feature));

    const std::vector<SFMFeature> features = fm.collectSFMFeatures(state, 1);
    ASSERT_EQ(features.size(), 1u);
    ASSERT_EQ(features.front().observation.size(), 3u);
    EXPECT_EQ(features.front().observation[0].first, 0);
    EXPECT_EQ(features.front().observation[1].first, 1);
    EXPECT_EQ(features.front().observation[2].first, 2);
}

TEST(FeatureManagerTest, ExcludesDepthOutsideConfiguredRange) {
    auto fm = manager();
    State state(1);
    state.latest_frame_index = 0;
    for (const auto& [id, depth] :
         std::vector<std::pair<int, double>>{{1, -1.0}, {2, 0.05}, {3, 101.0}}) {
        Feature feature(0, 1);
        feature.estimated_depth = depth;
        feature.observations = {observation()};
        fm.features().emplace(id, std::move(feature));
    }
    EXPECT_TRUE(fm.exportHostLandmarks(0, state).empty());
}

}  // namespace
}  // namespace tassel_core
