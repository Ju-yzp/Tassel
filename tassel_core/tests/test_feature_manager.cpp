#include <gtest/gtest.h>

#include <limits>

#include "cam/camera_rad_tan.h"
#include "frond_end/feature_manager.h"
#include "frond_end/reprojection.h"
#include "initial/initial_sfm.h"
#include "state/state.h"

namespace tassel_core {
namespace {

FeaturePerFrame observation(double x = 0.0, double delay = 0.0) {
    FeaturePerFrame result;
    result.setLeft(Eigen::Vector2d(x, 0.0), cv::Point2f(x, 0.0f));
    result.sync_delay = delay;
    return result;
}

FeatureManager manager() { return FeatureManager(3.0, 2, 0.0, 0.25, 0.1, 100.0); }

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

TEST(FeatureManagerTest, MarginalizationUsesContinuousTargetFrameIndex) {
    auto fm = manager();
    Feature feature(0, 4);
    feature.estimated_depth = 2.0;
    feature.observations = {observation(), observation(0.1)};
    fm.features().emplace(1, std::move(feature));

    auto marginalized = fm.collectMarginalizedObservations(0, 1);
    ASSERT_EQ(marginalized.size(), 1u);
    EXPECT_EQ(marginalized[0].target_frame_index, 1);
    EXPECT_TRUE(fm.features().at(1).has_been_marginalized);
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

TEST(FeatureManagerTest, ClassifiesFrameFromLatestKeyframeConnection) {
    auto fm = manager();
    std::unordered_map<int, FeaturePerFrame> keyframe = {
        {1, observation(0.0)}, {2, observation(1.0)}};
    EXPECT_TRUE(fm.addFeatureFrame(0, keyframe));

    std::unordered_map<int, FeaturePerFrame> current = {
        {1, observation(2.0)}, {2, observation(3.0)}};
    EXPECT_FALSE(fm.addFeatureFrame(1, current));
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

TEST(FeatureManagerTest, RecreatedOutlierKeepsKeyframeSnapshotConnection) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0);
    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    CameraRadTan camera(K, D, 100, 80);
    State state(2);
    state.latest_frame_index = 1;
    state.camera = &camera;

    FeaturePerFrame matching;
    matching.setLeft(Eigen::Vector2d::Zero(), cv::Point2f(50.0f, 40.0f));
    FeaturePerFrame outlier = matching;
    outlier.pt.x += 10.0f;

    auto fm = manager();
    EXPECT_TRUE(fm.addFeatureFrame(0, {{1, matching}, {2, matching}}));
    EXPECT_FALSE(fm.addFeatureFrame(1, {{1, matching}, {2, outlier}}));
    fm.features().at(1).estimated_depth = 2.0;
    fm.features().at(2).estimated_depth = 2.0;
    fm.removeOutliers(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    ASSERT_FALSE(fm.features().contains(2));

    EXPECT_FALSE(fm.addFeatureFrame(2, {{1, matching}, {2, matching}}));
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
