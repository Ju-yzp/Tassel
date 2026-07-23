#include "pose_graph.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

TEST(PoseGraphTest, BuildsPriorOdometryAndPoseLoopWithoutOptimization) {
    tassel_loop::PoseGraph graph;
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    graph.addKeyframe(200, Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0));
    graph.addKeyframe(300, rotation, Eigen::Vector3d(2.0, 0.3, 0.1));

    ASSERT_TRUE(graph.addPoseLoop(100, 300, rotation, Eigen::Vector3d(2.0, 0.3, 0.1)));

    const tassel_loop::PoseGraphStats stats = graph.stats();
    EXPECT_EQ(stats.node_count, 3U);
    EXPECT_EQ(stats.odometry_factor_count, 2U);
    EXPECT_EQ(stats.loop_factor_count, 1U);
    EXPECT_EQ(graph.factors().size(), 4U);
    EXPECT_EQ(graph.values().size(), 3U);
    EXPECT_NEAR(graph.factors().error(graph.values()), 0.0, 1e-10);
}

TEST(PoseGraphTest, RejectsDuplicateNodesAndUnknownLoopFrames) {
    tassel_loop::PoseGraph graph;
    EXPECT_FALSE(graph.contains(100));
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    EXPECT_TRUE(graph.contains(100));
    EXPECT_THROW(
        graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()),
        std::invalid_argument);
    EXPECT_FALSE(
        graph.addPoseLoop(100, 200, Eigen::Matrix3d::Identity(), Eigen::Vector3d::UnitX()));
    EXPECT_FALSE(graph.optimize());
    EXPECT_FALSE(graph.pose(200).has_value());
}

TEST(PoseGraphTest, OptimizesAndReturnsPoseByFrameId) {
    tassel_loop::PoseGraph graph;
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    graph.addKeyframe(200, Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0));
    graph.addKeyframe(300, Eigen::Matrix3d::Identity(), Eigen::Vector3d(2.0, 0.2, 0.0));
    ASSERT_TRUE(graph.addPoseLoop(
        100, 300, Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
        Eigen::Vector3d(2.0, 0.0, 0.0)));

    const double error_before = graph.factors().error(graph.values());
    ASSERT_TRUE(graph.optimize());
    const double error_after = graph.factors().error(graph.values());
    EXPECT_LE(error_after, error_before);
    EXPECT_TRUE(graph.stats().optimized);
    EXPECT_DOUBLE_EQ(graph.lastOptimizationStats().error_before, error_before);
    EXPECT_DOUBLE_EQ(graph.lastOptimizationStats().error_after, error_after);
    EXPECT_GT(graph.lastOptimizationStats().max_translation_correction, 0.0);
    EXPECT_GT(graph.lastOptimizationStats().max_rotation_correction, 0.0);

    const auto optimized_pose = graph.pose(300);
    ASSERT_TRUE(optimized_pose.has_value());
    EXPECT_TRUE(optimized_pose->world_R_camera.allFinite());
    EXPECT_TRUE(optimized_pose->world_t_camera.allFinite());

    const auto poses = graph.poses();
    ASSERT_EQ(poses.size(), 3u);
    EXPECT_EQ(poses[0].first, 100);
    EXPECT_EQ(poses[1].first, 200);
    EXPECT_EQ(poses[2].first, 300);
    EXPECT_TRUE(poses[2].second.world_t_camera.isApprox(optimized_pose->world_t_camera));
}

TEST(PoseGraphTest, AddsRobustSixDofPoseLoop) {
    tassel_loop::PoseGraph graph;
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    graph.addKeyframe(200, Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0));
    graph.addKeyframe(300, Eigen::Matrix3d::Identity(), Eigen::Vector3d(2.1, 0.1, 0.0));
    const Eigen::Matrix3d candidate_R_current =
        Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    ASSERT_TRUE(graph.addPoseLoop(100, 300, candidate_R_current, Eigen::Vector3d(2.0, 0.0, 0.0)));
    EXPECT_EQ(graph.stats().loop_factor_count, 1u);
    EXPECT_EQ(graph.factors().size(), 4u);
    EXPECT_TRUE(graph.optimize());
    EXPECT_TRUE(graph.pose(300)->world_t_camera.allFinite());
}

TEST(PoseGraphTest, ExtendsOptimizedGraphUsingLocalOdometryIncrement) {
    tassel_loop::PoseGraph graph;
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    graph.addKeyframe(200, Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0));
    graph.addKeyframe(300, Eigen::Matrix3d::Identity(), Eigen::Vector3d(2.1, 0.0, 0.0));
    ASSERT_TRUE(graph.addPoseLoop(
        100, 300, Eigen::Matrix3d::Identity(), Eigen::Vector3d(2.0, 0.0, 0.0),
        tassel_loop::PoseLoopNoise{1e-4, 1e-4}));
    ASSERT_TRUE(graph.optimize());

    const Eigen::Vector3d optimized_before = graph.pose(300)->world_t_camera;
    const double error_before = graph.factors().error(graph.values());
    graph.addKeyframe(400, Eigen::Matrix3d::Identity(), Eigen::Vector3d(3.1, 0.0, 0.0));
    const Eigen::Vector3d optimized_after = graph.pose(400)->world_t_camera;

    EXPECT_TRUE((optimized_after - optimized_before).isApprox(Eigen::Vector3d::UnitX(), 1e-9));
    EXPECT_NEAR(graph.factors().error(graph.values()), error_before, 1e-9);
}

TEST(PoseGraphTest, RejectsInvalidPoseLoop) {
    tassel_loop::PoseGraph graph;
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    graph.addKeyframe(200, Eigen::Matrix3d::Identity(), Eigen::Vector3d::UnitX());
    EXPECT_FALSE(
        graph.addPoseLoop(100, 300, Eigen::Matrix3d::Identity(), Eigen::Vector3d::UnitX()));
    EXPECT_THROW(
        graph.addPoseLoop(100, 200, Eigen::Matrix3d::Zero(), Eigen::Vector3d::UnitX()),
        std::invalid_argument);
}

TEST(PoseGraphTest, LimitsLoopInformationAndRejectsInconsistentLoop) {
    tassel_loop::PoseGraph graph;
    graph.addKeyframe(100, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    graph.addKeyframe(200, Eigen::Matrix3d::Identity(), Eigen::Vector3d::UnitX());
    graph.addKeyframe(300, Eigen::Matrix3d::Identity(), 2.0 * Eigen::Vector3d::UnitX());
    ASSERT_TRUE(graph.addPoseLoop(
        100, 300, Eigen::Matrix3d::Identity(), 20.0 * Eigen::Vector3d::UnitX(),
        tassel_loop::PoseLoopNoise{1e-12, 1e-12}));

    EXPECT_FALSE(graph.optimize(3.0));
    EXPECT_TRUE(graph.lastOptimizationStats().loop_rejected);
    EXPECT_GT(graph.lastOptimizationStats().max_normalized_loop_error, 3.0);
    EXPECT_DOUBLE_EQ(graph.lastOptimizationStats().loop_translation_variance, 0.0025);
    EXPECT_DOUBLE_EQ(graph.lastOptimizationStats().loop_rotation_variance, 0.0004);
    EXPECT_EQ(graph.stats().loop_factor_count, 0u);
    EXPECT_EQ(graph.factors().size(), 3u);
    EXPECT_TRUE(graph.pose(300)->world_t_camera.isApprox(2.0 * Eigen::Vector3d::UnitX()));
}
