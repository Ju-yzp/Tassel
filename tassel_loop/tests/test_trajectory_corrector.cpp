#include "trajectory_corrector.h"

#include <gtest/gtest.h>

namespace {

tassel_loop::TimedPose pose(tassel_utils::FrameId frame_id, double x) {
    return {frame_id, Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(x, 0.0, 0.0))};
}

TEST(TrajectoryCorrectorTest, AppliesLatestKeyframeCorrectionToFollowingLocalPoses) {
    const std::vector<tassel_loop::TimedPose> local_trajectory = {
        pose(5, 0.5), pose(10, 1.0), pose(15, 1.5), pose(20, 2.0), pose(25, 2.5)};
    const std::vector<tassel_loop::TimedPose> local_keyframes = {pose(10, 1.0), pose(20, 2.0)};
    const std::vector<tassel_loop::TimedPose> global_keyframes = {pose(10, 11.0), pose(20, 22.0)};

    const auto corrected = tassel_loop::TrajectoryCorrector::correct(
        local_trajectory, local_keyframes, global_keyframes);

    ASSERT_EQ(corrected.size(), local_trajectory.size());
    EXPECT_DOUBLE_EQ(corrected[0].translation().x(), 0.5);
    EXPECT_DOUBLE_EQ(corrected[1].translation().x(), 11.0);
    EXPECT_DOUBLE_EQ(corrected[2].translation().x(), 11.5);
    EXPECT_DOUBLE_EQ(corrected[3].translation().x(), 22.0);
    EXPECT_DOUBLE_EQ(corrected[4].translation().x(), 22.5);
}

TEST(TrajectoryCorrectorTest, RejectsMismatchedOrUnorderedInputs) {
    EXPECT_THROW(
        tassel_loop::TrajectoryCorrector::correct({pose(2, 0.0), pose(1, 0.0)}, {}, {}),
        std::invalid_argument);
    EXPECT_THROW(
        tassel_loop::TrajectoryCorrector::correct({pose(1, 0.0)}, {pose(1, 0.0)}, {pose(2, 0.0)}),
        std::invalid_argument);
}

}  // namespace
