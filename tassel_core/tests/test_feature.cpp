/*
 * tassel_core/feature.h 单元测试 — 多视图三角化和滑动窗口。
 *
 * 测试几何（俯视图）：
 *
 *      世界系 (X 右, Y 下, Z 前)
 *
 *         Cam0         Cam1         Cam2         Cam3
 *          |            |            |            |
 *          o------------o------------o------------o---→ X  (非共线：微小 Y/Z
 *          |   0.12m    |   0.13m    |   0.13m    |        偏移 + 姿态抖动)
 *         Z=0          Z≈0          Z≈0          Z≈0
 *
 *                     ↓            ↓            ↓
 *                  obs[0]       obs[1]       obs[2]       obs[3]
 *                     \            |           /
 *                      \           |          /
 *                       \          |         /
 *                        \         |        /
 *                         \        |       /
 *                          \       |      /
 *                           \      |     /
 *                            \     |    /
 *                             ● P_world = (0, 0, 1.5)
 *                               depth = 1.5 m
 *
 *  - 4 个摄像头，X 方向间距约 0.12 m，Y 呈二次分布，Z 交替 → 非共线
 *  - 每个摄像头有微小的确定性姿态抖动 (≤ 0.024 rad ≈ 1.4°)
 *  - 足够长的基线 (总长 0.38 m) → 良态三角化
 *  - 噪声模型：
 *      • 观测噪声：  N(0, σ_obs)  作用在归一化 uv 坐标上
 *      • 姿态噪声：  N(0, σ_rot)  作用在旋转轴角上，
 *                    N(0, σ_trans) 作用在平移上
 *  - 基于 SVD 的 DLT 求解器在 condition(A) > 1e6 或
 *    depth ∉ [MIN_DISTANCE=0.1, MAX_DISTANCE=3.0] 时拒绝解 → INVALID_DEPTH (-1)
 */

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <opencv2/core/types.hpp>

#include "frond_end/feature.h"
#include "state/state.h"

namespace tassel_core {
namespace {

// ── helpers ─────────────────────────────────────────────────────────────────

struct CameraPose {
    Eigen::Matrix3d R;  // rotation  camera → world
    Eigen::Vector3d t;  // position  camera in world
};

// 沿 X 方向生成 N 个非共线的相机位姿，Y 呈二次分布，Z 交替，
// 加上微小的确定性姿态抖动以保证 DLT 矩阵秩为 4。
std::vector<CameraPose> GeneratePoses(int n) {
    std::vector<CameraPose> poses;
    for (int i = 0; i < n; ++i) {
        double rx = i * 0.004;
        double ry = i * 0.006;
        Eigen::Matrix3d R = (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()))
                                .toRotationMatrix();
        poses.push_back(
            {R, Eigen::Vector3d(i * 0.12, i * i * 0.015, (i % 2 == 0) ? i * 0.003 : -i * 0.003)});
    }
    return poses;
}

// 纯旋转位姿（零基线）— 用于验证退化情况的拒绝。
std::vector<CameraPose> GeneratePureRotationPoses(int n, double angle_step_deg) {
    std::vector<CameraPose> poses;
    for (int i = 0; i < n; ++i) {
        poses.push_back(
            {Eigen::AngleAxisd(
                 i * angle_step_deg * std::numbers::pi / 180.0, Eigen::Vector3d::UnitY())
                 .toRotationMatrix(),
             Eigen::Vector3d::Zero()});
    }
    return poses;
}

// 透视投影 → 归一化图像坐标。
Eigen::Vector2d Project(const Eigen::Vector3d& P_world, const CameraPose& cam) {
    Eigen::Vector3d P_cam = cam.R.transpose() * (P_world - cam.t);
    return {P_cam.x() / P_cam.z(), P_cam.y() / P_cam.z()};
}

// 给定外参 ric, tic，将相机位姿转换为 IMU 状态 (Rs, Ps)。
void SetupState(
    const std::vector<CameraPose>& poses, State& state, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic) {
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix3d Rs = poses[i].R * ric.transpose();
        state.poses[i] = PoseStateWithLin(Sophus::SE3d(Rs, poses[i].t - Rs * tic));
    }
}

void AddNoisePoses(
    std::vector<CameraPose>& poses, double rot_sigma, double trans_sigma, std::mt19937& rng) {
    std::normal_distribution<double> r(0.0, rot_sigma);
    std::normal_distribution<double> t(0.0, trans_sigma);
    std::normal_distribution<double> a(0.0, 1.0);
    for (auto& p : poses) {
        Eigen::Vector3d axis(a(rng), a(rng), a(rng));
        axis.normalize();
        p.R = Eigen::AngleAxisd(r(rng), axis).toRotationMatrix() * p.R;
        p.t += Eigen::Vector3d(t(rng), t(rng), t(rng));
    }
}

void AddNoiseObs(std::vector<FeaturePerFrame>& obs, double sigma, std::mt19937& rng) {
    std::normal_distribution<double> n(0.0, sigma);
    for (auto& o : obs) {
        o.uv.x() += n(rng);
        o.uv.y() += n(rng);
        if (o.is_stereo) {
            o.uv_r.x() += n(rng);
            o.uv_r.y() += n(rng);
        }
    }
}

Feature MakeFeature(
    size_t start_id, const Eigen::Vector3d& P_world, const std::vector<CameraPose>& poses,
    size_t cap = 20) {
    Feature f(start_id, cap);
    for (const auto& p : poses) {
        FeaturePerFrame o;
        o.setLeft(Project(P_world, p), cv::Point2f());
        f.observations.push_back(o);
    }
    return f;
}

// =========================================================================
// monoTriangulate
// =========================================================================

class monoTriangulateTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d::Zero();
        P_world_ = Eigen::Vector3d(0.0, 0.0, 1.5);
        rng_.seed(42);

        poses_ = GeneratePoses(4);
        num_poses_ = static_cast<int>(poses_.size());
        state_ = std::make_shared<State>(num_poses_);
        SetupState(poses_, *state_, ric_, tic_);
    }

    // 低噪声三角化 → depth ≈ 1.5，误差 ≤ 5 mm
    void CheckLowNoise() {
        Feature f = MakeFeature(0, P_world_, poses_);
        AddNoiseObs(f.observations, 1e-4, rng_);
        f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
        EXPECT_NEAR(f.estimated_depth, P_world_.z(), 0.005);
    }

    int num_poses_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d P_world_;
    std::vector<CameraPose> poses_;
    std::shared_ptr<State> state_;
    std::mt19937 rng_;
};

TEST_F(monoTriangulateTest, AccurateTriangulation) { CheckLowNoise(); }

TEST_F(monoTriangulateTest, OffCenterPoint) {
    Eigen::Vector3d P(-0.3, 0.2, 1.5);
    Feature f = MakeFeature(0, P, poses_);
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_NEAR(f.estimated_depth, P.z(), 0.005);
}

TEST_F(monoTriangulateTest, MinimalFrames) {
    auto p = GeneratePoses(3);
    auto s = std::make_shared<State>(3);
    SetupState(p, *s, ric_, tic_);
    Feature f = MakeFeature(0, P_world_, p);
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.monoTriangulate(*s, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_NEAR(f.estimated_depth, P_world_.z(), 0.005);
}

TEST_F(monoTriangulateTest, TooFewFrames) {
    auto p = GeneratePoses(2);
    auto s = std::make_shared<State>(2);
    SetupState(p, *s, ric_, tic_);
    Feature f = MakeFeature(0, P_world_, p);
    f.monoTriangulate(*s, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, INVALID_DEPTH);
}

TEST_F(monoTriangulateTest, NoisyPoses) {
    // 0.1° 旋转噪声, 1 mm 平移噪声 + 观测抖动 → depth 误差 ≤ 10 cm
    AddNoisePoses(poses_, 0.001745, 0.001, rng_);
    auto s = std::make_shared<State>(num_poses_);
    SetupState(poses_, *s, ric_, tic_);
    Feature f = MakeFeature(0, P_world_, poses_);
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.monoTriangulate(*s, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_NEAR(f.estimated_depth, P_world_.z(), 0.1);
}

TEST_F(monoTriangulateTest, NoisyObservations) {
    // σ = 0.002 归一化坐标 ≈ 0.8 px (f=400) → depth 误差 ≤ 10 cm
    Feature f = MakeFeature(0, P_world_, poses_);
    AddNoiseObs(f.observations, 0.002, rng_);
    f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_NEAR(f.estimated_depth, P_world_.z(), 0.1);
}

TEST_F(monoTriangulateTest, PureRotationFails) {
    auto rp = GeneratePureRotationPoses(4, 3.0);
    auto s = std::make_shared<State>(4);
    SetupState(rp, *s, ric_, tic_);
    Feature f = MakeFeature(0, P_world_, rp);
    f.monoTriangulate(*s, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, INVALID_DEPTH);
}

TEST_F(monoTriangulateTest, PointTooCloseRejected) {
    Feature f = MakeFeature(0, {0.0, 0.0, 0.05}, poses_);
    f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, INVALID_DEPTH);
}

TEST_F(monoTriangulateTest, PointTooFarRejected) {
    Feature f = MakeFeature(0, {0.0, 0.0, 5.0}, poses_);
    f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, INVALID_DEPTH);
}

TEST_F(monoTriangulateTest, DepthAlreadySet) {
    Feature f = MakeFeature(0, P_world_, poses_);
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    ASSERT_NEAR(f.estimated_depth, P_world_.z(), 0.005);
    f.estimated_depth = 0.5;
    f.monoTriangulate(*state_, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, 0.5);
}

// =========================================================================
// stereoTriangulate
// =========================================================================

class stereoTriangulateTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d::Zero();
        // 右相机：微小旋转 + 10 cm 基线以避免退化
        ric1_ = Eigen::AngleAxisd(0.005, Eigen::Vector3d::UnitY()).toRotationMatrix();
        tic1_ = ric1_ * Eigen::Vector3d(0.1, 0.02, 0.0);
        P_left_ = Eigen::Vector3d(0.0, 0.0, 1.5);
        rng_.seed(42);
    }

    Feature MakeStereo() const {
        Feature f(1);
        Eigen::Vector2d ul(P_left_.x() / P_left_.z(), P_left_.y() / P_left_.z());
        Eigen::Vector3d Pr = ric1_.transpose() * P_left_ - ric1_.transpose() * tic1_;
        Eigen::Vector2d ur(Pr.x() / Pr.z(), Pr.y() / Pr.z());
        FeaturePerFrame o;
        o.setLeft(ul, cv::Point2f());
        o.setRight(ur, cv::Point2f());
        f.observations.push_back(o);
        return f;
    }

    Eigen::Matrix3d ric_, ric1_;
    Eigen::Vector3d tic_, tic1_;
    Eigen::Vector3d P_left_;
    std::mt19937 rng_;
};

TEST_F(stereoTriangulateTest, AccurateStereo) {
    Feature f = MakeStereo();
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.stereoTriangulate(ric_, tic_, ric1_, tic1_, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_NEAR(f.estimated_depth, P_left_.z(), 0.005);
}

TEST_F(stereoTriangulateTest, NoisyObservations) {
    Feature f = MakeStereo();
    AddNoiseObs(f.observations, 0.002, rng_);
    f.stereoTriangulate(ric_, tic_, ric1_, tic1_, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_NEAR(f.estimated_depth, P_left_.z(), 0.1);
}

TEST_F(stereoTriangulateTest, NoRightImage) {
    Feature f(1);
    FeaturePerFrame o;
    o.setLeft(Eigen::Vector2d(P_left_.x() / P_left_.z(), P_left_.y() / P_left_.z()), cv::Point2f());
    f.observations.push_back(o);
    f.stereoTriangulate(ric_, tic_, ric1_, tic1_, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, INVALID_DEPTH);
}

TEST_F(stereoTriangulateTest, DepthAlreadySet) {
    Feature f = MakeStereo();
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.stereoTriangulate(ric_, tic_, ric1_, tic1_, MIN_DISTANCE, MAX_DISTANCE);
    ASSERT_NEAR(f.estimated_depth, P_left_.z(), 0.005);
    f.estimated_depth = 0.5;
    f.stereoTriangulate(ric_, tic_, ric1_, tic1_, MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_DOUBLE_EQ(f.estimated_depth, 0.5);
}

// =========================================================================
// Slide window
// =========================================================================

class SlideWindowTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d::Zero();
        P_world_ = Eigen::Vector3d(0.0, 0.0, 1.5);
        rng_.seed(42);
    }

    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d P_world_;
    std::mt19937 rng_;
};

TEST_F(SlideWindowTest, RemoveOldestReparenting) {
    auto poses = GeneratePoses(4);
    auto state = std::make_shared<State>(4);
    SetupState(poses, *state, ric_, tic_);

    Feature f = MakeFeature(0, P_world_, poses);
    AddNoiseObs(f.observations, 1e-4, rng_);
    f.monoTriangulate(*state, ric_, tic_, 0.0, MIN_DISTANCE, MAX_DISTANCE);
    ASSERT_NEAR(f.estimated_depth, P_world_.z(), 0.005);

    f.removeOldest(
        state->poses[0].get_pose().rotationMatrix(), state->poses[0].get_pose().translation(),
        state->poses[1].get_pose().rotationMatrix(), state->poses[1].get_pose().translation(), ric_,
        tic_);
    EXPECT_EQ(f.start_frame_id, 0);
    EXPECT_EQ(f.observations.size(), 3);
    EXPECT_NEAR(f.estimated_depth, P_world_.z(), 0.005);
}

TEST_F(SlideWindowTest, RemoveOldestDecrementStartFrame) {
    auto poses = GeneratePoses(3);
    Feature f(5, 10);
    for (const auto& p : poses) {
        FeaturePerFrame o;
        o.setLeft(Project(P_world_, p), cv::Point2f());
        f.observations.push_back(o);
    }
    f.removeOldest(
        Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(),
        Eigen::Vector3d::Zero(), ric_, tic_);
    EXPECT_EQ(f.start_frame_id, 4);
    EXPECT_EQ(f.observations.size(), 2);
    EXPECT_DOUBLE_EQ(f.estimated_depth, INVALID_DEPTH);
}

TEST_F(SlideWindowTest, RemoveNewestMatch) {
    Feature f(2, 10);
    FeaturePerFrame o;
    o.setLeft(Eigen::Vector2d(0.1, 0.2), cv::Point2f());
    f.observations.push_back(o);
    f.observations.push_back(o);
    f.removeNewest(3);
    EXPECT_EQ(f.observations.size(), 1);
}

TEST_F(SlideWindowTest, RemoveNewestNoMatch) {
    Feature f(2, 10);
    FeaturePerFrame o;
    o.setLeft(Eigen::Vector2d(0.1, 0.2), cv::Point2f());
    f.observations.push_back(o);
    f.observations.push_back(o);
    f.removeNewest(5);
    EXPECT_EQ(f.observations.size(), 2);
}

}  // namespace
}  // namespace tassel_core
