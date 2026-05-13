#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <random>
#include <vector>

#include "frond_end/feature_manager.h"
#include "linearization/landmark_block.h"
#include "linearization/linearization_abs_qr.h"
#include "lm_optimizer/lm_optimizer.h"
#include "loss_fuction/loss_fuction_base.h"
#include "state/state.h"

namespace tassel_core {
namespace {

constexpr int NUM_FRAMES = 7;
constexpr int NUM_POINTS = 100;
constexpr int MIN_OBS_PER_POINT = 3;  // monoTriangulate needs > 2 obs

// ── helpers ──────────────────────────────────────────────────────────────────

/// zigzag trajectory with enough lateral motion for well-conditioned triangulation
std::vector<Sophus::SE3d> makeGroundTruthPoses(int n) {
    std::vector<Sophus::SE3d> poses(n);
    for (int i = 0; i < n; ++i) {
        double x = i * 0.3;                      // forward
        double y = (i % 2 == 0) ? 0.25 : -0.25;  // significant lateral zigzag
        double z = (i % 3 == 0) ? 0.05 : -0.03;  // small vertical variation
        Eigen::Vector3d t(x, y, z);
        double yaw = (i - 1) * 0.15;  // gentle yaw oscillation
        Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()).toRotationMatrix();
        poses[i] = Sophus::SE3d(R, t);
    }
    return poses;
}

/// random 3D points in a frustum in front of the reference camera
std::vector<Eigen::Vector3d> makeWorldPoints(
    int n, const Sophus::SE3d& T_w_ref, std::mt19937& rng) {
    std::uniform_real_distribution<double> du(-0.5, 0.5);
    std::uniform_real_distribution<double> dv(-0.4, 0.4);
    std::uniform_real_distribution<double> dz(0.5, 3.0);

    std::vector<Eigen::Vector3d> pts(n);
    for (int i = 0; i < n; ++i) {
        // random normalized image coordinate + depth in reference frame
        Eigen::Vector3d uv(du(rng), dv(rng), 1.0);
        double depth = dz(rng);
        Eigen::Vector3d P_ref = uv * depth;
        pts[i] = T_w_ref * P_ref;
    }
    return pts;
}

/// project world point to normalized image coordinate (u, v, 1) at given pose
Eigen::Vector3d project(const Eigen::Vector3d& P_w, const Sophus::SE3d& T_wc) {
    Eigen::Vector3d P_c = T_wc.inverse() * P_w;
    return Eigen::Vector3d(P_c.x() / P_c.z(), P_c.y() / P_c.z(), 1.0);
}

/// 6-DOF SE3 distance (log-map norm)
double poseDistance(const Sophus::SE3d& T_a, const Sophus::SE3d& T_b) {
    return (T_a.inverse() * T_b).log().norm();
}

/// average pose distance over all frames
double meanPoseDist(const std::shared_ptr<State>& state, const std::vector<Sophus::SE3d>& gt) {
    double sum = 0.0;
    for (int i = 0; i < NUM_FRAMES; ++i) sum += poseDistance(state->poses[i].get_pose(), gt[i]);
    return sum / NUM_FRAMES;
}

/// perturb a pose with multiplicative SE3 noise
Sophus::SE3d perturbPose(
    const Sophus::SE3d& T, double trans_sigma, double rot_sigma, std::mt19937& rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() << trans_sigma * n(rng), trans_sigma * n(rng), trans_sigma * n(rng);
    delta.tail<3>() << rot_sigma * n(rng), rot_sigma * n(rng), rot_sigma * n(rng);
    return T * Sophus::SE3d::exp(delta);
}

// ── fixtures ────────────────────────────────────────────────────────────────

class LMOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);

        // ground truth
        gt_poses_ = makeGroundTruthPoses(NUM_FRAMES);
        state_ = std::make_shared<State>(NUM_FRAMES);
        state_->cur_frame_count = NUM_FRAMES;

        // initialise state with ground-truth poses (will be noised later)
        for (int i = 0; i < NUM_FRAMES; ++i) {
            state_->poses[i] = PoseStateWithLin(gt_poses_[i]);
            state_->poses[i].set_optimized_pose(gt_poses_[i]);
        }

        // generate world points visible in the first frame
        world_pts_ = makeWorldPoints(NUM_POINTS, gt_poses_[0], rng_);

        // determine which frames see each point (up to 4 consecutive frames)
        point_visibility_.resize(NUM_POINTS);
        for (int p = 0; p < NUM_POINTS; ++p) {
            int first = std::uniform_int_distribution<int>(0, 3)(rng_);
            int last = std::min(NUM_FRAMES - 1, first + 3);
            for (int f = first; f <= last; ++f) {
                // only if in front of camera
                Eigen::Vector3d P_c = gt_poses_[f].inverse() * world_pts_[p];
                if (P_c.z() > 0.15) point_visibility_[p].push_back(f);
            }
            // ensure at least MIN_OBS_PER_POINT frames visible
            if (point_visibility_[p].size() < MIN_OBS_PER_POINT) {
                point_visibility_[p].clear();
                point_visibility_[p] = {0, 1, 2};
            }
        }

        // build FeatureManager with observations
        fm_ = std::make_shared<FeatureManager>(
            0.99,      // reprojection_error_thres (loose)
            0.001,     // parallax_thres (always trigger keyframe)
            2,         // tracked_times_thres (min 2 obs per feature)
            10, 5,     // PnP: max_needed, min_pt
            0.3,       // min_pnp_inliers_ratio
            0.05,      // min_translation
            0.1, 10.0  // min_depth, max_depth
        );

        // feed observations frame-by-frame through checkKeyFrameByParallax
        for (int f = 0; f < NUM_FRAMES; ++f) {
            std::unordered_map<int, FeaturePerFrame> frame_obs;
            for (int p = 0; p < NUM_POINTS; ++p) {
                auto it = std::find(point_visibility_[p].begin(), point_visibility_[p].end(), f);
                if (it == point_visibility_[p].end()) continue;

                Eigen::Vector3d uv = project(world_pts_[p], gt_poses_[f]);
                FeaturePerFrame fpf;
                fpf.uv = uv;
                fpf.pt = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
                frame_obs[p] = fpf;
            }
            fm_->checkKeyFrameByParallax(f, frame_obs);
        }

        // triangulate to set initial depths (uses get_optimized_pose)
        Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d z3 = Eigen::Vector3d::Zero();
        fm_->triangulate(*state_, I3, z3);

        // DLT fails on noise-free synthetic data (perfect epipolar → rank-3 matrix,
        // condition number ~1e16). Set depths analytically from ground truth.
        for (auto& [id, f] : fm_->testFeatures()) {
            Eigen::Vector3d P_ref = gt_poses_[f.start_frame_id].inverse() * world_pts_[id];
            f.estimated_depth = P_ref.z();
        }
    }

    std::mt19937 rng_;
    std::vector<Sophus::SE3d> gt_poses_;
    std::vector<Eigen::Vector3d> world_pts_;
    std::vector<std::vector<int>> point_visibility_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> fm_;
};

// ── helper: configure and run the full pipeline ─────────────────────────────

struct OptimizeResult {
    double initial_error;
    double final_error;
    double initial_dist;  // mean pose distance to GT before optimisation
    double final_dist;    // mean pose distance to GT after optimisation
    int iterations;
};

OptimizeResult runOptimize(
    std::shared_ptr<State> state, std::shared_ptr<FeatureManager> fm,
    const std::vector<Sophus::SE3d>& gt_poses, double pose_noise_trans, double pose_noise_rot) {
    // noise state poses (pose_linearized and T_wc_current)
    std::mt19937 rng(123);
    for (int i = 0; i < NUM_FRAMES; ++i) {
        auto T_noisy = perturbPose(gt_poses[i], pose_noise_trans, pose_noise_rot, rng);
        state->poses[i] = PoseStateWithLin(T_noisy);
    }

    // noise feature depths (~ ±20 %)
    {
        std::normal_distribution<double> nd(0.0, 0.15);
        auto feats = fm->collectOptimizationFeatures();
        for (auto* f : feats) {
            double true_depth = (gt_poses[f->start_frame_id].inverse() *
                                 /* compute GT depth from host obs */
                                 Eigen::Vector3d::Zero())
                                    .z();
            f->estimated_depth *= std::exp(nd(rng));
            f->estimated_depth = std::max(0.1, std::min(10.0, f->estimated_depth));
        }
    }

    // setup optimisation
    LinearizationAbsQR linearization(1, state, fm, TrivialLoss{}, DepthLoss::none(), nullptr);

    LMOptions opts;
    opts.max_iterations = 10;
    opts.lambda_initial = 1e-3;
    LMOptimizer optimizer(opts);

    // initial metrics
    OptimizeResult res;
    res.initial_dist = meanPoseDist(state, gt_poses);
    res.initial_error = linearization.computeError();
    // linearization.linearizeProbelm();  // also valid

    // run
    optimizer.optimize(&linearization);

    // final metrics (need to linearize again to get a clean error)
    res.final_error = linearization.computeError();
    res.final_dist = meanPoseDist(state, gt_poses);

    return res;
}

// ── tests ───────────────────────────────────────────────────────────────────

TEST_F(LMOptimizerTest, ErrorDecreases) {
    auto res = runOptimize(state_, fm_, gt_poses_, 0.05, 0.05);
    EXPECT_LT(res.final_error, res.initial_error)
        << "initial error " << res.initial_error << " final error " << res.final_error;
}

TEST_F(LMOptimizerTest, PosesConvergeTowardsGT) {
    auto res = runOptimize(state_, fm_, gt_poses_, 0.05, 0.05);
    EXPECT_LT(res.final_dist, res.initial_dist)
        << "initial dist " << res.initial_dist << " final dist " << res.final_dist;
    // should get substantially closer (at least 30 % improvement)
    EXPECT_LT(res.final_dist, res.initial_dist * 0.7);
}

TEST_F(LMOptimizerTest, SmallNoiseNearPerfect) {
    // very small noise → optimizer should bring it very close to GT
    auto res = runOptimize(state_, fm_, gt_poses_, 0.02, 0.01);
    EXPECT_LT(res.final_dist, res.initial_dist * 0.3);
    EXPECT_LT(res.final_dist, 0.05);  // < 5 cm average
}

}  // namespace
}  // namespace tassel_core
