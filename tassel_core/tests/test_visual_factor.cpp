#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <random>
#include <vector>

#include <sophus/se3.hpp>

#include <ceres/ceres.h>
#include <ceres/jet.h>

#include "factor/se3_right_manifold.h"
#include "factor/visual_factor.h"

namespace tassel_core {
namespace {

constexpr double kJacobianRelTol = 1e-6;

void se3ToArray(const Sophus::SE3d& T, double* x) {
    Eigen::Vector3d phi = T.so3().log();
    Eigen::Vector3d P = T.translation();
    Eigen::Map<Eigen::Vector3d>{x} = phi;
    Eigen::Map<Eigen::Vector3d>{x + 3} = P;
}

// ── Templated SO(3) exp for Ceres autodiff ───────────────────────────────────

template <typename T>
Eigen::Matrix<T, 3, 3> so3_exp(const Eigen::Matrix<T, 3, 1>& omega) {
    T theta2 = omega.squaredNorm();
    T theta = sqrt(theta2);

    Eigen::Matrix<T, 3, 3> K;
    K << T(0), -omega(2), omega(1), omega(2), T(0), -omega(0), -omega(1), omega(0), T(0);

    T A, B;
    if (theta < T(1e-10)) {
        A = T(1.0) - theta2 / T(6.0);
        B = T(0.5) - theta2 / T(24.0);
    } else {
        T inv_theta = T(1.0) / theta;
        A = sin(theta) * inv_theta;
        B = (T(1.0) - cos(theta)) * inv_theta * inv_theta;
    }
    return Eigen::Matrix<T, 3, 3>::Identity() + A * K + B * K * K;
}

// ── Autodiff-compatible residual functor (same math as VisualFactor) ─────────

struct VisualResidual {
    VisualResidual(
        const Eigen::Vector3d& uv_i, const Eigen::Vector3d& uv_j, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic)
        : uv_i_(uv_i), uv_j_(uv_j), ric_(ric), tic_(tic) {
        Eigen::Vector3d a = uv_j.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp) tmp << 1, 0, 0;
        Eigen::Vector3d b1 = (tmp - a * (a.transpose() * tmp)).normalized();
        Eigen::Vector3d b2 = a.cross(b1);
        tangent_base_.row(0) = b1.transpose();
        tangent_base_.row(1) = b2.transpose();
    }

    template <typename T>
    bool operator()(
        const T* host_pose, const T* target_pose, const T* inv_depth_ptr, T* residuals) const {
        using Vec3 = Eigen::Matrix<T, 3, 1>;
        using Mat3 = Eigen::Matrix<T, 3, 3>;

        Vec3 phi_h(host_pose[0], host_pose[1], host_pose[2]);
        Vec3 P_h(host_pose[3], host_pose[4], host_pose[5]);
        Vec3 phi_t(target_pose[0], target_pose[1], target_pose[2]);
        Vec3 P_t(target_pose[3], target_pose[4], target_pose[5]);
        T inv_depth = inv_depth_ptr[0];
        T depth = T(1.0) / inv_depth;

        Mat3 R_h = so3_exp(phi_h);
        Mat3 R_t = so3_exp(phi_t);

        Vec3 uv_i_t = uv_i_.cast<T>();
        Vec3 uv_j_t = uv_j_.cast<T>();
        Mat3 ric_t = ric_.cast<T>();
        Vec3 tic_t = tic_.cast<T>();

        Vec3 pi_in_H = uv_i_t * depth;
        Vec3 pi_in_I = ric_t * pi_in_H + tic_t;
        Vec3 pi_in_W = R_h * pi_in_I + P_h;
        Vec3 pj_in_I = R_t.transpose() * (pi_in_W - P_t);
        Vec3 pj_in_C = ric_t.transpose() * (pj_in_I - tic_t);

        T norm = pj_in_C.norm();
        Vec3 reproj = pj_in_C / norm - uv_j_t;

        Eigen::Matrix<T, 2, 3> tb;
        tb.row(0) = tangent_base_.row(0).cast<T>();
        tb.row(1) = tangent_base_.row(1).cast<T>();

        Eigen::Matrix<T, 2, 1> res = tb * reproj;
        residuals[0] = res(0);
        residuals[1] = res(1);
        return true;
    }

    Eigen::Vector3d uv_i_, uv_j_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Matrix<double, 2, 3> tangent_base_;
};

// ── Fixture ──────────────────────────────────────────────────────────────────

class VisualFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d::Zero();

        // Non-trivial poses to exercise manifold Jacobian conversion
        T_w_h_ = Sophus::SE3d(
            Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()).toRotationMatrix(),
            Eigen::Vector3d(0.1, -0.05, 0.0));
        T_w_t_ = Sophus::SE3d(
            Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitY()).toRotationMatrix(),
            Eigen::Vector3d(0.5, 0.03, -0.02));

        depth_gt_ = 2.5;
        uv_i_ = Eigen::Vector3d(0.15, -0.08, 1.0);
        Eigen::Vector3d P_h = uv_i_ * depth_gt_;
        Eigen::Vector3d P_w = T_w_h_ * P_h;

        Eigen::Vector3d P_t = T_w_t_.inverse() * P_w;
        uv_j_ = Eigen::Vector3d(P_t.x() / P_t.z(), P_t.y() / P_t.z(), 1.0);

        inv_depth_gt_ = 1.0 / depth_gt_;
        se3ToArray(T_w_h_, host_pose_);
        se3ToArray(T_w_t_, target_pose_);
        inv_depth_arr_[0] = inv_depth_gt_;
    }

    Sophus::SE3d T_w_h_, T_w_t_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d uv_i_, uv_j_;
    double depth_gt_, inv_depth_gt_;
    double host_pose_[6];
    double target_pose_[6];
    double inv_depth_arr_[1];
};

// ── Test 1: residual match between hand-coded and autodiff ─────────────────────

TEST_F(VisualFactorTest, ResidualMatchesAutoDiff) {
    VisualFactor hand(uv_i_, uv_j_, ric_, tic_, depth_gt_);
    ceres::AutoDiffCostFunction<VisualResidual, 2, 6, 6, 1> autodiff(
        new VisualResidual(uv_i_, uv_j_, ric_, tic_));

    double const* params[] = {host_pose_, target_pose_, inv_depth_arr_};

    double r_hand[2], r_auto[2];
    hand.Evaluate(params, r_hand, nullptr);
    autodiff.Evaluate(params, r_auto, nullptr);

    EXPECT_NEAR(r_hand[0], r_auto[0], 1e-12);
    EXPECT_NEAR(r_hand[1], r_auto[1], 1e-12);
}

// ── Test 2: Jacobian match — hand-coded vs autodiff (with manifold conversion)

TEST_F(VisualFactorTest, JacobiansMatchAutoDiff) {
    VisualFactor hand(uv_i_, uv_j_, ric_, tic_, depth_gt_);
    ceres::AutoDiffCostFunction<VisualResidual, 2, 6, 6, 1> autodiff(
        new VisualResidual(uv_i_, uv_j_, ric_, tic_));

    double const* params[] = {host_pose_, target_pose_, inv_depth_arr_};

    // --- hand-coded Jacobians ---
    double r_hand[2];
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_host_hand, J_target_hand;
    Eigen::Matrix<double, 2, 1> J_depth_hand;
    {
        double* jacs[3] = {J_host_hand.data(), J_target_hand.data(), J_depth_hand.data()};
        hand.Evaluate(params, r_hand, jacs);
    }

    // --- autodiff Jacobians (ambient space) ---
    double r_auto[2];
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_host_auto, J_target_auto;
    Eigen::Matrix<double, 2, 1> J_depth_auto;
    {
        double* jacs[3] = {J_host_auto.data(), J_target_auto.data(), J_depth_auto.data()};
        autodiff.Evaluate(params, r_auto, jacs);
    }

    // --- manifold PlusJacobians ---
    SE3RightManifold manifold;
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Jplus_h, Jplus_t;
    manifold.PlusJacobian(host_pose_, Jplus_h.data());
    manifold.PlusJacobian(target_pose_, Jplus_t.data());

    // Convert autodiff to tangent space: J_tangent = J_ambient * PlusJacobian
    Eigen::Matrix<double, 2, 6> J_host_tangent = J_host_auto * Jplus_h;
    Eigen::Matrix<double, 2, 6> J_target_tangent = J_target_auto * Jplus_t;
    // inv_depth has no manifold → ambient = tangent
    Eigen::Matrix<double, 2, 1> J_depth_tangent = J_depth_auto;

    // Hand-coded VisualFactor returns tangent-space Jacobians (right-mult on SE3)
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 6; ++c) {
            double denom = std::max(1e-6, std::abs(J_host_tangent(r, c)));
            EXPECT_NEAR(J_host_hand(r, c), J_host_tangent(r, c), kJacobianRelTol * denom)
                << "Host pose Jacobian mismatch at (" << r << "," << c << ")";
        }
        for (int c = 0; c < 6; ++c) {
            double denom = std::max(1e-6, std::abs(J_target_tangent(r, c)));
            EXPECT_NEAR(J_target_hand(r, c), J_target_tangent(r, c), kJacobianRelTol * denom)
                << "Target pose Jacobian mismatch at (" << r << "," << c << ")";
        }
    }
    for (int r = 0; r < 2; ++r) {
        double denom = std::max(1e-6, std::abs(J_depth_tangent(r, 0)));
        EXPECT_NEAR(J_depth_hand(r, 0), J_depth_tangent(r, 0), kJacobianRelTol * denom)
            << "Depth Jacobian mismatch at row " << r;
    }
}

// ── Test 3: Ceres optimization with VisualFactor + manifold converges ────────

class VisualFactorOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);

        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d::Zero();

        T_w_h_gt_ = Sophus::SE3d(
            Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
            Eigen::Vector3d(0.1, 0.02, 0.0));

        depth_gt_ = 2.0;
        uv_i_ = Eigen::Vector3d(0.1, 0.05, 1.0);
        Eigen::Vector3d P_h = uv_i_ * depth_gt_;
        Eigen::Vector3d P_w = T_w_h_gt_ * P_h;

        kNumTargets_ = 8;  // 8 factors × 2 = 16 constraints for 7 DOF → well overdetermined
        target_poses_gt_.resize(kNumTargets_);
        uv_js_.resize(kNumTargets_);
        target_poses_arr_.resize(kNumTargets_);

        for (int i = 0; i < kNumTargets_; ++i) {
            double x = 0.15 + i * 0.12;
            double y = (i % 2 == 0) ? 0.06 : -0.04;
            double yaw = (i % 3) * 0.04 - 0.02;
            target_poses_gt_[i] = Sophus::SE3d(
                Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()).toRotationMatrix(),
                Eigen::Vector3d(x, y, 0.0));

            Eigen::Vector3d P_t = target_poses_gt_[i].inverse() * P_w;
            uv_js_[i] = Eigen::Vector3d(P_t.x() / P_t.z(), P_t.y() / P_t.z(), 1.0);

            target_poses_arr_[i].resize(6);
            se3ToArray(target_poses_gt_[i], target_poses_arr_[i].data());
        }

        inv_depth_gt_ = 1.0 / depth_gt_;
        se3ToArray(T_w_h_gt_, host_pose_gt_);
        inv_depth_arr_gt_[0] = inv_depth_gt_;
    }

    int kNumTargets_;
    std::mt19937 rng_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Sophus::SE3d T_w_h_gt_;
    std::vector<Sophus::SE3d> target_poses_gt_;
    Eigen::Vector3d uv_i_;
    std::vector<Eigen::Vector3d> uv_js_;
    double depth_gt_, inv_depth_gt_;
    double host_pose_gt_[6];
    double inv_depth_arr_gt_[1];
    std::vector<std::vector<double>> target_poses_arr_;
};

TEST_F(VisualFactorOptimizationTest, CeresConvergesToGroundTruth) {
    std::mt19937 rng(42);
    std::normal_distribution<double> n(0.0, 1.0);

    // Fix host pose (GT), perturb only inverse depth → 1 DOF, 16 constraints
    std::vector<double> inv_depth(1);
    inv_depth[0] = inv_depth_arr_gt_[0] + 0.05 * n(rng);

    const double inv_depth_init = inv_depth[0];

    ceres::Problem problem;

    // Ceres takes ownership of manifolds — must be heap-allocated, one per block
    std::vector<double> host_pose(host_pose_gt_, host_pose_gt_ + 6);
    problem.AddParameterBlock(host_pose.data(), 6, new SE3RightManifold());
    problem.SetParameterBlockConstant(host_pose.data());

    problem.AddParameterBlock(inv_depth.data(), 1);

    for (int i = 0; i < kNumTargets_; ++i) {
        problem.AddParameterBlock(target_poses_arr_[i].data(), 6, new SE3RightManifold());
        problem.SetParameterBlockConstant(target_poses_arr_[i].data());

        auto* cost = new VisualFactor(uv_i_, uv_js_[i], ric_, tic_, depth_gt_);
        problem.AddResidualBlock(
            cost, nullptr, host_pose.data(), target_poses_arr_[i].data(), inv_depth.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    EXPECT_EQ(summary.termination_type, ceres::CONVERGENCE);
    EXPECT_LT(summary.final_cost, 1e-8) << "Final cost=" << summary.final_cost;

    EXPECT_NEAR(inv_depth[0], inv_depth_arr_gt_[0], 1e-4)
        << "Inverse depth  initial=" << inv_depth_init << " optimized=" << inv_depth[0]
        << " gt=" << inv_depth_arr_gt_[0];
}

}  // namespace
}  // namespace tassel_core
