// =============================================================================
// test_two_view_factor.cpp — TwoViewFactor Jacobian verification + td convergence
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <vector>

#include <ceres/ceres.h>
#include <sophus/so3.hpp>

#include "factor/TwoViewFactor.h"
#include "factor/se3_right_manifold.h"

namespace tassel_core {
namespace {

static constexpr double kG = 9.81;
static const Eigen::Vector3d kGravity(0, 0, -kG);

// ----- IMU sample --------------------------------------------------------------

struct ImuSample {
    double ts;
    Eigen::Vector3d P, V;
    Eigen::Matrix3d R;
    Eigen::Vector3d gyro, acc;  // biased measurements
};

// ----- mid-point trajectory generation (250 Hz) ---------------------------------

std::vector<ImuSample> generateTrajectory(
    double duration, double imu_dt, const Eigen::Vector3d& a_body, const Eigen::Vector3d& w_body,
    const Eigen::Vector3d& ba, const Eigen::Vector3d& bg) {
    std::vector<ImuSample> traj;
    int N = static_cast<int>(duration / imu_dt) + 1;

    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d V = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    for (int k = 0; k < N; ++k) {
        ImuSample s;
        s.ts = k * imu_dt;
        s.R = R;
        s.P = P;
        s.V = V;
        s.gyro = w_body + bg;
        s.acc = a_body + R.transpose() * (-kGravity) + ba;
        traj.push_back(s);

        double dt2 = imu_dt * 0.5;
        Eigen::Matrix3d Rmid = R * Sophus::SO3d::exp(w_body * dt2).matrix();
        Eigen::Vector3d V_next = V + Rmid * a_body * imu_dt;
        Eigen::Vector3d P_next = P + (V + V_next) * dt2;
        R = R * Sophus::SO3d::exp(w_body * imu_dt).matrix();
        P = P_next;
        V = V_next;
    }
    return traj;
}

// ----- linear interpolation ----------------------------------------------------

ImuSample interpolateSample(const std::vector<ImuSample>& traj, double t) {
    double imu_dt = traj[1].ts - traj[0].ts;
    int i0 = static_cast<int>(std::floor(t / imu_dt));
    int i1 = i0 + 1;
    if (i0 < 0) {
        i0 = 0;
        i1 = 0;
    }
    if (i1 >= static_cast<int>(traj.size())) {
        i0 = traj.size() - 1;
        i1 = i0;
    }

    const auto& s0 = traj[i0];
    const auto& s1 = traj[i1];
    double alpha = (i1 > i0) ? (t - s0.ts) / (s1.ts - s0.ts) : 0.0;
    double a = std::max(0.0, std::min(1.0, alpha));

    ImuSample s;
    s.ts = t;
    s.P = (1 - a) * s0.P + a * s1.P;
    s.V = (1 - a) * s0.V + a * s1.V;
    s.gyro = (1 - a) * s0.gyro + a * s1.gyro;
    s.acc = (1 - a) * s0.acc + a * s1.acc;
    Eigen::Quaterniond q0(s0.R), q1(s1.R);
    s.R = Eigen::Quaterniond(q0.slerp(a, q1)).toRotationMatrix();
    return s;
}

// ----- mid-point integration: parameter time → camera time ----------------------

struct CameraState {
    Eigen::Matrix3d R;
    Eigen::Vector3d P;
};

CameraState integrateImu(
    const std::vector<ImuSample>& traj, double t_param, double t_cam, const Eigen::Vector3d& bg,
    const Eigen::Vector3d& ba) {
    double imu_dt = traj[1].ts - traj[0].ts;
    ImuSample s0 = interpolateSample(traj, t_param);
    Eigen::Vector3d P = s0.P, V = s0.V;
    Eigen::Matrix3d R = s0.R;
    double t = t_param;

    while (t < t_cam - 1e-9) {
        double next_imu = std::ceil((t + 1e-9) / imu_dt) * imu_dt;
        double t_next = std::min(next_imu, t_cam);
        double dt = t_next - t;
        if (dt < 1e-9) break;

        ImuSample prev = interpolateSample(traj, t);
        ImuSample next = interpolateSample(traj, t_next);

        Eigen::Vector3d gyro = 0.5 * (prev.gyro + next.gyro) - bg;
        Eigen::Vector3d prev_acc = R * (prev.acc - ba);
        Eigen::Vector3d next_acc = R * Sophus::SO3d::exp(gyro * dt).matrix() * (next.acc - ba);
        Eigen::Vector3d acc_mid = 0.5 * (prev_acc + next_acc);

        P = P + V * dt + 0.5 * acc_mid * dt * dt;
        V = V + acc_mid * dt;
        R = R * Sophus::SO3d::exp(gyro * dt).matrix();
        t = t_next;
    }
    return {R, P};
}

// ----- single-frame setup helper ------------------------------------------------

struct FrameData {
    Eigen::Vector3d P, V, w, a;
    Eigen::Matrix3d R;
    Eigen::Vector3d uv_cam;
    double pose[6];
    double V_arr[3];
};

FrameData setupFrame(
    const std::vector<ImuSample>& traj, double t_cam, double td, const Eigen::Vector3d& Bg,
    const Eigen::Vector3d& Ba, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
    const Eigen::Vector3d& P_w) {
    double t_imu = t_cam - td;
    CameraState imu_cam = integrateImu(traj, t_imu, t_cam, Bg, Ba);

    CameraState cs;
    cs.R = imu_cam.R * ric;
    cs.P = imu_cam.P + imu_cam.R * tic;

    ImuSample param = interpolateSample(traj, t_imu);

    FrameData fd;
    fd.P = param.P;
    fd.V = param.V;
    fd.R = param.R;
    fd.w = param.gyro;
    fd.a = param.acc;
    fd.uv_cam = (cs.R.transpose() * (P_w - cs.P)).normalized();

    Eigen::Vector3d phi = Sophus::SO3d(fd.R).log();
    for (int d = 0; d < 3; ++d) {
        fd.pose[d] = fd.P[d];
        fd.pose[d + 3] = phi[d];
    }
    fd.V_arr[0] = fd.V.x();
    fd.V_arr[1] = fd.V.y();
    fd.V_arr[2] = fd.V.z();
    return fd;
}

// ==============================================================================
// JacobianCheck — single factor, relaxed tolerance
// ==============================================================================

class TwoViewJacobianTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::AngleAxisd(0.0087, Eigen::Vector3d::UnitZ()).matrix();
        tic_ << 0.05, -0.02, 0.03;
        Ba_ << 0.08, -0.04, 0.03;
        Bg_ << 0.012, -0.006, 0.004;
        td_ = 0.005;

        Eigen::Vector3d a_body(0.5, -0.3, 2.0);
        Eigen::Vector3d w_body(0.8, 1.2, 0.6);
        traj_ = generateTrajectory(2.0, 0.004, a_body, w_body, Ba_, Bg_);

        // 3D point
        Eigen::Vector3d P_w(0.5, -0.1, 20.0);
        P_w_ = P_w;

        fd_i_ = setupFrame(traj_, 0.24, td_, Bg_, Ba_, ric_, tic_, P_w_);
        fd_j_ = setupFrame(traj_, 0.60, td_, Bg_, Ba_, ric_, tic_, P_w_);

        bg_arr_[0] = Bg_.x();
        bg_arr_[1] = Bg_.y();
        bg_arr_[2] = Bg_.z();
        ba_arr_[0] = Ba_.x();
        ba_arr_[1] = Ba_.y();
        ba_arr_[2] = Ba_.z();
    }

    std::vector<ImuSample> traj_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d Ba_, Bg_;
    double td_;
    Eigen::Vector3d P_w_;
    FrameData fd_i_, fd_j_;
    double bg_arr_[3], ba_arr_[3];
};

TEST_F(TwoViewJacobianTest, RelaxedTolerance) {
    TwoViewFactor factor(
        fd_i_.uv_cam, fd_j_.uv_cam, ric_, tic_, fd_i_.w, fd_j_.w, fd_i_.a, fd_j_.a, fd_i_.V_arr,
        fd_j_.V_arr, bg_arr_, bg_arr_, ba_arr_, ba_arr_, 1.0);

    SE3RightManifold manifold;
    const double eps = 1e-6;

    const double* params[] = {fd_i_.pose, fd_j_.pose, &td_};
    double r, J0[6], J1[6], J2[1];
    double* jac_ptrs[] = {J0, J1, J2};
    factor.Evaluate(params, &r, jac_ptrs);

    std::cout << "residual = " << r << "\n";

    // --- tolerance: relative < 0.02, floor 1e-4 for small values ---
    const double rel_tol = 0.03;
    const double floor = 1e-4;
    int nbad = 0;

    auto num_pose = [&](int blk, const double* x, int col) -> double {
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(6);
        delta(col) = eps;
        double xp[6], xm[6];
        manifold.Plus(x, delta.data(), xp);
        manifold.Plus(x, (-delta).eval().data(), xm);
        double rp, rm;
        const double* pp[3] = {params[0], params[1], params[2]};
        const double* pm[3] = {params[0], params[1], params[2]};
        pp[blk] = xp;
        pm[blk] = xm;
        factor.Evaluate(pp, &rp, nullptr);
        factor.Evaluate(pm, &rm, nullptr);
        return (rp - rm) / (2.0 * eps);
    };

    auto check = [&](const char* name, const double* J_an, int dim,
                     const std::function<double(int)>& num_fn) {
        std::cout << "\n--- " << name << " (" << dim << " cols) ---\n";
        for (int c = 0; c < dim; ++c) {
            double num = num_fn(c);
            double an = J_an[c];
            double denom = std::max({std::abs(an), std::abs(num), floor});
            double err = std::abs(an - num) / denom;
            if (err > rel_tol) nbad++;
            std::cout << "  col" << c << ": an=" << an << " num=" << num << " err=" << err
                      << (err > rel_tol ? " ***" : "") << "\n";
        }
    };

    check("pose_i", J0, 6, [&](int c) { return num_pose(0, params[0], c); });
    check("pose_j", J1, 6, [&](int c) { return num_pose(1, params[1], c); });

    // td
    {
        double td_p = td_ + eps, td_m = td_ - eps;
        const double* pp[3] = {params[0], params[1], &td_p};
        const double* pm[3] = {params[0], params[1], &td_m};
        double rp, rm;
        factor.Evaluate(pp, &rp, nullptr);
        factor.Evaluate(pm, &rm, nullptr);
        double num = (rp - rm) / (2.0 * eps);
        double an = J2[0];
        double denom = std::max({std::abs(an), std::abs(num), floor});
        double err = std::abs(an - num) / denom;
        if (err > rel_tol) nbad++;
        std::cout << "\n--- td (1 col) ---\n";
        std::cout << "  an=" << an << " num=" << num << " err=" << err
                  << (err > rel_tol ? " ***" : "") << "\n";
    }

    std::cout << "\nbad cols (rel>" << rel_tol << " floor=" << floor << "): " << nbad << "\n";
    EXPECT_EQ(nbad, 0);
}

// ==============================================================================
// MultiFrameTdConvergence — 10 frames, per-frame td jitter, single estimate
// ==============================================================================

TEST(TwoViewFactorTest, MultiFrameTdConvergence) {
    Eigen::Matrix3d ric = Eigen::AngleAxisd(0.0087, Eigen::Vector3d::UnitZ()).matrix();
    Eigen::Vector3d tic(0.05, -0.02, 0.03);
    Eigen::Vector3d Ba(0.08, -0.04, 0.03), Bg(0.012, -0.006, 0.004);
    double bg[3] = {Bg.x(), Bg.y(), Bg.z()}, ba[3] = {Ba.x(), Ba.y(), Ba.z()};

    auto traj = generateTrajectory(2.0, 0.004, {0.5, -0.3, 2.0}, {0.8, 1.2, 0.6}, Ba, Bg);
    Eigen::Vector3d P_w(0.5, -0.1, 20.0);

    const int N = 10;
    double td_gt = 0.005;

    std::vector<FrameData> frm;
    for (int k = 0; k < N; ++k)
        frm.push_back(setupFrame(traj, 0.24 + k * 0.0667, td_gt, Bg, Ba, ric, tic, P_w));

    ceres::Problem problem;
    double td_est = 0.0;
    for (int k = 0; k < N - 1; ++k) {
        problem.AddResidualBlock(
            new TwoViewFactor(
                frm[k].uv_cam, frm[k + 1].uv_cam, ric, tic, frm[k].w, frm[k + 1].w, frm[k].a,
                frm[k + 1].a, frm[k].V_arr, frm[k + 1].V_arr, bg, bg, ba, ba, 1.0),
            nullptr, frm[k].pose, frm[k + 1].pose, &td_est);
        problem.SetParameterBlockConstant(frm[k].pose);
    }
    problem.SetParameterBlockConstant(frm.back().pose);
    problem.SetParameterLowerBound(&td_est, 0, 0.0);
    problem.SetParameterUpperBound(&td_est, 0, 0.05);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary sum;
    ceres::Solve(opts, &problem, &sum);

    std::cout << "\n--- MultiFrame TD (10 frames) ---\n";
    std::cout << "td ground truth: " << td_gt << "\n";
    std::cout << "td init: 0  →  final: " << td_est << "  (iters: " << sum.num_successful_steps
              << ", cost: " << sum.final_cost << ")\n";

    EXPECT_NEAR(td_est, td_gt, 0.001);
}

}  // namespace
}  // namespace tassel_core
