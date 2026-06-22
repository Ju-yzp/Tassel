// =============================================================================
// test_visual_factor.cpp — VisualFactor Jacobian verification & convergence
// =============================================================================
//
// IMU 400Hz trajectory → camera at IMU sample time k_cam_i + td (query state)
// Observations generated from query state (exact), factor uses td compensation
// approximation to reconstruct query state from camera-time state.
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <vector>

#include <sophus/so3.hpp>

#include <ceres/ceres.h>

#include "cam/camera_rad_tan.h"
#include "factor/visual_factor.h"
#include "tassel_utils/se3_right_manifold.h"
#include "tassel_utils/types.h"

namespace tassel_core {
namespace {

struct ImuState {
    double ts;
    Eigen::Matrix3d R;
    Eigen::Vector3d P, V;
    Eigen::Vector3d gyro, acc;
    Eigen::Vector3d Ba, Bg;
};

struct StateTimeline {
    double imu_dt;
    std::vector<ImuState> states;
    const ImuState& at_index(int k) const { return states[k]; }
};

StateTimeline generateImuTimeline(
    double duration, double imu_dt, const Eigen::Vector3d& a_body, const Eigen::Vector3d& w_body,
    const Eigen::Vector3d& Ba, const Eigen::Vector3d& Bg) {
    int total_steps = static_cast<int>(duration / imu_dt) + 1;
    StateTimeline tl;
    tl.imu_dt = imu_dt;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d V = Eigen::Vector3d::Zero();

    for (int k = 0; k < total_steps; ++k) {
        ImuState s;
        s.ts = k * imu_dt;
        s.R = R;
        s.P = P;
        s.V = V;
        s.Ba = Ba;
        s.Bg = Bg;
        s.acc = a_body + R.transpose() * tassel_utils::G + Ba;
        s.gyro = w_body + Bg;
        tl.states.push_back(s);

        double dt2 = imu_dt * 0.5;
        Eigen::Matrix3d Rmid = R * Sophus::SO3d::exp(w_body * dt2).matrix();
        Eigen::Vector3d V_next = V + Rmid * a_body * imu_dt;
        Eigen::Vector3d P_next = P + (V + V_next) * dt2;
        R = R * Sophus::SO3d::exp(w_body * imu_dt).matrix();
        P = P_next;
        V = V_next;
    }
    return tl;
}

// =============================================================================
// VisualFactorTest
// =============================================================================

class VisualFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d(0.05, 0.0, 0.0);
        Ba_ = Eigen::Vector3d(0.08, -0.04, 0.03);
        Bg_ = Eigen::Vector3d(0.012, -0.006, 0.004);
        td_ = 0.005;
        sqrt_info_ = Eigen::Matrix2d::Identity() * 320.0;

        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        Eigen::VectorXd D(4);
        D << 0, 0, 0, 0;
        camera_ = CameraRadTan(K, D, 640, 480);

        double imu_dt = 0.0025;
        Eigen::Vector3d a_body(0.3, -0.1, 0.05);
        Eigen::Vector3d w_body(0.1, 0.3, 0.4);
        timeline_ = generateImuTimeline(2.0, imu_dt, a_body, w_body, Ba_, Bg_);

        int cam_skip = 27;
        int k_ci = 200, k_cj = k_ci + cam_skip;
        int td_steps = static_cast<int>(std::round(td_ / imu_dt));

        const auto& sqi = timeline_.at_index(k_ci + td_steps);
        const auto& sqj = timeline_.at_index(k_cj + td_steps);
        const auto& sci = timeline_.at_index(k_ci);
        const auto& scj = timeline_.at_index(k_cj);

        w_i_ = sci.gyro;
        a_i_ = sci.acc;
        V_i_ = sci.V;
        w_j_ = scj.gyro;
        a_j_ = scj.acc;
        V_j_ = scj.V;

        // Generate landmarks from query state (exact, no approximation)
        std::vector<Eigen::Vector3d> P_cam_i_pts = {
            {0.3, -0.2, 1.5}, {-0.5, 0.3, 3.0}, {0.2, -0.4, 2.0},
            {1.2, -0.1, 8.0}, {-1.0, 0.5, 6.0}, {0.01, 0.01, 12.0},
        };
        for (const auto& Pc : P_cam_i_pts) {
            Landmark lm;
            lm.uv_i = Pc / Pc.z();
            lm.inv_depth = 1.0 / Pc.z();

            Eigen::Vector3d P_imu_i = ric_ * Pc + tic_;
            Eigen::Vector3d P_world = sqi.R * P_imu_i + sqi.P;
            Eigen::Vector3d P_imu_j = sqj.R.transpose() * (P_world - sqj.P);
            Eigen::Vector3d P_cam_j = ric_.transpose() * (P_imu_j - tic_);
            lm.pt_j = Eigen::Vector2d(P_cam_j.x() / P_cam_j.z(), P_cam_j.y() / P_cam_j.z());
            lms_.push_back(lm);
        }

        Eigen::Vector3d phi_ci = Sophus::SO3d(sci.R).log();
        Eigen::Vector3d phi_cj = Sophus::SO3d(scj.R).log();
        for (int d = 0; d < 3; ++d) {
            pose_i_[d] = sci.P[d];
            pose_i_[d + 3] = phi_ci[d];
            pose_j_[d] = scj.P[d];
            pose_j_[d + 3] = phi_cj[d];
        }
        v_i_[0] = V_i_.x();
        v_i_[1] = V_i_.y();
        v_i_[2] = V_i_.z();
        v_j_[0] = V_j_.x();
        v_j_[1] = V_j_.y();
        v_j_[2] = V_j_.z();
        bg_[0] = Bg_.x();
        bg_[1] = Bg_.y();
        bg_[2] = Bg_.z();
        ba_[0] = Ba_.x();
        ba_[1] = Ba_.y();
        ba_[2] = Ba_.z();
    }

    VisualFactor* makeFactor(int k) const {
        const auto& lm = lms_[k];
        return new VisualFactor(
            lm.uv_i, lm.pt_j, ric_, tic_, w_i_, w_j_, a_i_, a_j_, v_i_, v_j_, bg_, bg_, ba_, ba_,
            sqrt_info_, &camera_);
    }

    // --- data ---
    StateTimeline timeline_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d Ba_, Bg_;
    double td_;
    Eigen::Matrix2d sqrt_info_;
    CameraRadTan camera_{cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), 640, 480};
    Eigen::Vector3d V_i_, V_j_, w_i_, a_i_, w_j_, a_j_;

    struct Landmark {
        Eigen::Vector3d uv_i;
        Eigen::Vector2d pt_j;
        double inv_depth;
    };
    std::vector<Landmark> lms_;

    double pose_i_[6]{}, pose_j_[6]{};
    double v_i_[3]{}, v_j_[3]{}, bg_[3]{}, ba_[3]{};
};

// =============================================================================
// 测试 1: 数值微分逐块验证雅各比
// =============================================================================

TEST_F(VisualFactorTest, JacobianCheck) {
    const auto& lm = lms_[0];
    auto* factor = makeFactor(0);
    SE3RightManifold manifold;

    double inv_depth = lm.inv_depth;
    double dt_val = td_;
    const double eps = 1e-6;
    const double tol = 5e-3;

    double J0[12], J1[12], J2[2], J3[2];
    double* jac_ptrs[] = {J0, J1, J2, J3};
    const double* params[] = {pose_i_, pose_j_, &dt_val, &inv_depth};
    double r[2];
    factor->Evaluate(params, r, jac_ptrs);

    auto num_pose = [&](int blk, const double* x, int col) {
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(6);
        delta(col) = eps;
        double xp[6], xm[6];
        manifold.Plus(x, delta.data(), xp);
        manifold.Plus(x, (-delta).eval().data(), xm);
        double rp[2], rm[2];
        if (blk == 0) {
            const double* pp[] = {xp, params[1], params[2], params[3]};
            const double* pm[] = {xm, params[1], params[2], params[3]};
            factor->Evaluate(pp, rp, nullptr);
            factor->Evaluate(pm, rm, nullptr);
        } else {
            const double* pp[] = {params[0], xp, params[2], params[3]};
            const double* pm[] = {params[0], xm, params[2], params[3]};
            factor->Evaluate(pp, rp, nullptr);
            factor->Evaluate(pm, rm, nullptr);
        }
        return Eigen::Vector2d((rp[0] - rm[0]) / (2 * eps), (rp[1] - rm[1]) / (2 * eps));
    };

    auto num_scalar = [&](int blk, double val) {
        double vp = val + eps, vm = val - eps;
        double rp[2], rm[2];
        if (blk == 2) {
            const double* pp[] = {params[0], params[1], &vp, params[3]};
            const double* pm[] = {params[0], params[1], &vm, params[3]};
            factor->Evaluate(pp, rp, nullptr);
            factor->Evaluate(pm, rm, nullptr);
        } else {
            const double* pp[] = {params[0], params[1], params[2], &vp};
            const double* pm[] = {params[0], params[1], params[2], &vm};
            factor->Evaluate(pp, rp, nullptr);
            factor->Evaluate(pm, rm, nullptr);
        }
        return Eigen::Vector2d((rp[0] - rm[0]) / (2 * eps), (rp[1] - rm[1]) / (2 * eps));
    };

    int nbad = 0;
    auto check = [&](const char* label, const double* Jan, int c, const Eigen::Vector2d& num,
                     int td) {
        double a0 = Jan[c], a1 = Jan[td + c];
        double s = std::max({std::abs(a0), std::abs(a1), std::abs(num[0]), std::abs(num[1]), 1e-8});
        double e0 = std::abs(a0 - num[0]) / s, e1 = std::abs(a1 - num[1]) / s;
        if (e0 > tol || e1 > tol) nbad++;
        std::cout << "  " << label << "[col " << c << "] an=[" << a0 << "," << a1 << "] num=["
                  << num[0] << "," << num[1] << "] err=[" << e0 << "," << e1 << "]"
                  << ((e0 > tol || e1 > tol) ? " ***" : "") << "\n";
    };

    std::cout << "\n--- pose_i (2x6) ---\n";
    for (int c = 0; c < 6; ++c) check("J_pose_i", J0, c, num_pose(0, pose_i_, c), 6);
    std::cout << "--- pose_j (2x6) ---\n";
    for (int c = 0; c < 6; ++c) check("J_pose_j", J1, c, num_pose(1, pose_j_, c), 6);
    std::cout << "--- dt (2x1) ---\n";
    check("J_dt", J2, 0, num_scalar(2, dt_val), 1);
    std::cout << "--- inv_depth (2x1) ---\n";
    check("J_inv_depth", J3, 0, num_scalar(3, inv_depth), 1);

    std::cout << "\ntotal bad (>0.5%): " << nbad << "\n";
    EXPECT_EQ(nbad, 0);

    delete factor;
}

// =============================================================================
// 测试 2: 构建 problem → td 收敛
// =============================================================================

TEST_F(VisualFactorTest, TdConvergence) {
    ceres::Problem problem;

    SE3RightManifold* manifold = new SE3RightManifold();
    problem.AddParameterBlock(pose_i_, 6, manifold);
    problem.AddParameterBlock(pose_j_, 6, manifold);
    problem.SetParameterBlockConstant(pose_i_);
    problem.SetParameterBlockConstant(pose_j_);

    double td_opt = 0.0;
    problem.AddParameterBlock(&td_opt, 1);

    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

    std::vector<double> inv_depths(lms_.size());
    for (size_t k = 0; k < lms_.size(); ++k) {
        inv_depths[k] = lms_[k].inv_depth;
        problem.AddResidualBlock(makeFactor(k), loss, pose_i_, pose_j_, &td_opt, &inv_depths[k]);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = 50;
    opts.num_threads = 1;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    std::cout << "Td optimization: " << summary.BriefReport() << "\n"
              << "  initial_cost: " << summary.initial_cost << "\n"
              << "  final_cost: " << summary.final_cost << "\n"
              << "  td: initial=0, final=" << td_opt << ", true=" << td_ << "\n";

    EXPECT_NEAR(td_opt, td_, 0.001);
    EXPECT_LT(summary.final_cost, 1e-10);
}

}  // namespace
}  // namespace tassel_core
