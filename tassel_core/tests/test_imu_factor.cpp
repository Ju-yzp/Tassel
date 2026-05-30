#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <sophus/so3.hpp>

#include <ceres/ceres.h>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/se3_right_manifold.h"
#include "factor/visual_factor.h"
#include "tassel_utils/constants.h"
#include "tassel_utils/types.h"

namespace tassel_core {
namespace {

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

struct ConsistentTrajectory {
    int n_steps;
    double imu_dt;
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> Ps;
    std::vector<Eigen::Vector3d> Vs;
    std::vector<tassel_utils::IMUMeasurement> imu_biased;
};

ConsistentTrajectory makeTrajectory(
    double duration, double imu_dt, const Eigen::Vector3d& a_body, const Eigen::Vector3d& w_body,
    const Eigen::Vector3d& Ba_add, const Eigen::Vector3d& Bg_add) {
    ConsistentTrajectory ct;
    ct.imu_dt = imu_dt;
    int n = static_cast<int>(duration / imu_dt) + 1;
    ct.n_steps = n;
    ct.Rs.resize(n, Eigen::Matrix3d::Identity());
    ct.Ps.resize(n, Eigen::Vector3d::Zero());
    ct.Vs.resize(n, Eigen::Vector3d::Zero());

    for (int k = 1; k < n; ++k) {
        const auto& Rp = ct.Rs[k - 1];
        const auto& Vp = ct.Vs[k - 1];
        const auto& Pp = ct.Ps[k - 1];
        double dt2 = imu_dt * 0.5;

        Eigen::Matrix3d Rmid = Rp * Sophus::SO3d::exp(w_body * dt2).matrix();
        Eigen::Vector3d a_w_mid = Rmid * a_body;
        Eigen::Vector3d Vn = Vp + a_w_mid * imu_dt;
        Eigen::Vector3d Pn = Pp + (Vp + Vn) * dt2;
        Eigen::Matrix3d Rn = Rp * Sophus::SO3d::exp(w_body * imu_dt).matrix();

        ct.Rs[k] = Rn;
        ct.Vs[k] = Vn;
        ct.Ps[k] = Pn;
    }

    for (int k = 0; k < n; ++k) {
        tassel_utils::IMUMeasurement m;
        m.timestamp = k * imu_dt;
        m.acc = a_body + ct.Rs[k].transpose() * tassel_utils::G + Ba_add;
        m.gyro = w_body + Bg_add;
        ct.imu_biased.push_back(m);
    }
    return ct;
}

// ------------------------------------------------------------
// Basic IMU factor tests
// ------------------------------------------------------------

class IMUFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        double an = 0.0193 * 0.0193, gn = 0.00264 * 0.00264;
        double aw = 0.000804 * 0.000804, gw = 0.0000703 * 0.0000703;
        noise_.block<3, 3>(0, 0) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = aw * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = gw * Eigen::Matrix3d::Identity();

        Ba_true_ = Eigen::Vector3d(0.08, -0.04, 0.03);
        Bg_true_ = Eigen::Vector3d(0.012, -0.006, 0.004);

        Eigen::Vector3d a_body(0.4, -0.15, 0.0);
        Eigen::Vector3d w_body(0.0, 0.0, 0.4);
        auto ct = makeTrajectory(0.1, 0.005, a_body, w_body, Ba_true_, Bg_true_);

        imu_ = ct.imu_biased;
        int L = ct.n_steps - 1;
        R0_ = ct.Rs[0];
        P0_ = ct.Ps[0];
        V0_ = ct.Vs[0];
        R1_ = ct.Rs[L];
        P1_ = ct.Ps[L];
        V1_ = ct.Vs[L];

        pint_ = std::make_shared<MidPointIntegrator>(
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
        for (auto& m : imu_) pint_->update(m);

        auto phi = [](const Eigen::Matrix3d& R) { return Sophus::SO3d(R).log(); };
        auto pv = phi(R0_), pv1 = phi(R1_);
        for (int d = 0; d < 3; ++d) {
            pi_[d] = pv[d];
            pi_[d + 3] = P0_[d];
            pj_[d] = pv1[d];
            pj_[d + 3] = P1_[d];
        }
    }

    Eigen::Matrix<double, 18, 18> noise_;
    Eigen::Vector3d Ba_true_, Bg_true_;
    std::vector<tassel_utils::IMUMeasurement> imu_;
    Eigen::Matrix3d R0_, R1_;
    Eigen::Vector3d P0_, V0_, P1_, V1_;
    std::shared_ptr<MidPointIntegrator> pint_;
    double pi_[6]{}, pj_[6]{};
};

TEST_F(IMUFactorTest, ZeroResidualAtGroundTruth) {
    double sb_i[9] = {V0_.x(),      V0_.y(),      V0_.z(),      Ba_true_.x(), Ba_true_.y(),
                      Ba_true_.z(), Bg_true_.x(), Bg_true_.y(), Bg_true_.z()};
    double sb_j[9] = {V1_.x(),      V1_.y(),      V1_.z(),      Ba_true_.x(), Ba_true_.y(),
                      Ba_true_.z(), Bg_true_.x(), Bg_true_.y(), Bg_true_.z()};

    IMUFactor<MidPointIntegrator> f(pint_);
    const double* p[] = {pi_, sb_i, pj_, sb_j};
    double r[15];
    f.Evaluate(p, r, nullptr);
    double rms = 0;
    for (int i = 0; i < 15; ++i) rms += r[i] * r[i];
    rms = std::sqrt(rms / 15.0);
    EXPECT_LT(rms, 1e-2) << "RMS=" << rms;
}

// ------------------------------------------------------------
// Full VIO test: IMU + Visual factors, perturbed initial poses
// ------------------------------------------------------------

class VIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        double an = 0.0193 * 0.0193, gn = 0.00264 * 0.00264;
        double aw = 0.000804 * 0.000804, gw = 0.0000703 * 0.0000703;
        noise_.block<3, 3>(0, 0) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = aw * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = gw * Eigen::Matrix3d::Identity();

        Ba_true_ = Eigen::Vector3d(0.1, -0.06, 0.03);
        Bg_true_ = Eigen::Vector3d(0.01, 0.005, -0.003);
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d::Zero();

        // 5 keyframes, ~0.12s apart, 200Hz IMU
        double total_dur = 0.6;
        double imu_dt = 0.005;
        int cam_skip = 24;
        num_frames_ = 5;

        Eigen::Vector3d a_body(0.2, -0.05, 0.0);
        Eigen::Vector3d w_body(0.0, 0.0, 0.3);

        auto ct = makeTrajectory(total_dur, imu_dt, a_body, w_body, Ba_true_, Bg_true_);

        int step = cam_skip;
        frames_R_.resize(num_frames_);
        frames_P_.resize(num_frames_);
        frames_V_.resize(num_frames_);
        for (int i = 0; i < num_frames_; ++i) {
            int k = i * step;
            frames_R_[i] = ct.Rs[k];
            frames_P_[i] = ct.Ps[k];
            frames_V_[i] = ct.Vs[k];
        }

        // Preintegrators (zero lin point)
        preints_.resize(num_frames_ - 1);
        for (int i = 0; i < num_frames_ - 1; ++i) {
            preints_[i] = std::make_shared<MidPointIntegrator>(
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
            int start_k = i * step, end_k = (i + 1) * step;
            for (int k = start_k; k <= end_k; ++k) preints_[i]->update(ct.imu_biased[k]);
        }

        // Landmarks with STRONG geometric diversity:
        //   - near (1-2m), medium (3-5m), far (8-15m)
        //   - spread left/right/up/down across the FOV
        //   - not all visible in every frame (some go out of FOV for later frames)
        num_landmarks_ = 12;
        landmarks_3d_.resize(num_landmarks_);
        // clang-format off
        landmarks_3d_[0]  = Eigen::Vector3d( 0.3, -0.2,  1.5);  // near, center
        landmarks_3d_[1]  = Eigen::Vector3d( 1.5, -0.1,  2.0);  // near-mid, right
        landmarks_3d_[2]  = Eigen::Vector3d(-1.2,  0.3,  2.5);  // near-mid, left
        landmarks_3d_[3]  = Eigen::Vector3d( 0.1,  0.8,  3.0);  // mid, up
        landmarks_3d_[4]  = Eigen::Vector3d( 0.2, -0.7,  3.5);  // mid, down
        landmarks_3d_[5]  = Eigen::Vector3d( 2.5, -0.3,  4.0);  // mid, far right
        landmarks_3d_[6]  = Eigen::Vector3d(-2.0,  0.2,  4.5);  // mid, far left
        landmarks_3d_[7]  = Eigen::Vector3d( 0.5,  1.2,  6.0);  // far, up
        landmarks_3d_[8]  = Eigen::Vector3d(-0.3, -1.0,  7.0);  // far, down
        landmarks_3d_[9]  = Eigen::Vector3d( 3.0, -0.5,  8.0);  // very far, right
        landmarks_3d_[10] = Eigen::Vector3d(-3.0,  0.5, 10.0);  // very far, left
        landmarks_3d_[11] = Eigen::Vector3d( 0.0,  0.0, 15.0);  // very far, center
        // clang-format on

        // Project to observations (all landmarks visible in all frames for simplicity,
        // but far landmarks at edges may go out of FOV for later frames due to motion)
        obs_uv_.resize(num_landmarks_);
        obs_valid_.resize(num_landmarks_);
        for (int lm = 0; lm < num_landmarks_; ++lm) {
            obs_uv_[lm].resize(num_frames_);
            obs_valid_[lm].resize(num_frames_, true);
            for (int f = 0; f < num_frames_; ++f) {
                Eigen::Vector3d p_c = frames_R_[f].transpose() * (landmarks_3d_[lm] - frames_P_[f]);
                obs_uv_[lm][f] = p_c / p_c.z();
                // Mark as invalid if outside reasonable FOV (for robustness test)
                if (std::abs(p_c.x() / p_c.z()) > 1.5 || std::abs(p_c.y() / p_c.z()) > 1.2)
                    obs_valid_[lm][f] = false;
            }
        }

        // Gyro/acc snapshots for visual factor td compensation
        gyro_vec_.resize(num_frames_, w_body);
        acc_vec_.resize(num_frames_);
        for (int f = 0; f < num_frames_; ++f)
            acc_vec_[f] = a_body + frames_R_[f].transpose() * tassel_utils::G + Ba_true_;

        // Initialize params: poses perturbed from ground truth,
        // biases at ZERO (wrong), velocities at ground truth
        params_pose_.resize(num_frames_);
        params_sb_.resize(num_frames_);
        params_inv_depth_.resize(num_landmarks_, 0.0);

        auto phi = [](const Eigen::Matrix3d& R) { return Sophus::SO3d(R).log(); };
        std::mt19937 rng(42);
        std::normal_distribution<double> pos_noise(0.0, 0.03);   // 3cm position noise
        std::normal_distribution<double> rot_noise(0.0, 0.02);   // ~1 deg rotation noise
        std::normal_distribution<double> depth_noise(0.0, 0.1);  // 10% depth noise

        for (int f = 0; f < num_frames_; ++f) {
            params_pose_[f] = std::array<double, 6>{};
            auto pv = phi(frames_R_[f]);
            for (int d = 0; d < 3; ++d) {
                params_pose_[f][d] = pv[d] + rot_noise(rng);
                params_pose_[f][d + 3] = frames_P_[f][d] + pos_noise(rng);
            }
            params_sb_[f] = std::array<double, 9>{};
            for (int d = 0; d < 3; ++d) {
                params_sb_[f][d] = frames_V_[f][d];  // V at ground truth
                params_sb_[f][d + 3] = 0.0;          // Ba wrong
                params_sb_[f][d + 6] = 0.0;          // Bg wrong
            }
        }
        for (int lm = 0; lm < num_landmarks_; ++lm) {
            double d_gt = landmarks_3d_[lm].z();
            params_inv_depth_[lm] = (1.0 / d_gt) * (1.0 + depth_noise(rng));
        }
    }

    ceres::Solver::Summary solve() {
        ceres::Problem problem;

        for (int f = 0; f < num_frames_; ++f) {
            problem.AddParameterBlock(params_pose_[f].data(), 6, new SE3RightManifold());
            problem.AddParameterBlock(params_sb_[f].data(), 9);
        }
        problem.SetParameterBlockConstant(params_pose_[0].data());

        // IMU factors
        for (int f = 0; f < num_frames_ - 1; ++f) {
            auto* imu = new IMUFactor<MidPointIntegrator>(preints_[f]);
            problem.AddResidualBlock(
                imu, nullptr, params_pose_[f].data(), params_sb_[f].data(),
                params_pose_[f + 1].data(), params_sb_[f + 1].data());
        }

        // Visual factors with Huber loss
        Eigen::Matrix2d sqrt_info = Eigen::Matrix2d::Identity() * 320;
        int vis_count = 0;
        for (int lm = 0; lm < num_landmarks_; ++lm) {
            for (int obs = 1; obs < num_frames_; ++obs) {
                if (!obs_valid_[lm][obs]) continue;
                int host = 0;
                int tgt = obs;
                auto* vis = new VisualFactor(
                    obs_uv_[lm][host], obs_uv_[lm][tgt], ric_, tic_, gyro_vec_[host],
                    gyro_vec_[tgt], acc_vec_[host], acc_vec_[tgt], params_sb_[host].data(),
                    params_sb_[tgt].data(), params_sb_[host].data() + 6, params_sb_[tgt].data() + 6,
                    sqrt_info);
                problem.AddResidualBlock(
                    vis, new ceres::HuberLoss(0.005), params_pose_[host].data(),
                    params_pose_[tgt].data(), &params_inv_depth_[lm], &td_param_);
                ++vis_count;
            }
        }
        problem.AddParameterBlock(&td_param_, 1);

        std::cout << "Visual factors: " << vis_count << "\n";

        ceres::Solver::Options opts;
        opts.linear_solver_type = ceres::DENSE_SCHUR;
        opts.max_num_iterations = 50;
        opts.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        return summary;
    }

    Eigen::Matrix<double, 18, 18> noise_;
    Eigen::Vector3d Ba_true_, Bg_true_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    double td_param_ = 0.0;

    int num_frames_, num_landmarks_;
    std::vector<Eigen::Matrix3d> frames_R_;
    std::vector<Eigen::Vector3d> frames_P_, frames_V_;
    std::vector<Eigen::Vector3d> landmarks_3d_;
    std::vector<std::vector<Eigen::Vector3d>> obs_uv_;
    std::vector<std::vector<bool>> obs_valid_;
    std::vector<Eigen::Vector3d> gyro_vec_, acc_vec_;

    std::vector<std::shared_ptr<MidPointIntegrator>> preints_;
    std::vector<std::array<double, 6>> params_pose_;
    std::vector<std::array<double, 9>> params_sb_;
    std::vector<double> params_inv_depth_;
};

TEST_F(VIOTest, BiasChangesInFullVIO) {
    auto phi = [](const Eigen::Matrix3d& R) { return Sophus::SO3d(R).log(); };

    // ---- Before optimization ----
    std::cout << "\n=== Before optimization ===\n";
    std::cout << "Biases:\n";
    for (int d = 0; d < 3; ++d)
        std::cout << "  Ba[0][" << d << "] = " << params_sb_[0][3 + d] << " (true=" << Ba_true_[d]
                  << ")\n";
    for (int d = 0; d < 3; ++d)
        std::cout << "  Bg[0][" << d << "] = " << params_sb_[0][6 + d] << " (true=" << Bg_true_[d]
                  << ")\n";

    std::cout << "Pose errors (perturbed initial):\n";
    for (int f = 1; f < num_frames_; ++f) {
        Eigen::Vector3d phi_est(params_pose_[f][0], params_pose_[f][1], params_pose_[f][2]);
        Eigen::Vector3d phi_gt = phi(frames_R_[f]);
        double rot_err_deg = (phi_est - phi_gt).norm() * 180.0 / M_PI;
        Eigen::Vector3d p_est(params_pose_[f][3], params_pose_[f][4], params_pose_[f][5]);
        double pos_err = (p_est - frames_P_[f]).norm();
        std::cout << "  frame[" << f << "] rot_err=" << rot_err_deg << " deg, pos_err=" << pos_err
                  << " m\n";
    }

    std::cout << "Velocity errors:\n";
    for (int f = 0; f < num_frames_; ++f) {
        Eigen::Vector3d v_est(params_sb_[f][0], params_sb_[f][1], params_sb_[f][2]);
        double v_err = (v_est - frames_V_[f]).norm();
        std::cout << "  frame[" << f << "] v_err=" << v_err << " m/s\n";
    }

    // ---- Jacobian diagnostic: evaluate first IMU factor with jacobians ----
    std::cout << "\n=== IMU factor 0->1 Jacobian diagnostic ===\n";
    {
        IMUFactor<MidPointIntegrator> imu_f(preints_[0]);
        const double* p[] = {
            params_pose_[0].data(), params_sb_[0].data(), params_pose_[1].data(),
            params_sb_[1].data()};
        double r[15];
        double J0[15 * 6], J1[15 * 9], J2[15 * 6], J3[15 * 9];
        double* jacs[] = {J0, J1, J2, J3};
        imu_f.Evaluate(p, r, jacs);

        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_sb_i(J1);
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_sb_j(J3);
        Eigen::Map<Eigen::Matrix<double, 15, 1>> res(r);

        std::cout << "Residual (15D):\n  dp=" << res.segment<3>(0).transpose()
                  << "\n  dq=" << res.segment<3>(3).transpose()
                  << "\n  dv=" << res.segment<3>(6).transpose()
                  << "\n  dba=" << res.segment<3>(9).transpose()
                  << "\n  dbg=" << res.segment<3>(12).transpose() << "\n";

        // sb = [V(3), Ba(3), Bg(3)], columns: V=[0,1,2], Ba=[3,4,5], Bg=[6,7,8]
        auto print_block = [](const std::string& name, const Eigen::Matrix<double, 15, 9>& J) {
            std::cout << name << ":\n";
            // dp w.r.t. V, Ba, Bg
            std::cout << "  dp/dV : " << J.block<3, 3>(0, 0).row(0) << "\n";
            std::cout << "  dp/dBa: " << J.block<3, 3>(0, 3).row(0) << "\n";
            std::cout << "  dp/dBg: " << J.block<3, 3>(0, 6).row(0) << "\n";
            // dv w.r.t. V, Ba, Bg
            std::cout << "  dv/dV : " << J.block<3, 3>(6, 0).row(0) << "\n";
            std::cout << "  dv/dBa: " << J.block<3, 3>(6, 3).row(0) << "\n";
            std::cout << "  dv/dBg: " << J.block<3, 3>(6, 6).row(0) << "\n";
            // dba, dbg priors
            std::cout << "  dba/dBa: " << J.block<3, 3>(9, 3).row(0) << "\n";
            std::cout << "  dbg/dBg: " << J.block<3, 3>(12, 6).row(0) << "\n";
        };
        print_block("J_speed_bias_i (frame 0)", J_sb_i);
        print_block("J_speed_bias_j (frame 1)", J_sb_j);

        // Norms
        std::cout << "Frobenius norms:\n";
        std::cout << "  ||dp/dV_i|| = " << J_sb_i.block<3, 3>(0, 0).norm() << "\n";
        std::cout << "  ||dp/dBa_i|| = " << J_sb_i.block<3, 3>(0, 3).norm() << "\n";
        std::cout << "  ||dv/dV_i|| = " << J_sb_i.block<3, 3>(6, 0).norm() << "\n";
        std::cout << "  ||dv/dBa_i|| = " << J_sb_i.block<3, 3>(6, 3).norm() << "\n";
        std::cout << "  ||dba/dBa_i|| = " << J_sb_i.block<3, 3>(9, 3).norm() << "\n";
        std::cout << "  ||dba/dBa_j|| = " << J_sb_j.block<3, 3>(9, 3).norm() << "\n";
    }

    auto summary = solve();

    // ---- After optimization ----
    std::cout << "\n=== After optimization ===\n";
    std::cout << "Ceres: " << summary.BriefReport() << "\n";
    std::cout << "Final cost: " << summary.final_cost << "\n";

    std::cout << "Biases:\n";
    for (int d = 0; d < 3; ++d)
        std::cout << "  Ba[0][" << d << "] = " << params_sb_[0][3 + d] << " (true=" << Ba_true_[d]
                  << ")\n";
    for (int d = 0; d < 3; ++d)
        std::cout << "  Bg[0][" << d << "] = " << params_sb_[0][6 + d] << " (true=" << Bg_true_[d]
                  << ")\n";

    std::cout << "Pose errors (after optimization):\n";
    for (int f = 1; f < num_frames_; ++f) {
        Eigen::Vector3d phi_est(params_pose_[f][0], params_pose_[f][1], params_pose_[f][2]);
        Eigen::Vector3d phi_gt = phi(frames_R_[f]);
        double rot_err_deg = (phi_est - phi_gt).norm() * 180.0 / M_PI;
        Eigen::Vector3d p_est(params_pose_[f][3], params_pose_[f][4], params_pose_[f][5]);
        double pos_err = (p_est - frames_P_[f]).norm();
        std::cout << "  frame[" << f << "] rot_err=" << rot_err_deg << " deg, pos_err=" << pos_err
                  << " m\n";
    }

    std::cout << "Velocity errors:\n";
    for (int f = 0; f < num_frames_; ++f) {
        Eigen::Vector3d v_est(params_sb_[f][0], params_sb_[f][1], params_sb_[f][2]);
        double v_err = (v_est - frames_V_[f]).norm();
        std::cout << "  frame[" << f << "] v_err=" << v_err << " m/s\n";
    }

    double ba_err = std::sqrt(
        (std::pow(params_sb_[0][3] - Ba_true_.x(), 2) +
         std::pow(params_sb_[0][4] - Ba_true_.y(), 2) +
         std::pow(params_sb_[0][5] - Ba_true_.z(), 2)) /
        3.0);
    std::cout << "Ba RMS error: " << ba_err << "\n";

    EXPECT_LT(ba_err, 0.05) << "Bias should move toward true value with visual constraints";
}

}  // namespace
}  // namespace tassel_core
