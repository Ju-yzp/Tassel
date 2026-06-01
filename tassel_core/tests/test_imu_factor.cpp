// =============================================================================
// test_imu_factor.cpp — IMU 因子视觉-惯性联合单元测试
// =============================================================================
//
// 测试目标:
//   1. 联合优化中真值处视觉+IMU 残差
//   2. 偏置随机游走后速度先验使偏置被正确恢复
//   3. td 优化回归: 初始 td=0 应收敛到真实值 (5ms)
//
// 数据流:
//   400Hz IMU 连续轨迹, 帧间偏置随机游走 (~0.002)
//   → 双目深度点用 t_cam+td 查询态精确投影 (无 td 补偿近似)
//   → 预积分线性化点 = base bias (模拟未标定偏置)
//   → 速度先验约束锁定 V → 优化器被迫调整偏置降低 IMU 残差
//
// 核心诊断背景:
//   ||dp/dV|| ~ 8980 >> ||dp/dBa|| ~ 179 (Hessian 曲率比 ~2500)
//   无速度先验时优化器优先调整速度而非偏置来吸收 IMU 残差
//
// =============================================================================

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

// =============================================================================
// 速度先验: 3D 残差, 参数块 speed_bias (9D)
//   residual = (V_est - V_prior) / sigma_v
// =============================================================================

struct VelocityPrior : public ceres::SizedCostFunction<3, 9> {
    VelocityPrior(const Eigen::Vector3d& v, double sigma) : v_prior_(v), sigma_(sigma) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        Eigen::Map<const Eigen::Vector3d> V(parameters[0]);
        Eigen::Map<Eigen::Vector3d> r(residuals);
        r = (V - v_prior_) / sigma_;

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() / sigma_;
        }
        return true;
    }

    Eigen::Vector3d v_prior_;
    double sigma_;
};

// =============================================================================
// ImuFactorTest — IMU 视觉-惯性联合测试夹具
// =============================================================================

class ImuFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // --- 基本参数 ---
        imu_dt_ = 0.0025;       // 400Hz
        steps_per_frame_ = 27;  // ~67.5ms → ~15Hz
        num_frames_ = 6;
        td_ = 0.005;                                              // 5ms
        td_steps_ = static_cast<int>(std::round(td_ / imu_dt_));  // 2

        // --- 外参 ---
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d(0.05, 0.0, 0.0);

        // --- 体运动 (三轴角速度+加速度, 激励充分) ---
        a_body_ = Eigen::Vector3d(0.3, -0.1, 0.05);
        w_body_ = Eigen::Vector3d(0.1, 0.3, 0.4);

        // --- 基础偏置 + 随机游走 ---
        Ba_base_ = Eigen::Vector3d(0.08, -0.04, 0.03);
        Bg_base_ = Eigen::Vector3d(0.012, -0.006, 0.004);

        std::mt19937 rng(42);
        std::normal_distribution<double> rw(0.0, 0.002);

        Ba_true_.resize(num_frames_);
        Bg_true_.resize(num_frames_);
        Ba_true_[0] = Ba_base_;
        Bg_true_[0] = Bg_base_;
        for (int f = 1; f < num_frames_; ++f) {
            Ba_true_[f] = Ba_true_[f - 1] + Eigen::Vector3d(rw(rng), rw(rng), rw(rng));
            Bg_true_[f] = Bg_true_[f - 1] + Eigen::Vector3d(rw(rng), rw(rng), rw(rng));
        }

        // --- 噪声矩阵 ---
        // V=I convention: noise[12:15] = per-step discrete variance.
        // Bias walk 0.002/frame over 27 steps → per-step var = 0.002² / 27.
        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        double an = 0.0193 * 0.0193, gn = 0.00264 * 0.00264;
        double aw = 1.48e-7, gw = 1.48e-7;
        noise_.block<3, 3>(0, 0) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = aw * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = gw * Eigen::Matrix3d::Identity();

        // =====================================================================
        // 生成连续 IMU 轨迹 (含偏置游走)
        // =====================================================================

        int total_steps = num_frames_ * steps_per_frame_ + 1;
        imu_states_.resize(total_steps);

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d P = Eigen::Vector3d::Zero();
        Eigen::Vector3d V = Eigen::Vector3d::Zero();

        for (int k = 0; k < total_steps; ++k) {
            int f = std::min(k / steps_per_frame_, num_frames_ - 1);
            Eigen::Vector3d Ba = Ba_true_[f];
            Eigen::Vector3d Bg = Bg_true_[f];

            ImuState s;
            s.ts = k * imu_dt_;
            s.R = R;
            s.P = P;
            s.V = V;
            s.Ba = Ba;
            s.Bg = Bg;
            s.gyro = w_body_ + Bg;
            s.acc = a_body_ + R.transpose() * tassel_utils::G + Ba;
            imu_states_[k] = s;

            if (k < total_steps - 1) {
                double dt2 = imu_dt_ * 0.5;
                Eigen::Matrix3d Rmid = R * Sophus::SO3d::exp(w_body_ * dt2).matrix();
                Eigen::Vector3d a_w_mid = Rmid * a_body_;
                Eigen::Vector3d V_next = V + a_w_mid * imu_dt_;
                Eigen::Vector3d P_next = P + (V + V_next) * dt2;
                Eigen::Matrix3d R_next = R * Sophus::SO3d::exp(w_body_ * imu_dt_).matrix();
                R = R_next;
                P = P_next;
                V = V_next;
            }
        }

        // --- 提取帧状态 (t_cam 时刻) ---
        frames_R_.resize(num_frames_);
        frames_P_.resize(num_frames_);
        frames_V_.resize(num_frames_);
        frames_gyro_.resize(num_frames_);
        frames_acc_.resize(num_frames_);
        for (int f = 0; f < num_frames_; ++f) {
            int k_cam = f * steps_per_frame_;
            frames_R_[f] = imu_states_[k_cam].R;
            frames_P_[f] = imu_states_[k_cam].P;
            frames_V_[f] = imu_states_[k_cam].V;
            frames_gyro_[f] = imu_states_[k_cam].gyro;
            frames_acc_[f] = imu_states_[k_cam].acc;
        }

        // =====================================================================
        // 生成双目路标 (查询态投影, 无 td 补偿近似)
        // =====================================================================

        sqrt_info_vis_ = Eigen::Matrix2d::Identity() * 320.0;

        std::vector<Eigen::Vector3d> P_cam_pts = {
            Eigen::Vector3d(0.3, -0.2, 1.5),
            Eigen::Vector3d(-0.5, 0.3, 3.0),
            Eigen::Vector3d(0.2, -0.4, 2.0),
            Eigen::Vector3d(1.2, -0.1, 8.0),
        };

        landmarks_.resize(num_frames_ - 1);
        for (int host = 0; host < num_frames_ - 1; ++host) {
            int target = host + 1;
            int k_q_host = host * steps_per_frame_ + td_steps_;
            int k_q_target = target * steps_per_frame_ + td_steps_;
            const auto& sq_h = imu_states_[k_q_host];
            const auto& sq_t = imu_states_[k_q_target];

            for (const auto& Pc : P_cam_pts) {
                Landmark lm;
                lm.host_id = host;
                lm.depth_i = Pc.z();
                lm.inv_depth = 1.0 / Pc.z();
                lm.uv_i = Pc / Pc.z();

                Eigen::Vector3d P_imu_i = ric_ * Pc + tic_;
                Eigen::Vector3d P_world = sq_h.R * P_imu_i + sq_h.P;
                Eigen::Vector3d P_imu_j = sq_t.R.transpose() * (P_world - sq_t.P);
                Eigen::Vector3d P_cam_j = ric_.transpose() * (P_imu_j - tic_);
                lm.uv_j = P_cam_j / P_cam_j.norm();

                landmarks_[host].push_back(lm);
            }
        }

        // =====================================================================
        // 预积分器 (线性化点 = base bias, 模拟未标定偏置)
        // =====================================================================

        preints_.resize(num_frames_ - 1);
        for (int f = 0; f < num_frames_ - 1; ++f) {
            preints_[f] = std::make_shared<MidPointIntegrator>(Ba_base_, Bg_base_, noise_);
            int start_k = f * steps_per_frame_;
            int end_k = (f + 1) * steps_per_frame_;
            for (int k = start_k; k <= end_k; ++k) {
                tassel_utils::IMUMeasurement m;
                m.timestamp = imu_states_[k].ts;
                m.acc = imu_states_[k].acc;
                m.gyro = imu_states_[k].gyro;
                preints_[f]->update(m);
            }
        }

        // =====================================================================
        // 初始化参数块
        //   - 位姿: 真值
        //   - 速度: 真值
        //   - 偏置: base (错误, 模拟未标定状态)
        // =====================================================================

        params_pose_.resize(num_frames_);
        params_sb_.resize(num_frames_);
        auto phi = [](const Eigen::Matrix3d& R) { return Sophus::SO3d(R).log(); };

        for (int f = 0; f < num_frames_; ++f) {
            params_pose_[f] = std::array<double, 6>{};
            auto pv = phi(frames_R_[f]);
            for (int d = 0; d < 3; ++d) {
                params_pose_[f][d] = frames_P_[f][d];
                params_pose_[f][d + 3] = pv[d];
            }

            params_sb_[f] = std::array<double, 9>{};
            for (int d = 0; d < 3; ++d) {
                params_sb_[f][d] = frames_V_[f][d];
                params_sb_[f][d + 3] = Ba_base_[d];
                params_sb_[f][d + 6] = Bg_base_[d];
            }
        }

        // inv_depth 参数
        inv_depths_.clear();
        for (int host = 0; host < num_frames_ - 1; ++host) {
            for (const auto& lm : landmarks_[host]) {
                inv_depths_.push_back(lm.inv_depth);
            }
        }

        td_param_ = td_;
    }

    // =========================================================================
    // 构建并求解联合优化
    // =========================================================================

    ceres::Solver::Summary solve(
        bool use_velocity_prior, double sigma_v, bool optimize_td, double td_init,
        bool fix_speed_bias = false, bool fix_all_poses = false) {
        ceres::Problem problem;

        // 参数块
        for (int f = 0; f < num_frames_; ++f) {
            problem.AddParameterBlock(params_pose_[f].data(), 6, new SE3RightManifold());
            problem.AddParameterBlock(params_sb_[f].data(), 9);
            if (fix_speed_bias) problem.SetParameterBlockConstant(params_sb_[f].data());
            if (fix_all_poses) problem.SetParameterBlockConstant(params_pose_[f].data());
        }
        if (!fix_all_poses) problem.SetParameterBlockConstant(params_pose_[0].data());

        // IMU 因子
        for (int f = 0; f < num_frames_ - 1; ++f) {
            auto* imu = new IMUFactor<MidPointIntegrator>(preints_[f]);
            problem.AddResidualBlock(
                imu, nullptr, params_pose_[f].data(), params_sb_[f].data(),
                params_pose_[f + 1].data(), params_sb_[f + 1].data());
        }

        // 视觉因子
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        int inv_idx = 0;
        for (int host = 0; host < num_frames_ - 1; ++host) {
            int target = host + 1;
            for (size_t l = 0; l < landmarks_[host].size(); ++l) {
                const auto& lm = landmarks_[host][l];
                auto* vis = new VisualFactor(
                    lm.uv_i, lm.uv_j, ric_, tic_, frames_gyro_[host], frames_gyro_[target],
                    frames_acc_[host], frames_acc_[target], params_sb_[host].data(),
                    params_sb_[target].data(), params_sb_[host].data() + 6,
                    params_sb_[target].data() + 6, sqrt_info_vis_);
                problem.AddResidualBlock(
                    vis, loss, params_pose_[host].data(), params_pose_[target].data(),
                    &inv_depths_[inv_idx], &td_param_);
                ++inv_idx;
            }
        }

        // td 参数
        td_param_ = td_init;
        problem.AddParameterBlock(&td_param_, 1);
        if (!optimize_td) {
            problem.SetParameterBlockConstant(&td_param_);
        }

        // 速度先验
        if (use_velocity_prior) {
            for (int f = 0; f < num_frames_; ++f) {
                auto* vp = new VelocityPrior(frames_V_[f], sigma_v);
                problem.AddResidualBlock(vp, nullptr, params_sb_[f].data());
            }
        }

        ceres::Solver::Options opts;
        opts.linear_solver_type = ceres::DENSE_SCHUR;
        opts.max_num_iterations = 100;
        opts.minimizer_progress_to_stdout = false;
        opts.num_threads = 1;

        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        return summary;
    }

    void printDiagnostics(const std::string& label) {
        std::cout << "\n=== " << label << " ===\n";
        for (int f = 0; f < num_frames_; ++f) {
            double dBa = (Eigen::Vector3d(params_sb_[f][3], params_sb_[f][4], params_sb_[f][5]) -
                          Ba_true_[f])
                             .norm();
            double dBg = (Eigen::Vector3d(params_sb_[f][6], params_sb_[f][7], params_sb_[f][8]) -
                          Bg_true_[f])
                             .norm();
            double dV = (Eigen::Vector3d(params_sb_[f][0], params_sb_[f][1], params_sb_[f][2]) -
                         frames_V_[f])
                            .norm();
            std::cout << "f[" << f << "] dBa=" << dBa << " dBg=" << dBg << " dV=" << dV
                      << " | Ba_opt=[" << params_sb_[f][3] << "," << params_sb_[f][4] << ","
                      << params_sb_[f][5] << "] Ba_true=[" << Ba_true_[f].transpose() << "]\n";
        }
        std::cout << "td: " << td_param_ << " (true=" << td_ << ")\n";
    }

    // =========================================================================
    // 数据成员
    // =========================================================================

    struct ImuState {
        double ts;
        Eigen::Matrix3d R;
        Eigen::Vector3d P, V;
        Eigen::Vector3d gyro, acc;
        Eigen::Vector3d Ba, Bg;
    };

    struct Landmark {
        int host_id;
        Eigen::Vector3d uv_i, uv_j;
        double depth_i, inv_depth;
    };

    // 参数
    double imu_dt_, td_;
    int steps_per_frame_, num_frames_, td_steps_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d a_body_, w_body_;
    Eigen::Vector3d Ba_base_, Bg_base_;
    Eigen::Matrix<double, 18, 18> noise_;
    Eigen::Matrix2d sqrt_info_vis_;

    // 真值
    std::vector<Eigen::Vector3d> Ba_true_, Bg_true_;
    std::vector<ImuState> imu_states_;
    std::vector<Eigen::Matrix3d> frames_R_;
    std::vector<Eigen::Vector3d> frames_P_, frames_V_, frames_gyro_, frames_acc_;
    std::vector<std::vector<Landmark>> landmarks_;

    // 预积分器
    std::vector<std::shared_ptr<MidPointIntegrator>> preints_;

    // 参数块
    std::vector<std::array<double, 6>> params_pose_;
    std::vector<std::array<double, 9>> params_sb_;
    std::vector<double> inv_depths_;
    double td_param_;
};

// =============================================================================
// 测试 1: 真值处残差
// =============================================================================

TEST_F(ImuFactorTest, ZeroResidualAtGroundTruth) {
    for (int f = 0; f < num_frames_; ++f) {
        for (int d = 0; d < 3; ++d) {
            params_sb_[f][d + 3] = Ba_true_[f][d];
            params_sb_[f][d + 6] = Bg_true_[f][d];
        }
    }
    td_param_ = td_;

    std::cout << "\n--- Per-factor cost diagnostic ---\n";
    double total_cost = 0.0;
    for (int f = 0; f < num_frames_ - 1; ++f) {
        IMUFactor<MidPointIntegrator> imu_f(preints_[f]);
        const double* imu_p[] = {
            params_pose_[f].data(), params_sb_[f].data(), params_pose_[f + 1].data(),
            params_sb_[f + 1].data()};
        double imu_r[15];
        imu_f.Evaluate(imu_p, imu_r, nullptr);
        Eigen::Map<Eigen::Matrix<double, 15, 1>> rv(imu_r);
        double imu_cost = 0.5 * rv.squaredNorm();
        total_cost += imu_cost;
        std::cout << "IMU[" << f << "] cost=" << imu_cost << " r_dp=" << rv.segment<3>(0).norm()
                  << " r_dq=" << rv.segment<3>(3).norm() << " r_dv=" << rv.segment<3>(6).norm()
                  << " r_dba=" << rv.segment<3>(9).norm() << " r_dbg=" << rv.segment<3>(12).norm()
                  << "\n";
    }
    std::cout << "Total cost (manual): " << total_cost << "\n";

    auto summary = solve(false, 0.01, false, td_);
    std::cout << "Zero-residual: " << summary.BriefReport() << "\n";
    EXPECT_LT(total_cost, 100) << "Joint residual at ground truth with bias walk";
}

// =============================================================================
// 测试 2: 无速度先验 → 偏置几乎不动 (复现已知问题)
// =============================================================================

TEST_F(ImuFactorTest, BiasesBarelyMoveWithoutVelocityPrior) {
    printDiagnostics("Before (no prior)");
    auto summary = solve(false, 0.01, false, td_);
    printDiagnostics("After (no prior)");
    std::cout << summary.BriefReport() << "\n";
}

// =============================================================================
// 测试 3: 有速度先验 → 偏置向真值收敛
// =============================================================================

TEST_F(ImuFactorTest, BiasesConvergeWithVelocityPrior) {
    printDiagnostics("Before (with prior)");

    auto summary = solve(true, 0.01, false, td_);

    printDiagnostics("After (with prior)");
    std::cout << summary.BriefReport() << "\n";

    double avg_dBa = 0, avg_dBg = 0;
    for (int f = 0; f < num_frames_; ++f) {
        avg_dBa +=
            (Eigen::Vector3d(params_sb_[f][3], params_sb_[f][4], params_sb_[f][5]) - Ba_true_[f])
                .norm();
        avg_dBg +=
            (Eigen::Vector3d(params_sb_[f][6], params_sb_[f][7], params_sb_[f][8]) - Bg_true_[f])
                .norm();
    }
    avg_dBa /= num_frames_;
    avg_dBg /= num_frames_;

    std::cout << "Avg Ba error (with prior): " << avg_dBa << "\n";
    std::cout << "Avg Bg error (with prior): " << avg_dBg << "\n";

    EXPECT_LT(avg_dBa, 0.03) << "Ba should converge toward truth with velocity prior";
    EXPECT_LT(avg_dBg, 0.01) << "Bg should converge toward truth with velocity prior";
}

// =============================================================================
// 测试 4: td 优化回归
// =============================================================================

TEST_F(ImuFactorTest, TdOptimizationRegression) {
    for (int f = 0; f < num_frames_; ++f) {
        for (int d = 0; d < 3; ++d) {
            params_sb_[f][d + 3] = Ba_true_[f][d];
            params_sb_[f][d + 6] = Bg_true_[f][d];
        }
    }

    auto summary = solve(false, 0.01, true, 0.0, true, true);
    std::cout << "Td optimization: " << summary.BriefReport() << "\n"
              << "  td: initial=0.0, final=" << td_param_ << ", true=" << td_ << "\n";

    EXPECT_NEAR(td_param_, td_, 0.001) << "td should converge to ~5ms";
}

// =============================================================================
// 测试 5: 固定位姿+速度, 仅优化偏置 — 验证 IMU 因子的 ∂r/∂Ba, ∂r/∂Bg
// =============================================================================

TEST_F(ImuFactorTest, BiasOnlyRecoveryWithTrueVelocity) {
    // 1. 设置偏置为真值 (确保 ground truth 处残差为零)
    for (int f = 0; f < num_frames_; ++f) {
        for (int d = 0; d < 3; ++d) {
            params_sb_[f][d + 3] = Ba_true_[f][d];
            params_sb_[f][d + 6] = Bg_true_[f][d];
        }
    }
    td_param_ = td_;

    // 2. 固定所有位姿
    ceres::Problem problem;
    for (int f = 0; f < num_frames_; ++f) {
        problem.AddParameterBlock(params_pose_[f].data(), 6, new SE3RightManifold());
        problem.SetParameterBlockConstant(params_pose_[f].data());

        problem.AddParameterBlock(params_sb_[f].data(), 9);
    }

    // 3. IMU 因子
    for (int f = 0; f < num_frames_ - 1; ++f) {
        auto* imu = new IMUFactor<MidPointIntegrator>(preints_[f]);
        problem.AddResidualBlock(
            imu, nullptr, params_pose_[f].data(), params_sb_[f].data(), params_pose_[f + 1].data(),
            params_sb_[f + 1].data());
    }

    // 4. 极紧速度先验 — 把 V 钉在真值上
    double sigma_tight = 1e-8;
    for (int f = 0; f < num_frames_; ++f) {
        auto* vp = new VelocityPrior(frames_V_[f], sigma_tight);
        problem.AddResidualBlock(vp, nullptr, params_sb_[f].data());
    }

    // 5. 偏置回退到 base (错误初始值)
    for (int f = 0; f < num_frames_; ++f) {
        for (int d = 0; d < 3; ++d) {
            params_sb_[f][d + 3] = Ba_base_[d];
            params_sb_[f][d + 6] = Bg_base_[d];
        }
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = 100;
    opts.minimizer_progress_to_stdout = false;
    opts.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << "Bias-only recovery: " << summary.BriefReport() << "\n";

    // 诊断
    double avg_dBa = 0, avg_dBg = 0;
    for (int f = 0; f < num_frames_; ++f) {
        double dBa =
            (Eigen::Vector3d(params_sb_[f][3], params_sb_[f][4], params_sb_[f][5]) - Ba_true_[f])
                .norm();
        double dBg =
            (Eigen::Vector3d(params_sb_[f][6], params_sb_[f][7], params_sb_[f][8]) - Bg_true_[f])
                .norm();
        avg_dBa += dBa;
        avg_dBg += dBg;
        std::cout << "f[" << f << "] dBa=" << dBa << " dBg=" << dBg << "\n";
    }
    avg_dBa /= num_frames_;
    avg_dBg /= num_frames_;
    std::cout << "Avg: dBa=" << avg_dBa << " dBg=" << avg_dBg << "\n";

    EXPECT_LT(avg_dBa, 0.001) << "Ba should converge to truth with true velocity";
    EXPECT_LT(avg_dBg, 0.001) << "Bg should converge to truth with true velocity";
}

}  // namespace
}  // namespace tassel_core
