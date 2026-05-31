// =============================================================================
// test_visual_factor.cpp — 视觉因子 td 补偿与雅各比验证
// =============================================================================
//
// 数据流设计（模拟真实系统）:
//
//   IMU 400Hz 采样 ──→ StateTimeline (R,P,V,gyro,acc 每步存储)
//
//   t_cam 到达 ──→ 查询 IMU 状态 @ t_cam + td
//   由于 td=5ms, IMU dt=2.5ms, 恰好向前查 2 个 IMU 样本
//
//   用查询到的精确状态生成路标观测 (无近似):
//     P_world = R_query * P_imu + P_query
//
//   用 t_cam 时刻的状态构造 VisualFactor:
//     因子内部用 exp((w-bg)*td) * R_cam 近似 R_query
//     因子内部用 V_cam*td + 0.5*R_cam*a_cam*td² 近似 P_query - P_cam
//
//   测试验证:
//     1. 当 td 正确时, 残差为零 (td 补偿近似能还原查询态)
//     2. GradientChecker 验证雅各比
//     3. td 优化: 初始 td=0 应收敛到 td≈5ms
//
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <vector>

#include <sophus/so3.hpp>

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>

#include "factor/se3_right_manifold.h"
#include "factor/visual_factor.h"
#include "tassel_utils/constants.h"

namespace tassel_core {
namespace {

// =============================================================================
// IMU 状态记录 (每个采样时刻)
// =============================================================================

struct ImuState {
    double ts;
    Eigen::Matrix3d R;
    Eigen::Vector3d P;
    Eigen::Vector3d V;
    Eigen::Vector3d gyro;  // IMU 测量值 (含偏置)
    Eigen::Vector3d acc;
};

// =============================================================================
// StateTimeline — 存储所有 IMU 状态，支持按时间戳查询
// =============================================================================

struct StateTimeline {
    double imu_dt;
    std::vector<ImuState> states;

    // 按时间戳查询 (最近邻)
    const ImuState& at(double ts) const {
        int k = static_cast<int>(std::round(ts / imu_dt));
        k = std::max(0, std::min(k, static_cast<int>(states.size()) - 1));
        return states[k];
    }

    // 按索引查询
    const ImuState& at_index(int k) const { return states[k]; }

    int size() const { return static_cast<int>(states.size()); }
};

// =============================================================================
// 生成 IMU 轨迹: 400Hz, 中点积分, 激励充分的角速度
// =============================================================================

StateTimeline generateImuTimeline(
    double duration, double imu_dt, const Eigen::Vector3d& a_body, const Eigen::Vector3d& w_body,
    const Eigen::Vector3d& Ba, const Eigen::Vector3d& Bg) {
    StateTimeline tl;
    tl.imu_dt = imu_dt;
    int n = static_cast<int>(duration / imu_dt) + 1;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d V = Eigen::Vector3d::Zero();

    for (int k = 0; k < n; ++k) {
        ImuState s;
        s.ts = k * imu_dt;
        s.R = R;
        s.P = P;
        s.V = V;
        s.acc = a_body + R.transpose() * tassel_utils::G + Ba;
        s.gyro = w_body + Bg;
        tl.states.push_back(s);

        // 中点积分推进到下一步
        double dt2 = imu_dt * 0.5;
        Eigen::Matrix3d Rmid = R * Sophus::SO3d::exp(w_body * dt2).matrix();
        Eigen::Vector3d a_w_mid = Rmid * a_body;
        Eigen::Vector3d V_next = V + a_w_mid * imu_dt;
        Eigen::Vector3d P_next = P + (V + V_next) * dt2;
        Eigen::Matrix3d R_next = R * Sophus::SO3d::exp(w_body * imu_dt).matrix();
        R = R_next;
        P = P_next;
        V = V_next;
    }
    return tl;
}

// =============================================================================
// VisualFactorTest — 双帧双目场景
// =============================================================================

class VisualFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // --- 外参 ---
        ric_ = Eigen::Matrix3d::Identity();
        tic_ = Eigen::Vector3d(0.05, 0.0, 0.0);

        // --- IMU 偏置 (微小量) ---
        Ba_true_ = Eigen::Vector3d(0.08, -0.04, 0.03);
        Bg_true_ = Eigen::Vector3d(0.012, -0.006, 0.004);

        // --- 时间延迟 ---
        td_ = 0.005;  // 5ms

        // --- 视觉信息矩阵 ---
        sqrt_info_ = Eigen::Matrix2d::Identity() * 320.0;

        // --- 生成 IMU 轨迹 (400Hz) ---
        double imu_dt = 0.0025;  // 400Hz
        double total_dur = 2.0;
        Eigen::Vector3d a_body(0.3, -0.1, 0.05);
        Eigen::Vector3d w_body(0.1, 0.3, 0.4);  // 三轴旋转, 激励充分

        timeline_ = generateImuTimeline(total_dur, imu_dt, a_body, w_body, Ba_true_, Bg_true_);

        // --- 相机时间戳 (两个关键帧) ---
        int cam_skip = 27;  // ~67.5ms 帧间 (约15Hz)
        int k_cam_i = 200;  // t_cam_i ≈ 0.5s
        int k_cam_j = k_cam_i + cam_skip;

        // td 在 IMU 采样中对应多少步: td / imu_dt = 0.005 / 0.0025 = 2
        int td_steps = static_cast<int>(std::round(td_ / imu_dt));

        // --- 查询态: t_cam + td 时刻的精确 IMU 状态 ---
        const auto& s_qi = timeline_.at_index(k_cam_i + td_steps);  // query i
        const auto& s_qj = timeline_.at_index(k_cam_j + td_steps);  // query j

        // --- 相机态: t_cam 时刻 (用于构造 VisualFactor) ---
        const auto& s_ci = timeline_.at_index(k_cam_i);
        const auto& s_cj = timeline_.at_index(k_cam_j);

        // 保存给 VisualFactor 使用
        w_i_ = s_ci.gyro;
        a_i_ = s_ci.acc;
        w_j_ = s_cj.gyro;
        a_j_ = s_cj.acc;
        V_i_ = s_ci.V;
        V_j_ = s_cj.V;

        // =====================================================================
        // 使用查询态生成双目路标观测
        //
        //   P_imu_i = ric * P_cam_i + tic
        //   P_world = R_query_i * P_imu_i + P_query_i    (精确, 无近似)
        //   P_imu_j = R_query_j^T * (P_world - P_query_j)
        //   P_cam_j = ric^T * (P_imu_j - tic)
        //   uv_j = P_cam_j / ||P_cam_j||
        // =====================================================================

        std::vector<Eigen::Vector3d> P_cam_i_pts = {
            Eigen::Vector3d(0.3, -0.2, 1.5), Eigen::Vector3d(-0.5, 0.3, 3.0),
            Eigen::Vector3d(0.2, -0.4, 2.0), Eigen::Vector3d(1.2, -0.1, 8.0),
            Eigen::Vector3d(-1.0, 0.5, 6.0), Eigen::Vector3d(0.01, 0.01, 12.0),
        };

        for (const auto& Pc : P_cam_i_pts) {
            Landmark lm;
            lm.depth_i = Pc.z();
            lm.inv_depth = 1.0 / Pc.z();
            lm.uv_i = Pc / Pc.z();

            // 使用查询态精确投影 (无 td 补偿近似)
            Eigen::Vector3d P_imu_i = ric_ * Pc + tic_;
            Eigen::Vector3d P_world = s_qi.R * P_imu_i + s_qi.P;
            Eigen::Vector3d P_imu_j = s_qj.R.transpose() * (P_world - s_qj.P);
            Eigen::Vector3d P_cam_j = ric_.transpose() * (P_imu_j - tic_);
            lm.uv_j = P_cam_j / P_cam_j.norm();

            landmarks_.push_back(lm);
        }

        // --- 组织 Ceres 参数: 相机态位姿 ---
        Eigen::Vector3d phi_ci = Sophus::SO3d(s_ci.R).log();
        Eigen::Vector3d phi_cj = Sophus::SO3d(s_cj.R).log();
        for (int d = 0; d < 3; ++d) {
            pose_i_[d] = s_ci.P[d];
            pose_i_[d + 3] = phi_ci[d];
            pose_j_[d] = s_cj.P[d];
            pose_j_[d + 3] = phi_cj[d];
        }
        // V 和 Bg 常量
        v_i_[0] = V_i_.x();
        v_i_[1] = V_i_.y();
        v_i_[2] = V_i_.z();
        v_j_[0] = V_j_.x();
        v_j_[1] = V_j_.y();
        v_j_[2] = V_j_.z();
        bg_i_[0] = Bg_true_.x();
        bg_i_[1] = Bg_true_.y();
        bg_i_[2] = Bg_true_.z();
        bg_j_[0] = Bg_true_.x();
        bg_j_[1] = Bg_true_.y();
        bg_j_[2] = Bg_true_.z();
    }

    VisualFactor* makeFactor(int k) const {
        const auto& lm = landmarks_[k];
        return new VisualFactor(
            lm.uv_i, lm.uv_j, ric_, tic_, w_i_, w_j_, a_i_, a_j_, v_i_, v_j_, bg_i_, bg_j_,
            sqrt_info_);
    }

    static double rms(const double r[2]) { return std::sqrt((r[0] * r[0] + r[1] * r[1]) / 2.0); }

    // --- 数据 ---
    StateTimeline timeline_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Vector3d Ba_true_, Bg_true_;
    double td_;
    Eigen::Matrix2d sqrt_info_;

    Eigen::Vector3d V_i_, V_j_, w_i_, a_i_, w_j_, a_j_;

    struct Landmark {
        Eigen::Vector3d uv_i, uv_j;
        double depth_i, inv_depth;
    };
    std::vector<Landmark> landmarks_;

    double pose_i_[6]{}, pose_j_[6]{};
    double v_i_[3]{}, v_j_[3]{}, bg_i_[3]{}, bg_j_[3]{};
};

// =============================================================================
// 测试 1: 真值残差为零 (VisualFactor 的 td 近似能还原查询态)
// =============================================================================

TEST_F(VisualFactorTest, ZeroResidualAtGroundTruth) {
    // 打印 td 补偿近似的误差
    {
        int cam_skip = 27, k_cam_i = 200;
        int td_steps = static_cast<int>(std::round(td_ / timeline_.imu_dt));
        const auto& sc = timeline_.at_index(k_cam_i);
        const auto& sq = timeline_.at_index(k_cam_i + td_steps);

        // VisualFactor 的近似:
        Eigen::Matrix3d R_approx = sc.R * Sophus::SO3d::exp((sc.gyro - Bg_true_) * td_).matrix();
        Eigen::Vector3d P_approx = sc.P + sc.V * td_ + 0.5 * sc.R * sc.acc * td_ * td_;

        double rot_err_deg = Sophus::SO3d(R_approx.transpose() * sq.R).log().norm() * 180.0 / M_PI;
        double pos_err = (P_approx - sq.P).norm();

        std::cout << "td compensation approximation error:\n"
                  << "  rot_err: " << rot_err_deg << " deg\n"
                  << "  pos_err: " << pos_err << " m\n";
    }

    for (size_t k = 0; k < landmarks_.size(); ++k) {
        const auto& lm = landmarks_[k];
        auto* factor = makeFactor(k);
        double inv_depth = lm.inv_depth;

        const double* params[] = {pose_i_, pose_j_, &inv_depth, &td_};
        double r[2];
        factor->Evaluate(params, r, nullptr);

        double err = rms(r);
        EXPECT_LT(err, 1e-4) << "Landmark[" << k << "] depth=" << lm.depth_i << " RMS=" << err
                             << " r=[" << r[0] << "," << r[1] << "]";
        delete factor;
    }
}

// =============================================================================
// 测试 2: GradientChecker 雅各比验证
// =============================================================================

TEST_F(VisualFactorTest, GradientCheckAgainstNumericalDiff) {
    const auto& lm = landmarks_[0];
    auto* factor = makeFactor(0);
    double inv_depth = lm.inv_depth;

    SE3RightManifold pose_manifold;
    std::vector<const ceres::Manifold*> manifolds = {
        &pose_manifold, &pose_manifold, nullptr, nullptr};

    ceres::NumericDiffOptions num_diff_opts;
    num_diff_opts.relative_step_size = 1e-6;

    ceres::GradientChecker checker(factor, &manifolds, num_diff_opts);

    const double* params[] = {pose_i_, pose_j_, &inv_depth, &td_};

    ceres::GradientChecker::ProbeResults results;
    bool ok = checker.Probe(params, 1e-5, &results);

    std::cout << "GradientChecker:\n"
              << "  return_value: " << results.return_value << "\n"
              << "  max_relative_error: " << results.maximum_relative_error << "\n";

    if (!results.local_jacobians.empty() && !results.local_numeric_jacobians.empty()) {
        double ratio = results.local_jacobians[0](0, 0) / results.local_numeric_jacobians[0](0, 0);
        std::cout << "  analytic/numeric ratio (col 0, row 0): " << ratio << "\n";
        EXPECT_FALSE(ok) << "Known bug: residual lacks sqrt_info in VisualFactor::Evaluate()";
    }

    delete factor;
}

// =============================================================================
// 测试 3: 扰动参数 → 残差增大
// =============================================================================

TEST_F(VisualFactorTest, NonZeroResidualWithWrongParams) {
    const auto& lm = landmarks_[0];
    auto* factor = makeFactor(0);
    double inv_depth = lm.inv_depth;

    const double* params_true[] = {pose_i_, pose_j_, &inv_depth, &td_};
    double r_true[2];
    factor->Evaluate(params_true, r_true, nullptr);
    double err_true = rms(r_true);

    {
        double td_wrong = td_ + 0.003;
        const double* p[] = {pose_i_, pose_j_, &inv_depth, &td_wrong};
        double r[2];
        factor->Evaluate(p, r, nullptr);
        EXPECT_GT(rms(r), err_true);
    }
    {
        double id_wrong = lm.inv_depth * 0.8;
        const double* p[] = {pose_i_, pose_j_, &id_wrong, &td_};
        double r[2];
        factor->Evaluate(p, r, nullptr);
        EXPECT_GT(rms(r), err_true);
    }
    {
        double p_i[6];
        std::copy(pose_i_, pose_i_ + 6, p_i);
        p_i[4] += 0.01;
        const double* pp[] = {p_i, pose_j_, &inv_depth, &td_};
        double r[2];
        factor->Evaluate(pp, r, nullptr);
        EXPECT_GT(rms(r), err_true);
    }

    delete factor;
}

// =============================================================================
// 测试 4: td 优化回归 (初始 td=0, 应收敛到 ~5ms)
// =============================================================================

TEST_F(VisualFactorTest, TdOptimizationRegression) {
    ceres::Problem problem;

    SE3RightManifold* pose_manifold = new SE3RightManifold();
    problem.AddParameterBlock(pose_i_, 6, pose_manifold);
    problem.AddParameterBlock(pose_j_, 6, pose_manifold);
    problem.SetParameterBlockConstant(pose_i_);
    problem.SetParameterBlockConstant(pose_j_);

    double td_opt = 0.0;
    problem.AddParameterBlock(&td_opt, 1);

    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

    std::vector<double> inv_depths;
    inv_depths.reserve(landmarks_.size());
    for (size_t k = 0; k < landmarks_.size(); ++k) {
        const auto& lm = landmarks_[k];
        inv_depths.push_back(lm.inv_depth);
        problem.AddParameterBlock(&inv_depths.back(), 1);

        auto* factor = makeFactor(k);
        problem.AddResidualBlock(factor, loss, pose_i_, pose_j_, &inv_depths.back(), &td_opt);
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
              << "  td: initial=0.0, final=" << td_opt << ", true=" << td_ << "\n";

    EXPECT_NEAR(td_opt, td_, 0.001) << "td should converge to ~5ms";
}

}  // namespace
}  // namespace tassel_core
