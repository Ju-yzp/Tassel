// =============================================================================
// test_reprojection_factor.cpp
//
// 目的：
//   验证 ReprojectionFactor 的解析雅各比和时间延迟 td 的优化反馈。
//
// 测试设计：
//   使用 imu_test_utils 生成 400Hz IMU 轨迹; 观测由 camera-time state 加真实 td 后的
//   query state 精确投影生成, factor 端从 camera-time state、IMU 读数和 td 近似恢复
//   query state。单因子检查雅各比, 多 landmark 优化检查 td 收敛。
//
// 通过条件：
//   位姿、速度、偏置、逆深度和 td 的解析雅各比通过数值微分检查; Ceres 优化后 td
//   明显靠近构造数据时使用的真实延迟。
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <sophus/so3.hpp>

#include "cam/camera_rad_tan.h"
#include "factor/reprojection_factor.h"
#include "imu_test_utils.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

// =============================================================================
// ReprojectionFactorTest
// =============================================================================

class ReprojectionFactorTest : public ::testing::Test {
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
        timeline_ = test::generateConstantMotionTimeline(2.0, imu_dt, a_body, w_body, Ba_, Bg_);

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

        // 从查询状态生成路标（精确计算，不使用近似）。
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

    ReprojectionFactor* makeFactor(int k) const {
        const auto& lm = lms_[k];
        return new ReprojectionFactor(
            lm.uv_i, lm.pt_j, ric_, tic_, w_i_, w_j_, a_i_, a_j_, v_i_, v_j_, bg_, bg_, ba_, ba_,
            sqrt_info_, &camera_);
    }

    // --- 数据 ---
    test::ImuTimeline timeline_;
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

TEST_F(ReprojectionFactorTest, JacobianCheck) {
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

    double plus_i_data[36], plus_j_data[36];
    manifold.PlusJacobian(pose_i_, plus_i_data);
    manifold.PlusJacobian(pose_j_, plus_j_data);
    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J0_ambient(J0);
    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J1_ambient(J1);
    Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus_i(plus_i_data);
    Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus_j(plus_j_data);
    const Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_tangent = J0_ambient * plus_i;
    const Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J1_tangent = J1_ambient * plus_j;

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
        if (e0 > tol || e1 > tol) {
            nbad++;
        }
        std::cout << "  " << label << "[col " << c << "] an=[" << a0 << "," << a1 << "] num=["
                  << num[0] << "," << num[1] << "] err=[" << e0 << "," << e1 << "]"
                  << ((e0 > tol || e1 > tol) ? " ***" : "") << "\n";
    };

    std::cout << "\n--- pose_i (2x6) ---\n";
    for (int c = 0; c < 6; ++c) {
        check("J_pose_i", J0_tangent.data(), c, num_pose(0, pose_i_, c), 6);
    }
    std::cout << "--- pose_j (2x6) ---\n";
    for (int c = 0; c < 6; ++c) {
        check("J_pose_j", J1_tangent.data(), c, num_pose(1, pose_j_, c), 6);
    }
    std::cout << "--- dt (2x1) ---\n";
    check("J_dt", J2, 0, num_scalar(2, dt_val), 1);
    std::cout << "--- inv_depth (2x1) ---\n";
    check("J_inv_depth", J3, 0, num_scalar(3, inv_depth), 1);

    std::cout << "\ntotal bad (>0.5%): " << nbad << "\n";
    EXPECT_EQ(nbad, 0);

    delete factor;
}

TEST_F(ReprojectionFactorTest, CeresGradientCheckerContract) {
    SE3RightManifold manifold;
    std::vector<const ceres::Manifold*> manifolds = {&manifold, &manifold, nullptr, nullptr};
    ceres::NumericDiffOptions options;
    options.relative_step_size = 1e-6;
    for (size_t k = 0; k < lms_.size(); ++k) {
        std::unique_ptr<ReprojectionFactor> factor(makeFactor(static_cast<int>(k)));
        ceres::GradientChecker checker(factor.get(), &manifolds, options);
        double inv_depth = lms_[k].inv_depth;
        double delay = td_;
        const double* parameters[] = {pose_i_, pose_j_, &delay, &inv_depth};
        ceres::GradientChecker::ProbeResults results;
        EXPECT_TRUE(checker.Probe(parameters, 5e-5, &results))
            << "landmark=" << k << " max_relative_error=" << results.maximum_relative_error << "\n"
            << results.error_log;
    }
}

TEST_F(ReprojectionFactorTest, HuberCorrectionMatchesCeres) {
    for (double delay : {td_, 0.2}) {
        std::unique_ptr<ReprojectionFactor> factor(makeFactor(0));
        ceres::HuberLoss loss(1.0);
        SE3RightManifold manifold;
        ceres::Problem::Options problem_options;
        problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem problem(problem_options);
        double inv_depth = lms_[0].inv_depth;
        problem.AddParameterBlock(pose_i_, 6, &manifold);
        problem.AddParameterBlock(pose_j_, 6, &manifold);
        problem.AddParameterBlock(&delay, 1);
        problem.AddParameterBlock(&inv_depth, 1);
        const auto residual_id =
            problem.AddResidualBlock(factor.get(), &loss, pose_i_, pose_j_, &delay, &inv_depth);

        double ceres_residual[2];
        double ceres_J0[12], ceres_J1[12], ceres_J2[2], ceres_J3[2];
        double* ceres_jacobians[] = {ceres_J0, ceres_J1, ceres_J2, ceres_J3};
        double cost = 0.0;
        ASSERT_TRUE(problem.EvaluateResidualBlock(
            residual_id, true, &cost, ceres_residual, ceres_jacobians));

        double raw_residual[2];
        double raw_J0[12], raw_J1[12], raw_J2[2], raw_J3[2];
        double* raw_jacobians[] = {raw_J0, raw_J1, raw_J2, raw_J3};
        const double* parameters[] = {pose_i_, pose_j_, &delay, &inv_depth};
        factor->Evaluate(parameters, raw_residual, raw_jacobians);

        double plus_i_data[36], plus_j_data[36];
        manifold.PlusJacobian(pose_i_, plus_i_data);
        manifold.PlusJacobian(pose_j_, plus_j_data);
        Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> raw_pose_i(raw_J0);
        Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> raw_pose_j(raw_J1);
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus_i(plus_i_data);
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus_j(plus_j_data);
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> expected_J0 = raw_pose_i * plus_i;
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> expected_J1 = raw_pose_j * plus_j;

        double rho[3];
        const Eigen::Map<const Eigen::Vector2d> raw_r(raw_residual);
        loss.Evaluate(raw_r.squaredNorm(), rho);
        const double scale = std::sqrt(rho[1]);
        expected_J0 *= scale;
        expected_J1 *= scale;
        Eigen::Map<Eigen::Vector2d>(raw_J2) *= scale;
        Eigen::Map<Eigen::Vector2d>(raw_J3) *= scale;
        const Eigen::Vector2d expected_residual = scale * raw_r;

        Eigen::Map<const Eigen::Vector2d> actual_residual(ceres_residual);
        Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> actual_J0(ceres_J0);
        Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> actual_J1(ceres_J1);
        EXPECT_TRUE(actual_residual.isApprox(expected_residual, 1e-12));
        EXPECT_TRUE(actual_J0.isApprox(expected_J0, 1e-12));
        EXPECT_TRUE(actual_J1.isApprox(expected_J1, 1e-12));
        EXPECT_TRUE(Eigen::Map<const Eigen::Vector2d>(ceres_J2).isApprox(
            Eigen::Map<const Eigen::Vector2d>(raw_J2), 1e-12));
        EXPECT_TRUE(Eigen::Map<const Eigen::Vector2d>(ceres_J3).isApprox(
            Eigen::Map<const Eigen::Vector2d>(raw_J3), 1e-12));
    }
}

// =============================================================================
// 测试 2：构建优化问题并验证 td 收敛
// =============================================================================

TEST_F(ReprojectionFactorTest, TdConvergence) {
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
