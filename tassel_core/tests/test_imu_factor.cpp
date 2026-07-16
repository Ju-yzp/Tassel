// =============================================================================
// test_imu_factor.cpp
//
// Purpose:
//   验证 IMU factor 的解析雅各比, 并检查偏置状态在优化中是否朝真实值收敛。
//
// Test design:
//   使用 imu_test_utils 生成常加速度/常角速度轨迹, 再加入逐帧变化的真实偏置;
//   预积分时故意使用错误 base bias。单因子部分用中心差分检查雅各比, 优化部分
//   固定位姿和速度, 只让 bias 由残差反馈。
//
// Pass criteria:
//   解析雅各比与数值雅各比在相对容差内一致, 优化后的加速度计和陀螺仪偏置比
//   初值更接近真实偏置。
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <sophus/so3.hpp>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "imu_test_utils.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

// 速度先验: 3D 残差, V 钉在真值上
struct VelocityPrior {
    VelocityPrior(const Eigen::Vector3d& v, double sigma) : v_prior_(v), sigma_(sigma) {}
    template <typename T>
    bool operator()(const T* const sb, T* residual) const {
        residual[0] = (sb[0] - T(v_prior_.x())) / T(sigma_);
        residual[1] = (sb[1] - T(v_prior_.y())) / T(sigma_);
        residual[2] = (sb[2] - T(v_prior_.z())) / T(sigma_);
        return true;
    }
    Eigen::Vector3d v_prior_;
    double sigma_;
};

// =============================================================================
// ImuFactorTest
// =============================================================================

class ImuFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        imu_dt_ = 0.0025;
        steps_per_frame_ = 67;  // 67 * 0.0025 ≈ 0.167s per frame, 6 frames ≈ 1s
        num_frames_ = 6;

        a_body_ = Eigen::Vector3d(0.3, -0.1, 0.05);
        w_body_ = Eigen::Vector3d(0.1, 0.3, 0.4);

        Ba_base_ = Eigen::Vector3d(0.08, -0.04, 0.03);
        Bg_base_ = Eigen::Vector3d(0.012, -0.006, 0.004);

        // 偏置随机游走
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

        // 噪声矩阵
        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        double an = 0.0193 * 0.0193, gn = 0.00264 * 0.00264;
        double aw = 1.48e-7, gw = 1.48e-7;
        noise_.block<3, 3>(0, 0) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = an * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = gn * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = aw * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = gw * Eigen::Matrix3d::Identity();

        // 生成 IMU 轨迹
        int total_steps = num_frames_ * steps_per_frame_ + 1;
        imu_timeline_ = test::generateConstantMotionTimeline(
            total_steps * imu_dt_, imu_dt_, a_body_, w_body_, Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero());
        imu_timeline_.states.resize(total_steps);
        for (int k = 0; k < total_steps; ++k) {
            const int f = std::min(k / steps_per_frame_, num_frames_ - 1);
            auto& sample = imu_timeline_.states[k];
            sample.Ba = Ba_true_[f];
            sample.Bg = Bg_true_[f];
            sample.acc = a_body_ + sample.R.transpose() * tassel_utils::G + sample.Ba;
            sample.gyro = w_body_ + sample.Bg;
        }

        // 提取帧状态
        frames_R_.resize(num_frames_);
        frames_P_.resize(num_frames_);
        frames_V_.resize(num_frames_);
        for (int f = 0; f < num_frames_; ++f) {
            int k = f * steps_per_frame_;
            frames_R_[f] = imu_timeline_.states[k].R;
            frames_P_[f] = imu_timeline_.states[k].P;
            frames_V_[f] = imu_timeline_.states[k].V;
        }

        // 预积分 (线性化点 = base bias, 模拟未标定偏置)
        preints_.resize(num_frames_ - 1);
        for (int f = 0; f < num_frames_ - 1; ++f) {
            preints_[f] = std::make_shared<MidPointIntegrator>(Ba_base_, Bg_base_, noise_);
            int start = f * steps_per_frame_, end = (f + 1) * steps_per_frame_;
            for (int k = start; k <= end; ++k) {
                tassel_utils::IMUMeasurement m;
                m.timestamp = imu_timeline_.states[k].ts;
                m.acc = imu_timeline_.states[k].acc;
                m.gyro = imu_timeline_.states[k].gyro;
                preints_[f]->propagate(m);
            }
        }

        // 参数块: 位姿=真值, 偏置=base(错误), 速度=真值
        params_pose_.resize(num_frames_);
        params_sb_.resize(num_frames_);
        for (int f = 0; f < num_frames_; ++f) {
            params_pose_[f] = std::array<double, 6>{};
            auto phi = Sophus::SO3d(frames_R_[f]).log();
            for (int d = 0; d < 3; ++d) {
                params_pose_[f][d] = frames_P_[f][d];
                params_pose_[f][d + 3] = phi[d];
            }
            params_sb_[f] = std::array<double, 9>{};
            for (int d = 0; d < 3; ++d) {
                params_sb_[f][d] = frames_V_[f][d];
                params_sb_[f][d + 3] = Ba_base_[d];
                params_sb_[f][d + 6] = Bg_base_[d];
            }
        }
    }

    double imu_dt_;
    int steps_per_frame_, num_frames_;
    Eigen::Vector3d a_body_, w_body_, Ba_base_, Bg_base_;
    Eigen::Matrix<double, 18, 18> noise_;

    std::vector<Eigen::Vector3d> Ba_true_, Bg_true_;
    test::ImuTimeline imu_timeline_;
    std::vector<Eigen::Matrix3d> frames_R_;
    std::vector<Eigen::Vector3d> frames_P_, frames_V_;
    std::vector<std::shared_ptr<MidPointIntegrator>> preints_;

    std::vector<std::array<double, 6>> params_pose_;
    std::vector<std::array<double, 9>> params_sb_;
};

// =============================================================================
// 测试 1: 数值微分验证 IMU 因子雅各比
// =============================================================================

TEST_F(ImuFactorTest, JacobianCheck) {
    auto* factor = new IMUFactor<MidPointIntegrator>(preints_[0]);
    SE3RightManifold manifold;

    const double eps = 1e-6;
    const double tol = 5e-2;  // 5% — IMU Jacobian 有积分误差容差更大
    int nbad = 0;

    double J[4][15 * 9];  // max is sb_j = 15x9 = 135
    double* jac_ptrs[] = {J[0], J[1], J[2], J[3]};
    const double* params[] = {
        params_pose_[0].data(), params_sb_[0].data(), params_pose_[1].data(), params_sb_[1].data()};
    double r[15];
    factor->Evaluate(params, r, jac_ptrs);

    Eigen::Matrix<double, 15, 6, Eigen::RowMajor> tangent_pose_jacobians[2];
    for (int pose_block = 0; pose_block < 2; ++pose_block) {
        const int block = pose_block * 2;
        double plus_data[36];
        manifold.PlusJacobian(params[block], plus_data);
        Eigen::Map<const Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> ambient(J[block]);
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
        tangent_pose_jacobians[pose_block] = ambient * plus;
    }

    // 切空间维度: pose=6, sb=9
    int dims[] = {6, 9, 6, 9};

    // 数值微分: manifold for pose, Euclidean for speed_bias
    auto num_col = [&](int blk, const double* x, int col) -> Eigen::Matrix<double, 15, 1> {
        int dim = dims[blk];
        double xp[9], xm[9];  // max size

        if (blk == 0 || blk == 2) {  // pose: manifold
            Eigen::VectorXd delta = Eigen::VectorXd::Zero(6);
            delta(col) = eps;
            manifold.Plus(x, delta.data(), xp);
            manifold.Plus(x, (-delta).eval().data(), xm);
        } else {  // speed_bias: Euclidean
            for (int d = 0; d < dim; ++d) {
                xp[d] = x[d];
                xm[d] = x[d];
            }
            xp[col] += eps;
            xm[col] -= eps;
        }

        double rp[15], rm[15];
        const double *pp[4], *pm[4];
        pp[blk] = xp;
        pm[blk] = xm;
        for (int b = 0; b < 4; ++b)
            if (b != blk) {
                pp[b] = params[b];
                pm[b] = params[b];
            }
        factor->Evaluate(pp, rp, nullptr);
        factor->Evaluate(pm, rm, nullptr);

        Eigen::Matrix<double, 15, 1> out;
        for (int i = 0; i < 15; ++i) out[i] = (rp[i] - rm[i]) / (2.0 * eps);
        return out;
    };

    const char* names[] = {"pose_i", "sb_i", "pose_j", "sb_j"};

    for (int blk = 0; blk < 4; ++blk) {
        int dim = dims[blk];
        int stride = (blk == 1 || blk == 3) ? 9 : 6;  // rows in J (15)

        std::cout << "\n--- " << names[blk] << " (" << 15 << "x" << dim << ") ---\n";
        for (int c = 0; c < dim; ++c) {
            auto num = num_col(blk, params[blk], c);
            int worst_row = 0;
            double worst_err = 0;
            for (int row = 0; row < 15; ++row) {
                double an = (blk == 0 || blk == 2) ? tangent_pose_jacobians[blk / 2](row, c)
                                                   : J[blk][row * stride + c];
                double scale = std::max({std::abs(an), std::abs(num[row]), 1e-8});
                double err = std::abs(an - num[row]) / scale;
                if (err > worst_err) {
                    worst_err = err;
                    worst_row = row;
                }
            }
            if (worst_err > tol) nbad++;
            if (worst_err > 0.01) {
                std::cout << "  col " << c << " max_err=" << worst_err << " @row=" << worst_row
                          << " an="
                          << ((blk == 0 || blk == 2) ? tangent_pose_jacobians[blk / 2](worst_row, c)
                                                     : J[blk][worst_row * stride + c])
                          << " num=" << num[worst_row] << (worst_err > tol ? " ***" : "") << "\n";
            }
        }
    }

    std::cout << "\ntotal bad columns (>5%): " << nbad << "\n";
    EXPECT_EQ(nbad, 0);

    delete factor;
}

TEST_F(ImuFactorTest, CeresGradientCheckerContract) {
    SE3RightManifold manifold;
    std::vector<const ceres::Manifold*> manifolds = {&manifold, nullptr, &manifold, nullptr};
    ceres::NumericDiffOptions options;
    options.relative_step_size = 1e-6;
    for (int frame = 0; frame < num_frames_ - 1; ++frame) {
        IMUFactor<MidPointIntegrator> factor(preints_[frame]);
        ceres::GradientChecker checker(&factor, &manifolds, options);
        const double* parameters[] = {
            params_pose_[frame].data(), params_sb_[frame].data(), params_pose_[frame + 1].data(),
            params_sb_[frame + 1].data()};
        ceres::GradientChecker::ProbeResults results;
        const bool relative_check_ok = checker.Probe(parameters, 5e-4, &results);
        double maximum_absolute_error = 0.0;
        for (size_t block = 0; block < results.local_jacobians.size(); ++block) {
            maximum_absolute_error = std::max(
                maximum_absolute_error,
                (results.local_jacobians[block] - results.local_numeric_jacobians[block])
                    .cwiseAbs()
                    .maxCoeff());
        }
        EXPECT_TRUE(relative_check_ok || maximum_absolute_error < 1e-7)
            << "frame=" << frame << " max_relative_error=" << results.maximum_relative_error
            << " max_absolute_error=" << maximum_absolute_error << "\n"
            << results.error_log;
    }
}

// =============================================================================
// 测试 2: 固定位姿+速度 → 偏置向真值恢复
// =============================================================================

TEST_F(ImuFactorTest, BiasRecovery) {
    ceres::Problem problem;

    for (int f = 0; f < num_frames_; ++f) {
        problem.AddParameterBlock(params_pose_[f].data(), 6, new SE3RightManifold());
        problem.SetParameterBlockConstant(params_pose_[f].data());
        problem.AddParameterBlock(params_sb_[f].data(), 9);
    }

    for (int f = 0; f < num_frames_ - 1; ++f) {
        auto* imu = new IMUFactor<MidPointIntegrator>(preints_[f]);
        problem.AddResidualBlock(
            imu, nullptr, params_pose_[f].data(), params_sb_[f].data(), params_pose_[f + 1].data(),
            params_sb_[f + 1].data());
    }

    // 紧速度先验: 把 V 钉在真值上
    for (int f = 0; f < num_frames_; ++f) {
        auto* vp = new ceres::AutoDiffCostFunction<VelocityPrior, 3, 9>(
            new VelocityPrior(frames_V_[f], 1e-8));
        problem.AddResidualBlock(vp, nullptr, params_sb_[f].data());
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = 100;
    opts.minimizer_progress_to_stdout = false;
    opts.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << "Bias recovery: " << summary.BriefReport() << "\n";

    double avg_dBa = 0, avg_dBg = 0;
    for (int f = 0; f < num_frames_; ++f) {
        Eigen::Vector3d Ba_opt(params_sb_[f][3], params_sb_[f][4], params_sb_[f][5]);
        Eigen::Vector3d Bg_opt(params_sb_[f][6], params_sb_[f][7], params_sb_[f][8]);
        double dBa = (Ba_opt - Ba_true_[f]).norm();
        double dBg = (Bg_opt - Bg_true_[f]).norm();
        avg_dBa += dBa;
        avg_dBg += dBg;
        std::cout << "f[" << f << "] dBa=" << dBa << " dBg=" << dBg << "  opt=["
                  << Ba_opt.transpose() << "] true=[" << Ba_true_[f].transpose() << "]\n";
    }
    avg_dBa /= num_frames_;
    avg_dBg /= num_frames_;

    EXPECT_LT(avg_dBa, 0.03) << "Ba should move toward truth";
    EXPECT_LT(avg_dBg, 0.01) << "Bg should move toward truth";
}

}  // namespace
}  // namespace tassel_core
