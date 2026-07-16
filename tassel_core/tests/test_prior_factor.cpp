// =============================================================================
// test_prior_factor.cpp
//
// Purpose:
//   验证平方根边缘化输出和 MarginalizationPriorFactor 的残差/雅各比一致性。
//
// Test design:
//   对 MargHelper::marginalizeSqrtToSqrt 使用 Eigen HouseholderQR 构造参考结果;
//   对 prior factor 使用人工线性化数据, 分别检查残差计算、参数块布局和 manifold
//   下的数值雅各比。
//
// Pass criteria:
//   消元后的 sqrt_H/sqrt_b 与 QR 参考一致, prior factor residual 与线性模型一致,
//   解析雅各比通过数值微分检查。
// =============================================================================

#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <random>

#include "factor/marginalization_prior_factor.h"
#include "marg/marg_helper.h"
#include "marg/marg_lin_data.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

constexpr double kJacobianRelTol = 1e-6;

// ── Helper: HouseholderQR reference implementation ────────────────────────

void computeExpected(
    const Eigen::MatrixXd& Q2Jp_orig, const Eigen::VectorXd& Q2r_orig, Eigen::MatrixXd& expected_H,
    Eigen::VectorXd& expected_b, size_t marg_size = 1) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Q2Jp_orig);
    Eigen::MatrixXd R = qr.matrixQR().template triangularView<Eigen::Upper>();
    Eigen::VectorXd QTr = qr.householderQ().transpose() * Q2r_orig;

    const Eigen::Index ms = static_cast<Eigen::Index>(marg_size);
    const Eigen::Index e_rows = R.rows() - ms;
    const Eigen::Index e_cols = R.cols() - ms;
    if (e_rows > 0 && e_cols > 0) {
        expected_H = R.block(ms, ms, e_rows, e_cols);
        expected_b = QTr.segment(ms, e_rows);
    } else if (e_rows > 0) {
        expected_H.resize(e_rows, 0);
        expected_b = QTr.segment(ms, e_rows);
    } else {
        expected_H.resize(0, 0);
        expected_b.resize(0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MargHelper::marginalizeSqrtToSqrt tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(MarginalizeSqrtToSqrtTest, FullRank3x4) {
    Eigen::MatrixXd Q2Jp(3, 4);
    Q2Jp << 2.0, 1.0, 3.0, 4.0, 4.0, 5.0, 9.0, 2.0, 7.0, 8.0, 6.0, 1.0;
    Eigen::VectorXd Q2r(3);
    Q2r << 1.0, 2.0, 3.0;

    const size_t keep_size = 3;
    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    Eigen::VectorXd Q2r_orig = Q2r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeSqrtToSqrt(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    // Inputs are consumed
    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2Jp.cols(), 0);
    EXPECT_EQ(Q2r.rows(), 0);

    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
    EXPECT_EQ(marg_sqrt_H.cols(), static_cast<Eigen::Index>(keep_size));

    // Upper-triangular check
    for (Eigen::Index r = 0; r < marg_sqrt_H.rows(); ++r) {
        for (Eigen::Index c = 0; c < std::min(r, marg_sqrt_H.cols()); ++c) {
            EXPECT_NEAR(marg_sqrt_H(r, c), 0.0, 1e-14)
                << "marg_sqrt_H(" << r << "," << c << ") should be zero";
        }
    }

    Eigen::MatrixXd expected_H;
    Eigen::VectorXd expected_b;
    computeExpected(Q2Jp_orig, Q2r_orig, expected_H, expected_b);

    ASSERT_EQ(marg_sqrt_H.rows(), expected_H.rows());
    ASSERT_EQ(marg_sqrt_H.cols(), expected_H.cols());
    ASSERT_EQ(marg_sqrt_b.rows(), expected_b.rows());

    EXPECT_TRUE(
        marg_sqrt_H.isApprox(expected_H, 1e-12) || marg_sqrt_H.isApprox(-expected_H, 1e-12));
    EXPECT_TRUE(
        marg_sqrt_b.isApprox(expected_b, 1e-12) || marg_sqrt_b.isApprox(-expected_b, 1e-12));
}

TEST(MarginalizeSqrtToSqrtTest, ShortWide2x3) {
    Eigen::MatrixXd Q2Jp(2, 3);
    Q2Jp << 3.0, 1.0, 4.0, 0.0, 2.0, 5.0;
    Eigen::VectorXd Q2r(2);
    Q2r << 1.0, 2.0;

    const size_t keep_size = 2;
    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    Eigen::VectorXd Q2r_orig = Q2r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeSqrtToSqrt(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_GT(marg_sqrt_H.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());

    Eigen::MatrixXd expected_H;
    Eigen::VectorXd expected_b;
    computeExpected(Q2Jp_orig, Q2r_orig, expected_H, expected_b);

    EXPECT_TRUE(
        marg_sqrt_H.isApprox(expected_H, 1e-12) || marg_sqrt_H.isApprox(-expected_H, 1e-12));
    EXPECT_TRUE(
        marg_sqrt_b.isApprox(expected_b, 1e-12) || marg_sqrt_b.isApprox(-expected_b, 1e-12));
}

TEST(MarginalizeSqrtToSqrtTest, RankDeficient4x5) {
    Eigen::MatrixXd Q2Jp(4, 5);
    Q2Jp << 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 1.0, 0.0, 1.0,
        1.0, 0.0, -1.0;
    Eigen::VectorXd Q2r(4);
    Q2r << 0.1, 0.2, 0.3, 0.4;

    const size_t keep_size = 4;
    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeSqrtToSqrt(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);
    EXPECT_LE(marg_sqrt_H.rows(), 2);  // rank-limited
    EXPECT_GE(marg_sqrt_H.rows(), 1);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
}

TEST(MarginalizeSqrtToSqrtTest, EmptyInput) {
    Eigen::MatrixXd Q2Jp(0, 4);
    Eigen::VectorXd Q2r(0);

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeSqrtToSqrt(1, 3, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(marg_sqrt_H.rows(), 0);
    EXPECT_EQ(marg_sqrt_b.rows(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// MarginalizationPriorFactor cost function tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(MarginalizationPriorTest, ResidualMatchesDirect) {
    std::mt19937 rng(42);
    std::normal_distribution<double> n(0.0, 1.0);

    // H: 6 × 12 (2 frames), b: 6 × 1
    Eigen::MatrixXd H(6, 12);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 12; ++c) H(r, c) = n(rng);
    Eigen::VectorXd b = Eigen::VectorXd::Random(6);

    std::array<double, 6> lin0 = {0, 0, 0, 0, 0, 0};
    std::array<double, 6> lin1 = {0, 0, 0, 0, 0, 0};

    MargLinData data;
    data.H = H;
    data.b = b;
    data.linearization_poses = {lin0, lin1};
    MarginalizationPriorFactor factor(data);

    // Random optimized pose
    double pose0[6], pose1[6];
    for (int i = 0; i < 6; ++i) {
        pose0[i] = n(rng);
        pose1[i] = n(rng);
    }

    // Expected residual
    Eigen::VectorXd x_opt(12);
    x_opt << pose0[0], pose0[1], pose0[2], pose0[3], pose0[4], pose0[5], pose1[0], pose1[1],
        pose1[2], pose1[3], pose1[4], pose1[5];
    Eigen::VectorXd expected_r = H * x_opt + b;

    double const* params[] = {pose0, pose1};
    Eigen::VectorXd r(factor.num_residuals());
    factor.Evaluate(params, r.data(), nullptr);

    EXPECT_EQ(r.size(), 6);
    EXPECT_TRUE(r.isApprox(expected_r, 1e-12));
}

TEST(MarginalizationPriorTest, JacobiansMatchManifoldFiniteDiff) {
    std::mt19937 rng(123);
    std::normal_distribution<double> n(0.0, 1.0);

    Eigen::MatrixXd H(8, 12);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 12; ++c) H(r, c) = n(rng);
    Eigen::VectorXd b = Eigen::VectorXd::Random(8);

    std::array<double, 6> lin0 = {0.2, -0.1, 0.3, 0.4, -0.2, 0.1};
    std::array<double, 6> lin1 = {-0.3, 0.2, 0.1, -0.25, 0.35, 0.15};

    MargLinData data;
    data.H = H;
    data.b = b;
    data.linearization_poses = {lin0, lin1};
    MarginalizationPriorFactor factor(data);

    double pose0[6], pose1[6];
    for (int i = 0; i < 6; ++i) {
        pose0[i] = lin0[i] + 0.2 * n(rng);
        pose1[i] = lin1[i] + 0.2 * n(rng);
    }

    double const* params[] = {pose0, pose1};
    int res_dim = factor.num_residuals();

    Eigen::VectorXd r0(res_dim);
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J0(res_dim, 6);
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J1(res_dim, 6);
    {
        double* jacs[] = {J0.data(), J1.data()};
        factor.Evaluate(params, r0.data(), jacs);
    }

    const double eps = 1e-6;
    SE3RightManifold manifold;
    for (int p = 0; p < 2; ++p) {
        double* pose = (p == 0) ? pose0 : pose1;
        auto& J = (p == 0) ? J0 : J1;

        double plus_data[36];
        manifold.PlusJacobian(pose, plus_data);
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
        Eigen::MatrixXd J_tangent = J * plus;

        for (int c = 0; c < 6; ++c) {
            Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
            delta[c] = eps;
            double pose_plus[6], pose_minus[6];
            manifold.Plus(pose, delta.data(), pose_plus);
            manifold.Plus(pose, (-delta).eval().data(), pose_minus);
            const double* original = params[p];
            params[p] = pose_plus;
            Eigen::VectorXd rp(res_dim);
            factor.Evaluate(params, rp.data(), nullptr);
            params[p] = pose_minus;
            Eigen::VectorXd rm(res_dim);
            factor.Evaluate(params, rm.data(), nullptr);
            params[p] = original;

            Eigen::VectorXd fd = (rp - rm) / (2.0 * eps);
            for (int row = 0; row < res_dim; ++row) {
                double denom = std::max(1e-6, std::abs(J_tangent(row, c)));
                EXPECT_NEAR(J_tangent(row, c), fd(row), 5e-5 * denom)
                    << "Block " << p << " col " << c << " row " << row;
            }
        }
    }
}

TEST(MarginalizationPriorTest, RebaseMatchesOldPriorAtCurrentState) {
    constexpr int kFrames = 2;
    constexpr int kCols = kFrames * 15 + 1;
    constexpr int kRows = 24;
    std::mt19937 rng(789);
    std::normal_distribution<double> n(0.0, 1.0);

    MargLinData old_prior;
    old_prior.H.resize(kRows, kCols);
    old_prior.b.resize(kRows);
    for (int r = 0; r < kRows; ++r) {
        old_prior.b[r] = 0.1 * n(rng);
        for (int c = 0; c < kCols; ++c) old_prior.H(r, c) = n(rng);
    }
    old_prior.linearization_poses = {
        std::array<double, 6>{0.1, -0.2, 0.3, 0.35, -0.15, 0.2},
        std::array<double, 6>{-0.3, 0.15, 0.2, -0.2, 0.3, 0.1}};
    old_prior.linearization_speed_bias.resize(kFrames);
    for (int i = 0; i < kFrames; ++i)
        for (double& value : old_prior.linearization_speed_bias[i]) value = 0.1 * n(rng);
    old_prior.linearization_delay_time = 0.004;

    auto current_poses = old_prior.linearization_poses;
    auto current_speed_bias = old_prior.linearization_speed_bias;
    for (int i = 0; i < kFrames; ++i) {
        for (double& value : current_poses[i]) value += 0.15 * n(rng);
        for (double& value : current_speed_bias[i]) value += 0.05 * n(rng);
    }
    double current_delay = -0.003;

    const MargLinData rebased =
        MargHelper::rebasePrior(old_prior, current_poses, current_speed_bias, current_delay);
    MarginalizationPriorFactor old_factor(old_prior);
    std::vector<const double*> parameters;
    for (int i = 0; i < kFrames; ++i) {
        parameters.push_back(current_poses[i].data());
        parameters.push_back(current_speed_bias[i].data());
    }
    parameters.push_back(&current_delay);

    Eigen::VectorXd residual(kRows);
    ASSERT_TRUE(old_factor.Evaluate(parameters.data(), residual.data(), nullptr));
    EXPECT_TRUE(rebased.b.isApprox(residual, 1e-12));

    const double eps = 1e-7;
    SE3RightManifold manifold;
    Eigen::MatrixXd numerical(kRows, kCols);
    for (int frame = 0; frame < kFrames; ++frame) {
        for (int c = 0; c < 6; ++c) {
            Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
            delta[c] = eps;
            double plus[6], minus[6];
            manifold.Plus(current_poses[frame].data(), delta.data(), plus);
            manifold.Plus(current_poses[frame].data(), (-delta).eval().data(), minus);
            const int parameter_index = frame * 2;
            const double* original = parameters[parameter_index];
            parameters[parameter_index] = plus;
            Eigen::VectorXd rp(kRows);
            old_factor.Evaluate(parameters.data(), rp.data(), nullptr);
            parameters[parameter_index] = minus;
            Eigen::VectorXd rm(kRows);
            old_factor.Evaluate(parameters.data(), rm.data(), nullptr);
            parameters[parameter_index] = original;
            numerical.col(frame * 15 + c) = (rp - rm) / (2.0 * eps);
        }
        for (int c = 0; c < 9; ++c) {
            const int parameter_index = frame * 2 + 1;
            const double original = current_speed_bias[frame][c];
            current_speed_bias[frame][c] = original + eps;
            Eigen::VectorXd rp(kRows);
            old_factor.Evaluate(parameters.data(), rp.data(), nullptr);
            current_speed_bias[frame][c] = original - eps;
            Eigen::VectorXd rm(kRows);
            old_factor.Evaluate(parameters.data(), rm.data(), nullptr);
            current_speed_bias[frame][c] = original;
            numerical.col(frame * 15 + 6 + c) = (rp - rm) / (2.0 * eps);
        }
    }
    current_delay += eps;
    Eigen::VectorXd rp(kRows);
    old_factor.Evaluate(parameters.data(), rp.data(), nullptr);
    current_delay -= 2.0 * eps;
    Eigen::VectorXd rm(kRows);
    old_factor.Evaluate(parameters.data(), rm.data(), nullptr);
    current_delay += eps;
    numerical.col(kCols - 1) = (rp - rm) / (2.0 * eps);

    EXPECT_TRUE(rebased.H.isApprox(numerical, 2e-7))
        << "max error: " << (rebased.H - numerical).cwiseAbs().maxCoeff();
}

TEST(SE3RightManifoldTest, PlusAndMinusJacobiansAreInverse) {
    SE3RightManifold manifold;
    double pose[6] = {0.3, -0.2, 0.1, 0.7, -0.4, 0.25};
    double plus_data[36], minus_data[36];
    ASSERT_TRUE(manifold.PlusJacobian(pose, plus_data));
    ASSERT_TRUE(manifold.MinusJacobian(pose, minus_data));
    Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
    Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(minus_data);
    EXPECT_TRUE((minus * plus).isApprox(Eigen::Matrix<double, 6, 6>::Identity(), 1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════
// Ceres optimization with MarginalizationPriorFactor
// ═══════════════════════════════════════════════════════════════════════════

TEST(MarginalizationPriorTest, CeresConvergesWithPrior) {
    std::mt19937 rng(99);
    std::normal_distribution<double> n(0.0, 1.0);

    // Ground truth: random x_gt. Prior residual: r = H*(x - x_lin) + b.
    // For optimum at x = x_gt with x_lin = 0, we need b = -H * x_gt.
    Eigen::VectorXd x_gt(12);
    for (int i = 0; i < 12; ++i) x_gt(i) = n(rng);

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(12, 12);
    Eigen::VectorXd b = -H * x_gt;

    std::array<double, 6> lin0, lin1;
    for (int i = 0; i < 6; ++i) {
        lin0[i] = 0.0;
        lin1[i] = 0.0;
    }

    MargLinData marg_data;
    marg_data.H = H;
    marg_data.b = b;
    marg_data.linearization_poses = {lin0, lin1};
    auto* factor = new MarginalizationPriorFactor(marg_data);

    // 扰动的初值
    double pose0[6], pose1[6];
    {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> p0(pose0);
        p0 = x_gt.segment<6>(0) + Eigen::VectorXd::Random(6) * 0.5;
    }
    {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> p1(pose1);
        p1 = x_gt.segment<6>(6) + Eigen::VectorXd::Random(6) * 0.5;
    }

    ceres::Problem problem;
    problem.AddParameterBlock(pose0, 6, new SE3RightManifold());
    problem.AddParameterBlock(pose1, 6, new SE3RightManifold());
    problem.AddResidualBlock(factor, nullptr, pose0, pose1);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = false;
    opts.max_num_iterations = 20;
    opts.function_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    EXPECT_EQ(summary.termination_type, ceres::CONVERGENCE);
    EXPECT_NEAR(pose0[0], x_gt(0), 1e-6);
    EXPECT_NEAR(pose0[1], x_gt(1), 1e-6);
    EXPECT_NEAR(pose1[0], x_gt(6), 1e-6);
    EXPECT_NEAR(pose1[1], x_gt(7), 1e-6);
}

}  // namespace
}  // namespace tassel_core
