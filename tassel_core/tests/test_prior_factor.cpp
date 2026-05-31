#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <random>

#include "factor/marg_helper.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/se3_right_manifold.h"

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

    MarginalizationPriorFactor factor(H, b, {lin0, lin1}, {});

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

TEST(MarginalizationPriorTest, JacobiansMatchFiniteDiff) {
    std::mt19937 rng(123);
    std::normal_distribution<double> n(0.0, 1.0);

    Eigen::MatrixXd H(8, 12);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 12; ++c) H(r, c) = n(rng);
    Eigen::VectorXd b = Eigen::VectorXd::Random(8);

    std::array<double, 6> lin0 = {0, 0, 0, 0, 0, 0};
    std::array<double, 6> lin1 = {0, 0, 0, 0, 0, 0};

    MarginalizationPriorFactor factor(H, b, {lin0, lin1}, {});

    double pose0[6], pose1[6];
    for (int i = 0; i < 6; ++i) {
        pose0[i] = n(rng);
        pose1[i] = n(rng);
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
    for (int p = 0; p < 2; ++p) {
        double* pose = (p == 0) ? pose0 : pose1;
        auto& J = (p == 0) ? J0 : J1;

        for (int c = 0; c < 6; ++c) {
            double orig = pose[c];
            pose[c] = orig + eps;
            Eigen::VectorXd rp(res_dim);
            factor.Evaluate(params, rp.data(), nullptr);
            pose[c] = orig - eps;
            Eigen::VectorXd rm(res_dim);
            factor.Evaluate(params, rm.data(), nullptr);
            pose[c] = orig;

            Eigen::VectorXd fd = (rp - rm) / (2.0 * eps);
            for (int row = 0; row < res_dim; ++row) {
                double denom = std::max(1e-6, std::abs(J(row, c)));
                EXPECT_NEAR(J(row, c), fd(row), 5e-5 * denom)
                    << "Block " << p << " col " << c << " row " << row;
            }
        }
    }
}

TEST(MarginalizationPriorTest, JacobianIsConstantH) {
    std::mt19937 rng(456);
    std::normal_distribution<double> n(0.0, 1.0);

    Eigen::MatrixXd H(6, 6);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) H(r, c) = n(rng);
    Eigen::VectorXd b = Eigen::VectorXd::Random(6);

    std::array<double, 6> lin0 = {0, 0, 0, 0, 0, 0};
    MarginalizationPriorFactor factor(H, b, {lin0}, {});

    // Jacobian should equal H regardless of evaluation point
    for (int trial = 0; trial < 3; ++trial) {
        double pose0[6];
        for (int i = 0; i < 6; ++i) pose0[i] = n(rng);

        double const* params[] = {pose0};
        Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J(6, 6);
        double* jacs[] = {J.data()};
        Eigen::VectorXd r(6);
        factor.Evaluate(params, r.data(), jacs);

        EXPECT_TRUE(J.isApprox(H, 1e-12)) << "Jacobian should equal H (trial " << trial << ")";
    }
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

    auto* factor = new MarginalizationPriorFactor(H, b, {lin0, lin1}, {});

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
