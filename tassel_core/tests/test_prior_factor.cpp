// =============================================================================
// test_prior_factor.cpp
//
// 目的：
//   验证平方根边缘化输出和 MarginalizationPriorFactor 的残差/雅各比一致性。
//
// 测试设计：
//   对 MargHelper::marginalizeSquareRootSystem 使用 Eigen HouseholderQR 构造参考结果;
//   对 prior factor 使用人工线性化数据, 分别检查残差计算、参数块布局和 manifold
//   下的数值雅各比。
//
// 通过条件：
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

// ── 辅助函数：HouseholderQR 参考实现 ──────────────────────────────────────

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
// MargHelper::marginalizeSquareRootSystem 测试
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
    MargHelper::marginalizeSquareRootSystem(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    // 输入矩阵会被消耗
    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2Jp.cols(), 0);
    EXPECT_EQ(Q2r.rows(), 0);

    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
    EXPECT_EQ(marg_sqrt_H.cols(), static_cast<Eigen::Index>(keep_size));

    // 检查上三角结构
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
    MargHelper::marginalizeSquareRootSystem(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

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
    Eigen::MatrixXd jacobian(4, 5);
    jacobian << 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 1.0, 0.0, 1.0,
        1.0, 0.0, -1.0;
    Eigen::VectorXd residual(4);
    residual << 0.1, 0.2, 0.3, 0.4;

    Eigen::Index reference_rank = -1;
    for (const double scale : {1e-12, 1.0, 1e12}) {
        Eigen::MatrixXd scaled_jacobian = scale * jacobian;
        Eigen::VectorXd scaled_residual = scale * residual;
        Eigen::MatrixXd prior_jacobian;
        Eigen::VectorXd prior_residual;
        MargHelper::marginalizeSquareRootSystem(
            1, 4, scaled_jacobian, scaled_residual, prior_jacobian, prior_residual);

        EXPECT_LE(prior_jacobian.rows(), 2);  // rank-limited
        EXPECT_GE(prior_jacobian.rows(), 1);
        EXPECT_EQ(prior_jacobian.rows(), prior_residual.rows());
        if (reference_rank < 0) {
            reference_rank = prior_jacobian.rows();
        }
        EXPECT_EQ(prior_jacobian.rows(), reference_rank);
    }
}

TEST(MarginalizeSqrtToSqrtTest, RankIsInvariantToSystemScale) {
    Eigen::MatrixXd jacobian(3, 3);
    jacobian << 2.0, 1.0, 4.0, 1.0, 3.0, 2.0, 0.5, -1.0, 3.0;
    Eigen::VectorXd residual(3);
    residual << 1.0, -2.0, 0.5;

    Eigen::Index reference_rank = -1;
    for (const double scale : {1e-12, 1.0, 1e12}) {
        Eigen::MatrixXd scaled_jacobian = scale * jacobian;
        Eigen::VectorXd scaled_residual = scale * residual;
        Eigen::MatrixXd prior_jacobian;
        Eigen::VectorXd prior_residual;
        MargHelper::marginalizeSquareRootSystem(
            1, 2, scaled_jacobian, scaled_residual, prior_jacobian, prior_residual);

        if (reference_rank < 0) {
            reference_rank = prior_jacobian.rows();
        }
        EXPECT_EQ(prior_jacobian.rows(), reference_rank);
    }
}

TEST(MarginalizeSqrtToSqrtTest, EmptyInput) {
    Eigen::MatrixXd Q2Jp(0, 4);
    Eigen::VectorXd Q2r(0);

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeSquareRootSystem(1, 3, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(marg_sqrt_H.rows(), 0);
    EXPECT_EQ(marg_sqrt_b.rows(), 0);
}

TEST(MarginalizeSqrtToSqrtTest, NoConstraintRemainsAfterMarginalization) {
    Eigen::MatrixXd Q2Jp(1, 2);
    Q2Jp << 2.0, 3.0;
    Eigen::VectorXd Q2r(1);
    Q2r << 1.0;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeSquareRootSystem(1, 1, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(marg_sqrt_H.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), 1);
    EXPECT_EQ(marg_sqrt_b.rows(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// MarginalizationPriorFactor 代价函数测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(MarginalizationPriorTest, ResidualMatchesDirect) {
    std::mt19937 rng(42);
    std::normal_distribution<double> n(0.0, 1.0);

    constexpr int kCols = 6 + 15 + 1;
    Eigen::MatrixXd H(6, kCols);
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < kCols; ++c) {
            H(r, c) = n(rng);
        }
    }
    Eigen::VectorXd b = Eigen::VectorXd::Random(6);

    std::array<double, 6> lin0 = {0, 0, 0, 0, 0, 0};
    std::array<double, 6> lin1 = {0, 0, 0, 0, 0, 0};

    MargLinData data;
    data.H = H;
    data.b = b;
    data.linearization_poses = {lin0, lin1};
    data.linearization_speed_bias.resize(2);
    MarginalizationPriorFactor factor(data);

    // 随机优化位姿
    double pose0[6], pose1[6];
    for (int i = 0; i < 6; ++i) {
        pose0[i] = n(rng);
        pose1[i] = n(rng);
    }

    // 预期残差
    double speed_bias1[9] = {};
    double delay = 0.0;
    Eigen::VectorXd x_opt = Eigen::VectorXd::Zero(kCols);
    x_opt << pose0[0], pose0[1], pose0[2], pose0[3], pose0[4], pose0[5], pose1[0], pose1[1],
        pose1[2], pose1[3], pose1[4], pose1[5], speed_bias1[0], speed_bias1[1], speed_bias1[2],
        speed_bias1[3], speed_bias1[4], speed_bias1[5], speed_bias1[6], speed_bias1[7],
        speed_bias1[8], delay;
    Eigen::VectorXd expected_r = H * x_opt + b;

    double const* params[] = {pose0, pose1, speed_bias1, &delay};
    Eigen::VectorXd r(factor.num_residuals());
    factor.Evaluate(params, r.data(), nullptr);

    EXPECT_EQ(r.size(), 6);
    EXPECT_TRUE(r.isApprox(expected_r, 1e-12));
}

TEST(MarginalizationPriorTest, JacobiansRemainFixedInLocalTangent) {
    std::mt19937 rng(123);
    std::normal_distribution<double> n(0.0, 1.0);

    constexpr int kCols = 6 + 15 + 1;
    Eigen::MatrixXd H(8, kCols);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < kCols; ++c) {
            H(r, c) = n(rng);
        }
    }
    Eigen::VectorXd b = Eigen::VectorXd::Random(8);

    std::array<double, 6> lin0 = {0.2, -0.1, 0.3, 0.4, -0.2, 0.1};
    std::array<double, 6> lin1 = {-0.3, 0.2, 0.1, -0.25, 0.35, 0.15};

    MargLinData data;
    data.H = H;
    data.b = b;
    data.linearization_poses = {lin0, lin1};
    data.linearization_speed_bias.resize(2);
    MarginalizationPriorFactor factor(data);

    double pose0[6], pose1[6];
    for (int i = 0; i < 6; ++i) {
        pose0[i] = lin0[i] + 0.2 * n(rng);
        pose1[i] = lin1[i] + 0.2 * n(rng);
    }

    double speed_bias1[9] = {};
    double delay = 0.0;
    double const* params[] = {pose0, pose1, speed_bias1, &delay};
    int res_dim = factor.num_residuals();

    Eigen::VectorXd r0(res_dim);
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J0(res_dim, 6);
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J1(res_dim, 6);
    {
        double* jacs[] = {J0.data(), J1.data(), nullptr, nullptr};
        factor.Evaluate(params, r0.data(), jacs);
    }

    SE3RightManifold manifold;
    for (int p = 0; p < 2; ++p) {
        double* pose = (p == 0) ? pose0 : pose1;
        auto& J = (p == 0) ? J0 : J1;

        double plus_data[36];
        manifold.PlusJacobian(pose, plus_data);
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
        const Eigen::MatrixXd J_tangent = J * plus;
        EXPECT_TRUE(J_tangent.isApprox(H.middleCols(p * 6, 6), 1e-12));
    }
}

TEST(MarginalizationPriorTest, CurrentResidualMatchesStoredPrior) {
    constexpr int kFrames = 2;
    constexpr int kCols = 6 + (kFrames - 1) * 15 + 1;
    constexpr int kRows = 24;
    std::mt19937 rng(789);
    std::normal_distribution<double> n(0.0, 1.0);

    MargLinData old_prior;
    old_prior.H.resize(kRows, kCols);
    old_prior.b.resize(kRows);
    for (int r = 0; r < kRows; ++r) {
        old_prior.b[r] = 0.1 * n(rng);
        for (int c = 0; c < kCols; ++c) {
            old_prior.H(r, c) = n(rng);
        }
    }
    old_prior.linearization_poses = {
        std::array<double, 6>{0.1, -0.2, 0.3, 0.35, -0.15, 0.2},
        std::array<double, 6>{-0.3, 0.15, 0.2, -0.2, 0.3, 0.1}};
    old_prior.linearization_speed_bias.resize(kFrames);
    for (int i = 0; i < kFrames; ++i) {
        for (double& value : old_prior.linearization_speed_bias[i]) {
            value = 0.1 * n(rng);
        }
    }
    old_prior.linearization_delay_time = 0.004;

    auto current_poses = old_prior.linearization_poses;
    auto current_speed_bias = old_prior.linearization_speed_bias;
    for (int i = 0; i < kFrames; ++i) {
        for (double& value : current_poses[i]) {
            value += 0.15 * n(rng);
        }
        for (double& value : current_speed_bias[i]) {
            value += 0.05 * n(rng);
        }
    }
    double current_delay = -0.003;

    const Eigen::VectorXd current_residual = MargHelper::evaluatePriorResidual(
        old_prior, current_poses, current_speed_bias, current_delay);
    MarginalizationPriorFactor old_factor(old_prior);
    std::vector<const double*> parameters;
    parameters.push_back(current_poses[0].data());
    for (int i = 1; i < kFrames; ++i) {
        parameters.push_back(current_poses[i].data());
        parameters.push_back(current_speed_bias[i].data());
    }
    parameters.push_back(&current_delay);

    Eigen::VectorXd residual(kRows);
    ASSERT_TRUE(old_factor.Evaluate(parameters.data(), residual.data(), nullptr));
    EXPECT_TRUE(current_residual.isApprox(residual, 1e-12));
}

TEST(MarginalizationPriorTest, RecenterPreservesResidualAndLocalJacobian) {
    constexpr int kFrames = 2;
    constexpr int kCols = 6 + (kFrames - 1) * 15 + 1;
    constexpr int kRows = 12;
    MargLinData old_prior;
    old_prior.H = Eigen::MatrixXd::Random(kRows, kCols);
    old_prior.b = Eigen::VectorXd::Random(kRows);
    old_prior.linearization_poses = {
        std::array<double, 6>{0.1, -0.2, 0.3, 0.2, -0.1, 0.15},
        std::array<double, 6>{0.4, 0.1, -0.2, -0.15, 0.25, 0.1}};
    old_prior.linearization_speed_bias = {
        std::array<double, 9>{0.1, 0.2, 0.3, 0.01, 0.02, 0.03, -0.01, 0.01, 0.02},
        std::array<double, 9>{-0.2, 0.1, 0.4, -0.02, 0.01, 0.04, 0.02, -0.03, 0.01}};
    old_prior.linearization_delay_time = 0.003;

    auto poses = old_prior.linearization_poses;
    auto speed_bias = old_prior.linearization_speed_bias;
    poses[0] = {0.25, -0.1, 0.28, 0.27, -0.04, 0.08};
    poses[1] = {0.5, 0.03, -0.1, -0.08, 0.31, 0.04};
    speed_bias[1][0] += 0.08;
    speed_bias[1][4] -= 0.02;
    const double delay = -0.002;

    const Eigen::VectorXd old_residual =
        MargHelper::evaluatePriorResidual(old_prior, poses, speed_bias, delay);
    MargLinData recentered = old_prior;
    MargHelper::recenterPrior(recentered, poses, speed_bias, delay);
    EXPECT_TRUE(recentered.b.isApprox(old_residual, 1e-12));

    constexpr double kEps = 1e-7;
    SE3RightManifold manifold;
    for (int frame = 0; frame < kFrames; ++frame) {
        const int col = frame == 0 ? 0 : 6;
        for (int axis = 0; axis < 6; ++axis) {
            double delta[6] = {};
            delta[axis] = kEps;
            auto perturbed_poses = poses;
            ASSERT_TRUE(manifold.Plus(poses[frame].data(), delta, perturbed_poses[frame].data()));
            const Eigen::VectorXd perturbed =
                MargHelper::evaluatePriorResidual(old_prior, perturbed_poses, speed_bias, delay);
            const Eigen::VectorXd numerical = (perturbed - old_residual) / kEps;
            EXPECT_TRUE(numerical.isApprox(recentered.H.col(col + axis), 2e-6));
        }
    }
}

TEST(MarginalizationPriorTest, GaugeTransformPreservesResidual) {
    constexpr int kFrames = 3;
    constexpr int kCols = 6 + (kFrames - 1) * 15 + 1;
    constexpr int kRows = 10;
    MargLinData prior;
    prior.H = Eigen::MatrixXd::Random(kRows, kCols);
    prior.b = Eigen::VectorXd::Random(kRows);
    prior.linearization_poses = {
        std::array<double, 6>{0.1, 0.2, -0.1, 0.2, -0.1, 0.05},
        std::array<double, 6>{0.4, -0.2, 0.3, -0.1, 0.3, 0.1},
        std::array<double, 6>{0.7, 0.1, 0.2, 0.15, 0.2, -0.1}};
    prior.linearization_speed_bias.resize(kFrames);
    for (int i = 0; i < kFrames; ++i) {
        prior.linearization_speed_bias[i] = {0.1 * i, -0.2,  0.3,  0.01, -0.02,
                                             0.03,    -0.01, 0.02, -0.03};
    }
    prior.linearization_delay_time = 0.004;
    auto poses = prior.linearization_poses;
    auto speed_bias = prior.linearization_speed_bias;
    for (int i = 0; i < kFrames; ++i) {
        poses[i][0] += 0.05;
        poses[i][2] -= 0.03;
        poses[i][4] += 0.02;
        speed_bias[i][0] += 0.04;
    }
    const double delay = -0.001;
    const Eigen::VectorXd original =
        MargHelper::evaluatePriorResidual(prior, poses, speed_bias, delay);

    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d translation(1.2, -0.4, 0.3);
    MargHelper::transformPriorGauge(prior, rotation, translation);
    for (int i = 0; i < kFrames; ++i) {
        const Eigen::Vector3d position(poses[i][0], poses[i][1], poses[i][2]);
        const Eigen::Vector3d phi(poses[i][3], poses[i][4], poses[i][5]);
        const Eigen::Vector3d velocity(speed_bias[i][0], speed_bias[i][1], speed_bias[i][2]);
        const Eigen::Vector3d transformed_position = rotation * position + translation;
        const Eigen::Vector3d transformed_phi =
            (Sophus::SO3d(rotation) * Sophus::SO3d::exp(phi)).log();
        const Eigen::Vector3d transformed_velocity = rotation * velocity;
        for (int d = 0; d < 3; ++d) {
            poses[i][d] = transformed_position[d];
            poses[i][3 + d] = transformed_phi[d];
            speed_bias[i][d] = transformed_velocity[d];
        }
    }
    const Eigen::VectorXd transformed =
        MargHelper::evaluatePriorResidual(prior, poses, speed_bias, delay);
    EXPECT_TRUE(transformed.isApprox(original, 1e-11));
}

TEST(MarginalizationPriorTest, GaugeTransformRejectsNonRotation) {
    MargLinData prior;
    prior.H = Eigen::MatrixXd::Identity(6, 6);
    prior.b = Eigen::VectorXd::Zero(6);
    prior.linearization_poses = {std::array<double, 6>{}};
    prior.linearization_speed_bias = {std::array<double, 9>{}};
    Eigen::Matrix3d scaling = Eigen::Matrix3d::Identity();
    scaling(0, 0) = 2.0;

    EXPECT_THROW(
        MargHelper::transformPriorGauge(prior, scaling, Eigen::Vector3d::Zero()), std::logic_error);
}

TEST(MarginalizationPriorTest, RecenterRejectsRotationLogSingularity) {
    MargLinData prior;
    prior.H = Eigen::MatrixXd::Identity(7, 7);
    prior.b = Eigen::VectorXd::Zero(7);
    prior.linearization_poses = {std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    prior.linearization_speed_bias = {std::array<double, 9>{}};
    auto poses = prior.linearization_poses;
    poses[0][3] = 3.14159265358979323846;

    EXPECT_THROW(
        MargHelper::recenterPrior(prior, poses, prior.linearization_speed_bias, 0.0),
        std::logic_error);
}

TEST(MarginalizationPriorTest, FixedWindowLayout) {
    constexpr int kFrames = 3;
    constexpr int kCols = 6 + (kFrames - 1) * 15 + 1;
    constexpr int kRows = 10;
    std::mt19937 rng(321);
    std::normal_distribution<double> n(0.0, 1.0);

    MargLinData data;
    data.H.resize(kRows, kCols);
    data.b.resize(kRows);
    for (int r = 0; r < kRows; ++r) {
        data.b[r] = 0.1 * n(rng);
        for (int c = 0; c < kCols; ++c) {
            data.H(r, c) = n(rng);
        }
    }
    data.linearization_poses.resize(kFrames);
    data.linearization_speed_bias.resize(kFrames);
    for (int i = 0; i < kFrames; ++i) {
        for (double& value : data.linearization_poses[i]) {
            value = 0.1 * n(rng);
        }
        for (double& value : data.linearization_speed_bias[i]) {
            value = 0.1 * n(rng);
        }
    }
    data.linearization_delay_time = 0.002;

    auto poses = data.linearization_poses;
    auto speed_bias = data.linearization_speed_bias;
    for (int i = 0; i < kFrames; ++i) {
        for (double& value : poses[i]) {
            value += 0.03 * n(rng);
        }
        for (double& value : speed_bias[i]) {
            value += 0.02 * n(rng);
        }
    }
    double delay_time = -0.004;

    MarginalizationPriorFactor factor(data);
    std::vector<const double*> parameters = {poses[0].data(),      poses[1].data(),
                                             speed_bias[1].data(), poses[2].data(),
                                             speed_bias[2].data(), &delay_time};

    Eigen::VectorXd residual(kRows);
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        jacobian_blocks;
    jacobian_blocks.emplace_back(kRows, 6);
    jacobian_blocks.emplace_back(kRows, 6);
    jacobian_blocks.emplace_back(kRows, 9);
    jacobian_blocks.emplace_back(kRows, 6);
    jacobian_blocks.emplace_back(kRows, 9);
    jacobian_blocks.emplace_back(kRows, 1);
    std::vector<double*> jacobians;
    for (auto& block : jacobian_blocks) {
        jacobians.push_back(block.data());
    }

    ASSERT_TRUE(factor.Evaluate(parameters.data(), residual.data(), jacobians.data()));
    const Eigen::VectorXd helper_residual =
        MargHelper::evaluatePriorResidual(data, poses, speed_bias, delay_time);
    EXPECT_TRUE(residual.isApprox(helper_residual, 1e-12));

    SE3RightManifold manifold;
    std::array<int, kFrames> pose_parameter_indices = {0, 1, 3};
    std::array<int, kFrames> pose_column_indices = {0, 6, 21};
    for (int i = 0; i < kFrames; ++i) {
        double plus_data[36];
        ASSERT_TRUE(manifold.PlusJacobian(parameters[pose_parameter_indices[i]], plus_data));
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> plus(plus_data);
        const Eigen::MatrixXd J_tangent = jacobian_blocks[pose_parameter_indices[i]] * plus;
        EXPECT_TRUE(J_tangent.isApprox(data.H.middleCols(pose_column_indices[i], 6), 1e-12));
    }
    EXPECT_TRUE(jacobian_blocks[2].isApprox(data.H.middleCols(12, 9), 1e-12));
    EXPECT_TRUE(jacobian_blocks[4].isApprox(data.H.middleCols(27, 9), 1e-12));
    EXPECT_TRUE(jacobian_blocks[5].isApprox(data.H.rightCols(1), 1e-12));
}

TEST(MarginalizationPriorTest, RejectsNonWindowPriorLayout) {
    MargLinData data;
    data.H = Eigen::MatrixXd::Zero(3, 12);
    data.b = Eigen::VectorXd::Zero(3);
    data.linearization_poses.resize(2);
    data.linearization_speed_bias.resize(2);

    EXPECT_THROW(MarginalizationPriorFactor factor(data), std::invalid_argument);
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
// 使用 MarginalizationPriorFactor 进行 Ceres 优化
// ═══════════════════════════════════════════════════════════════════════════

TEST(MarginalizationPriorTest, CeresConvergesWithPrior) {
    std::mt19937 rng(99);
    std::normal_distribution<double> n(0.0, 1.0);

    // FEJ 旋转雅各比有意不等于远离线性化点后的真实导数；Ceres 集成测试使用精确线性的平移方向。
    constexpr int kCols = 6 + 15 + 1;
    Eigen::VectorXd x_gt = Eigen::VectorXd::Zero(kCols);
    for (const int index : {0, 1, 2, 6, 7, 8}) {
        x_gt(index) = n(rng);
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(kCols, kCols);
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
    marg_data.linearization_speed_bias.resize(2);
    auto* factor = new MarginalizationPriorFactor(marg_data);

    double pose0[6] = {};
    double pose1[6] = {};
    double speed_bias1[9] = {};
    double delay = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        pose0[axis] = x_gt(axis) + 0.5 * n(rng);
        pose1[axis] = x_gt(6 + axis) + 0.5 * n(rng);
    }

    ceres::Problem problem;
    problem.AddParameterBlock(pose0, 6, new SE3RightManifold());
    problem.AddParameterBlock(pose1, 6, new SE3RightManifold());
    problem.AddResidualBlock(factor, nullptr, pose0, pose1, speed_bias1, &delay);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = false;
    opts.max_num_iterations = 20;
    opts.function_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    EXPECT_EQ(summary.termination_type, ceres::CONVERGENCE);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(pose0[axis], x_gt(axis), 1e-9);
        EXPECT_NEAR(pose1[axis], x_gt(6 + axis), 1e-9);
    }
}

}  // namespace
}  // namespace tassel_core
