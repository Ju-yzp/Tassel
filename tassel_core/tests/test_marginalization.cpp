#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <cmath>
#include <limits>

#include "marginalization/marg_helper.h"

namespace tassel_core {
namespace {

// ── helpers ─────────────────────────────────────────────────────────────────

// 用 Eigen HouseholderQR 提取"正确"的 marg_sqrt_H / marg_sqrt_b：
//   1. 对原始 Q2Jp 做 QR → R = Q^T * Q2Jp
//   2. Q2r 变换：QTr = Q^T * Q2r
//   3. 取 R(1..end, 1..end)  = marg_sqrt_H
//      取 QTr(1..end)         = marg_sqrt_b
//
// 注意：Eigen HouseholderQR 的 R 符号可能与被测函数不同（整体符号翻转等价）
void computeExpected(
    const Eigen::MatrixXd& Q2Jp_orig, const Eigen::VectorXd& Q2r_orig, Eigen::MatrixXd& expected_H,
    Eigen::VectorXd& expected_b) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Q2Jp_orig);
    Eigen::MatrixXd R = qr.matrixQR().template triangularView<Eigen::Upper>();
    Eigen::VectorXd QTr = qr.householderQ().transpose() * Q2r_orig;

    const Eigen::Index e_rows = R.rows() - 1;  // keep_valid_rows
    const Eigen::Index e_cols = R.cols() - 1;  // keep_size
    if (e_rows > 0 && e_cols > 0) {
        expected_H = R.block(1, 1, e_rows, e_cols);
        expected_b = QTr.segment(1, e_rows);
    } else if (e_rows > 0) {
        expected_H.resize(e_rows, 0);
        expected_b = QTr.segment(1, e_rows);
    } else {
        expected_H.resize(0, 0);
        expected_b.resize(0);
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

// 3×4 满秩矩阵（rows < cols，满足算法约束）
TEST(MarginalizeOldestTest, FullRank3x4) {
    MargHelper helper;
    Eigen::MatrixXd Q2Jp(3, 4);
    Q2Jp << 2.0, 1.0, 3.0, 4.0, 4.0, 5.0, 9.0, 2.0, 7.0, 8.0, 6.0, 1.0;
    Eigen::VectorXd Q2r(3);
    Q2r << 1.0, 2.0, 3.0;

    const size_t keep_size = 3;
    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    Eigen::VectorXd Q2r_orig = Q2r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    helper.marginalizeOldest(keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    // 输入被清空
    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2Jp.cols(), 0);
    EXPECT_EQ(Q2r.rows(), 0);

    // 输出维度
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
    EXPECT_EQ(marg_sqrt_H.cols(), static_cast<Eigen::Index>(keep_size));

    // marg_sqrt_H 应为上三角
    for (Eigen::Index r = 0; r < marg_sqrt_H.rows(); ++r) {
        for (Eigen::Index c = 0; c < std::min(r, marg_sqrt_H.cols()); ++c) {
            EXPECT_NEAR(marg_sqrt_H(r, c), 0.0, 1e-14)
                << "marg_sqrt_H(" << r << "," << c << ") should be zero (below diagonal)";
        }
    }

    // 与 Eigen QR 对比
    Eigen::MatrixXd expected_H;
    Eigen::VectorXd expected_b;
    computeExpected(Q2Jp_orig, Q2r_orig, expected_H, expected_b);

    ASSERT_EQ(marg_sqrt_H.rows(), expected_H.rows());
    ASSERT_EQ(marg_sqrt_H.cols(), expected_H.cols());
    ASSERT_EQ(marg_sqrt_b.rows(), expected_b.rows());

    EXPECT_TRUE(marg_sqrt_H.isApprox(expected_H, 1e-12) || marg_sqrt_H.isApprox(-expected_H, 1e-12))
        << "marg_sqrt_H:\n"
        << marg_sqrt_H << "\nexpected_H:\n"
        << expected_H;
    EXPECT_TRUE(marg_sqrt_b.isApprox(expected_b, 1e-12) || marg_sqrt_b.isApprox(-expected_b, 1e-12))
        << "marg_sqrt_b:\n"
        << marg_sqrt_b.transpose() << "\nexpected_b:\n"
        << expected_b.transpose();
}

// 2×3 短胖矩阵 — rows < cols 边界
TEST(MarginalizeOldestTest, ShortWide2x3) {
    MargHelper helper;
    Eigen::MatrixXd Q2Jp(2, 3);
    Q2Jp << 3.0, 1.0, 4.0, 0.0, 2.0, 5.0;
    Eigen::VectorXd Q2r(2);
    Q2r << 1.0, 2.0;

    const size_t keep_size = 2;
    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    Eigen::VectorXd Q2r_orig = Q2r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    helper.marginalizeOldest(keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
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

// 4×5 缺秩矩阵 — rank=2，验证缺秩处理
TEST(MarginalizeOldestTest, RankDeficient4x5) {
    MargHelper helper;
    Eigen::MatrixXd Q2Jp(4, 5);
    // col0 和 col1 独立；col2=col0+col1；col3=2*col0；col4=col0-col1
    Q2Jp << 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 1.0, 0.0, 1.0,
        1.0, 0.0, -1.0;
    Eigen::VectorXd Q2r(4);
    Q2r << 0.1, 0.2, 0.3, 0.4;

    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    const size_t keep_size = 4;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    helper.marginalizeOldest(keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);

    // rank=2, marg_rank=1 → keep_valid_rows ≤ 1
    EXPECT_LE(marg_sqrt_H.rows(), 2);
    EXPECT_GE(marg_sqrt_H.rows(), 1);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
}

// 第一列接近零 — marg_rank 应为 0
TEST(MarginalizeOldestTest, NearZeroFirstColumn) {
    MargHelper helper;
    const double eps = std::numeric_limits<double>::epsilon();
    Eigen::MatrixXd Q2Jp(3, 4);
    Q2Jp << eps * 0.01, 2.0, 3.0, 4.0,  // col 0 低于 rank 阈值
        eps * 0.01, 5.0, 6.0, 7.0, eps * 0.01, 8.0, 9.0, 1.0;
    Eigen::VectorXd Q2r(3);
    Q2r << 1.0, 2.0, 3.0;

    const size_t keep_size = 3;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    helper.marginalizeOldest(keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());

    // 第一列被丢弃，剩余 3 列应都贡献 rank → 最多 3 有效行
    EXPECT_LE(marg_sqrt_H.rows(), 3);
}

// 2×3 keep_size=2 — 最小可工作的非退化尺寸
TEST(MarginalizeOldestTest, Minimal2x3) {
    MargHelper helper;
    Eigen::MatrixXd Q2Jp(2, 3);
    Q2Jp << 1.0, 3.0, 5.0, 2.0, 4.0, 6.0;
    Eigen::VectorXd Q2r(2);
    Q2r << 0.5, 1.5;

    const size_t keep_size = 2;
    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    helper.marginalizeOldest(keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());

    // 上三角校验（marg_sqrt_H 最多 1 行，只有 1 个可能的对角下元素）
    if (marg_sqrt_H.rows() > 1) {
        for (Eigen::Index r = 0; r < marg_sqrt_H.rows(); ++r) {
            for (Eigen::Index c = 0; c < std::min(r, marg_sqrt_H.cols()); ++c) {
                EXPECT_NEAR(marg_sqrt_H(r, c), 0.0, 1e-14);
            }
        }
    }
}

// ── computeDelta ─────────────────────────────────────────────────────────────

std::shared_ptr<State> MakeTestState(int n, bool linearize) {
    auto s = std::make_shared<State>(n);
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix3d R =
            Eigen::AngleAxisd(i * 0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Vector3d t(i * 0.1, i * 0.02, 0);
        s->poses[i] = PoseStateWithLin(Sophus::SE3d(R, t));
    }
    if (linearize) {
        for (int i = 0; i < n - 1; ++i) {  // last frame not linearized
            s->poses[i].setLinearized();
        }
    }
    return s;
}

TEST(ComputeDeltaTest, AllZeroDeltas) {
    MargHelper helper;
    auto state = MakeTestState(4, true);  // 4 frames, frames 0-2 linearized, all deltas zero

    Eigen::VectorXd delta;
    helper.computeDelta(*state, delta);

    EXPECT_EQ(delta.size(), 4 * 6);
    EXPECT_TRUE(delta.isApproxToConstant(0, 1e-14));
}

TEST(ComputeDeltaTest, NonZeroDeltas) {
    MargHelper helper;
    auto state = MakeTestState(3, true);  // frames 0-1 linearized

    // Apply deltas to frames 0 and 1
    Eigen::Vector<double, 6> d0;
    d0 << 0.1, 0.2, 0.3, 0.01, 0.02, 0.03;
    state->poses[0].applyDelta(d0);
    Eigen::Vector<double, 6> d1;
    d1 << -0.1, 0.0, 0.5, -0.01, 0.0, 0.01;
    state->poses[1].applyDelta(d1);

    Eigen::VectorXd delta;
    helper.computeDelta(*state, delta);

    EXPECT_EQ(delta.size(), 3 * 6);
    // Frame 0: cols 0-5
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(delta(i), d0(i), 1e-14);
    // Frame 1: cols 6-11
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(delta(6 + i), d1(i), 1e-14);
    // Frame 2 (not linearized, last frame): cols 12-17 — skipped by computeDelta, stays zero
    for (int i = 12; i < 18; ++i) EXPECT_NEAR(delta(i), 0.0, 1e-14);
}

TEST(ComputeDeltaTest, OnlySkippedNewFrame) {
    MargHelper helper;
    auto state = MakeTestState(2, true);  // frame 0 linearized, frame 1 is new (not linearized)

    Eigen::VectorXd delta;
    helper.computeDelta(*state, delta);

    EXPECT_EQ(delta.size(), 2 * 6);
    // Frame 0: zero delta
    EXPECT_TRUE(delta.head(6).isApproxToConstant(0, 1e-14));
    // Frame 1: not touched (stays zero from setZero)
    EXPECT_TRUE(delta.tail(6).isApproxToConstant(0, 1e-14));
}

// ── linearizeMargPrior ───────────────────────────────────────────────────────

TEST(LinearizeMargPriorTest, IdentitySqrtH) {
    // H = I (6×6 for a 1-frame old_state), b = 0, delta = 0
    // Expected: abs_H += I^T*I = I, abs_b += I^T*(0 + I*0) = 0, error = 0
    MargHelper helper;

    int n = 2;                                // 1 old frame + 1 new frame
    auto old_state = MakeTestState(n, true);  // frame 0 linearized
    auto cur_state = MakeTestState(n, true);

    MargLinData mld;
    mld.old_state = *old_state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6);
    mld.b = Eigen::VectorXd::Zero(n * 6);

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Zero(n * 6, n * 6);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Zero(n * 6);
    double error = 999.0;

    helper.linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    EXPECT_TRUE(abs_H.isApprox(Eigen::MatrixXd::Identity(n * 6, n * 6), 1e-14));
    EXPECT_TRUE(abs_b.isApproxToConstant(0, 1e-14));
    EXPECT_NEAR(error, 0.0, 1e-14);
}

TEST(LinearizeMargPriorTest, NonZeroDeltaCompensation) {
    // H = 2*I (6×6), b = 0, delta_0 = [1,0,0,0,0,0]
    // Expected: abs_H += 4*I, abs_b += 2*I * (0 + 2*I * delta) = 4*delta
    //           error = delta^T * 2*I * (I*delta + 0) = 2*delta^T*delta = 2
    MargHelper helper;

    int n = 2;
    auto old_state = MakeTestState(n, true);
    auto cur_state = MakeTestState(n, true);

    // Set delta on old_state's frame 0
    Eigen::Vector<double, 6> d0;
    d0 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    old_state->poses[0].applyDelta(d0);

    MargLinData mld;
    mld.old_state = *old_state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6) * 2.0;
    mld.b = Eigen::VectorXd::Zero(n * 6);

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Zero(n * 6, n * 6);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Zero(n * 6);
    double error = 0.0;

    helper.linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    // abs_H: 4*I
    Eigen::MatrixXd expected_H = Eigen::MatrixXd::Identity(n * 6, n * 6) * 4.0;
    EXPECT_TRUE(abs_H.isApprox(expected_H, 1e-14));

    // abs_b: H^T * (b + H * delta) = 2*I * (0 + 2*I*delta) = 4*delta
    // delta = [1,0,0,0,0,0, 0,...,0], so abs_b = [4,0,0,0,0,0, 0,...,0]
    EXPECT_NEAR(abs_b(0), 4.0, 1e-14);
    for (int i = 1; i < n * 6; ++i) EXPECT_NEAR(abs_b(i), 0.0, 1e-14);

    // error = delta^T * H^T * (0.5 * H * delta + b) = delta^T * 2I * (I*delta) = 2
    EXPECT_NEAR(error, 2.0, 1e-14);
}

TEST(LinearizeMargPriorTest, NonZeroB) {
    // H = I (12×12), b = [1,2,...,12], delta = 0
    // abs_H += I, abs_b += I^T * (b + I*0) = b
    // error = 0^T * I * (0.5*I*0 + b) = 0
    MargHelper helper;

    int n = 2;
    auto old_state = MakeTestState(n, true);
    auto cur_state = MakeTestState(n, true);

    MargLinData mld;
    mld.old_state = *old_state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6);

    Eigen::VectorXd b_vec(n * 6);
    for (int i = 0; i < n * 6; ++i) b_vec(i) = static_cast<double>(i + 1);
    mld.b = b_vec;

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Zero(n * 6, n * 6);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Zero(n * 6);
    double error = 0.0;

    helper.linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    EXPECT_TRUE(abs_H.isApprox(Eigen::MatrixXd::Identity(n * 6, n * 6), 1e-14));
    EXPECT_TRUE(abs_b.isApprox(b_vec, 1e-14));
    EXPECT_NEAR(error, 0.0, 1e-14);
}

TEST(LinearizeMargPriorTest, Accumulation) {
    // Call twice — verify that abs_H and abs_b accumulate
    MargHelper helper;

    int n = 2;
    auto old_state = MakeTestState(n, true);
    auto cur_state = MakeTestState(n, true);

    MargLinData mld;
    mld.old_state = *old_state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6);
    mld.b = Eigen::VectorXd::Ones(n * 6);

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Ones(n * 6, n * 6);  // start non-zero
    Eigen::VectorXd abs_b = Eigen::VectorXd::Ones(n * 6);         // start non-zero
    double error = 0.0;

    helper.linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);
    // Second call with same data
    helper.linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    // abs_H: identity added twice + initial ones
    Eigen::MatrixXd expected_H = 3.0 * Eigen::MatrixXd::Identity(n * 6, n * 6);
    for (int r = 0; r < n * 6; ++r)
        for (int c = 0; c < n * 6; ++c)
            if (r != c) expected_H(r, c) = 1.0;
    EXPECT_TRUE(abs_H.isApprox(expected_H, 1e-14));

    // abs_b: added b_vec twice + initial ones = vector of 3's
    for (int i = 0; i < n * 6; ++i) EXPECT_NEAR(abs_b(i), 3.0, 1e-14);
}

// ── computeMargPriorError ────────────────────────────────────────────────────

TEST(ComputeMargPriorErrorTest, ZeroDeltaZeroB) {
    MargHelper helper;

    int n = 2;
    auto state = MakeTestState(n, true);

    MargLinData mld;
    mld.old_state = *state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6);
    mld.b = Eigen::VectorXd::Zero(n * 6);

    double error = 999.0;
    helper.computeMargPriorError(mld, error);

    EXPECT_NEAR(error, 0.0, 1e-14);
}

TEST(ComputeMargPriorErrorTest, NonZeroDelta) {
    MargHelper helper;

    int n = 2;
    auto state = MakeTestState(n, true);
    Eigen::Vector<double, 6> d0;
    d0 << 0.5, 0.0, 0.0, 0.0, 0.0, 0.0;
    state->poses[0].applyDelta(d0);

    MargLinData mld;
    mld.old_state = *state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6) * 2.0;  // H=2I
    mld.b = Eigen::VectorXd::Zero(n * 6);

    double error = 0.0;
    helper.computeMargPriorError(mld, error);

    // error = delta^T * H^T * (0.5 * H * delta + b)
    //       = delta^T * 2I * (I * delta + 0)
    //       = 2 * ||delta||^2 = 2 * 0.5^2 = 0.5
    EXPECT_NEAR(error, 0.5, 1e-14);
}

TEST(ComputeMargPriorErrorTest, NonZeroB) {
    MargHelper helper;

    int n = 2;
    auto state = MakeTestState(n, true);
    // delta = 0

    MargLinData mld;
    mld.old_state = *state;
    mld.H = Eigen::MatrixXd::Identity(n * 6, n * 6);
    Eigen::VectorXd b_vec(n * 6);
    b_vec << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    mld.b = b_vec;

    double error = 0.0;
    helper.computeMargPriorError(mld, error);

    // error = 0^T * I * (0.5*I*0 + b) = 0
    EXPECT_NEAR(error, 0.0, 1e-14);
}

}  // namespace
}  // namespace tassel_core
