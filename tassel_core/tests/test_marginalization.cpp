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

}  // namespace
}  // namespace tassel_core
