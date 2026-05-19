#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <cmath>
#include <limits>

#include "marginalization/marg_helper.h"

namespace tassel_core {
namespace {

// ── helpers ─────────────────────────────────────────────────────────────────

void computeExpected(
    const Eigen::MatrixXd& Q2Jp_orig, const Eigen::VectorXd& Q2r_orig, Eigen::MatrixXd& expected_H,
    Eigen::VectorXd& expected_b) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Q2Jp_orig);
    Eigen::MatrixXd R = qr.matrixQR().template triangularView<Eigen::Upper>();
    Eigen::VectorXd QTr = qr.householderQ().transpose() * Q2r_orig;

    const Eigen::Index e_rows = R.rows() - 1;
    const Eigen::Index e_cols = R.cols() - 1;
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

// ── MakeTestState: linearizes ALL frames (computeDelta now reads all except frame 0) ──
std::shared_ptr<State> MakeTestState(int n, bool linearize) {
    auto s = std::make_shared<State>(n);
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix3d R =
            Eigen::AngleAxisd(i * 0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Vector3d t(i * 0.1, i * 0.02, 0);
        s->poses[i] = PoseVelBiasState(Sophus::SE3d(R, t));
    }
    if (linearize) {
        for (int i = 0; i < n; ++i) {
            s->poses[i].setLinearized();
        }
    }
    return s;
}

// ── marginalizeOldest ───────────────────────────────────────────────────────

// 3×4 full rank (rows < cols)
TEST(MarginalizeOldestTest, FullRank3x4) {
    Eigen::MatrixXd Q2Jp(3, 4);
    Q2Jp << 2.0, 1.0, 3.0, 4.0, 4.0, 5.0, 9.0, 2.0, 7.0, 8.0, 6.0, 1.0;
    Eigen::VectorXd Q2r(3);
    Q2r << 1.0, 2.0, 3.0;

    const size_t keep_size = 3;
    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    Eigen::VectorXd Q2r_orig = Q2r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeOldest(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2Jp.cols(), 0);
    EXPECT_EQ(Q2r.rows(), 0);

    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
    EXPECT_EQ(marg_sqrt_H.cols(), static_cast<Eigen::Index>(keep_size));

    for (Eigen::Index r = 0; r < marg_sqrt_H.rows(); ++r) {
        for (Eigen::Index c = 0; c < std::min(r, marg_sqrt_H.cols()); ++c) {
            EXPECT_NEAR(marg_sqrt_H(r, c), 0.0, 1e-14)
                << "marg_sqrt_H(" << r << "," << c << ") should be zero (below diagonal)";
        }
    }

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

// 2×3 short-wide
TEST(MarginalizeOldestTest, ShortWide2x3) {
    Eigen::MatrixXd Q2Jp(2, 3);
    Q2Jp << 3.0, 1.0, 4.0, 0.0, 2.0, 5.0;
    Eigen::VectorXd Q2r(2);
    Q2r << 1.0, 2.0;

    const size_t keep_size = 2;
    Eigen::MatrixXd Q2Jp_orig = Q2Jp;
    Eigen::VectorXd Q2r_orig = Q2r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeOldest(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

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

// 4×5 rank-deficient (rank=2)
TEST(MarginalizeOldestTest, RankDeficient4x5) {
    Eigen::MatrixXd Q2Jp(4, 5);
    Q2Jp << 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 1.0, 0.0, 1.0,
        1.0, 0.0, -1.0;
    Eigen::VectorXd Q2r(4);
    Q2r << 0.1, 0.2, 0.3, 0.4;

    const size_t keep_size = 4;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeOldest(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);

    EXPECT_LE(marg_sqrt_H.rows(), 2);
    EXPECT_GE(marg_sqrt_H.rows(), 1);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());
}

// First column near-zero → marg_rank = 0
TEST(MarginalizeOldestTest, NearZeroFirstColumn) {
    const double eps = std::numeric_limits<double>::epsilon();
    Eigen::MatrixXd Q2Jp(3, 4);
    Q2Jp << eps * 0.01, 2.0, 3.0, 4.0, eps * 0.01, 5.0, 6.0, 7.0, eps * 0.01, 8.0, 9.0, 1.0;
    Eigen::VectorXd Q2r(3);
    Q2r << 1.0, 2.0, 3.0;

    const size_t keep_size = 3;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeOldest(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());

    EXPECT_LE(marg_sqrt_H.rows(), 3);
}

// 2×3 minimal working size
TEST(MarginalizeOldestTest, Minimal2x3) {
    Eigen::MatrixXd Q2Jp(2, 3);
    Q2Jp << 1.0, 3.0, 5.0, 2.0, 4.0, 6.0;
    Eigen::VectorXd Q2r(2);
    Q2r << 0.5, 1.5;

    const size_t keep_size = 2;
    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeOldest(1, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    EXPECT_EQ(Q2Jp.rows(), 0);
    EXPECT_EQ(Q2r.rows(), 0);
    EXPECT_EQ(marg_sqrt_H.cols(), keep_size);
    EXPECT_EQ(marg_sqrt_H.rows(), marg_sqrt_b.rows());

    if (marg_sqrt_H.rows() > 1) {
        for (Eigen::Index r = 0; r < marg_sqrt_H.rows(); ++r) {
            for (Eigen::Index c = 0; c < std::min(r, marg_sqrt_H.cols()); ++c) {
                EXPECT_NEAR(marg_sqrt_H(r, c), 0.0, 1e-14);
            }
        }
    }
}

// ── computeDelta: extracts kept frames 0..n-2, excludes newest frame n-1 ─────

TEST(ComputeDeltaTest, AllZeroDeltas) {
    auto state = MakeTestState(4, true);  // 4 frames, all linearized, deltas zero

    Eigen::VectorXd delta;
    MargHelper::computeDelta(*state, delta);

    // size = (max_frame_count - 1) * 6 = 3 * 6 = 18
    EXPECT_EQ(delta.size(), 3 * 6);
    EXPECT_TRUE(delta.isApproxToConstant(0, 1e-14));
}

TEST(ComputeDeltaTest, NonZeroDeltas) {
    auto state = MakeTestState(3, true);  // all linearized

    // Apply deltas to kept frames 0 and 1
    Eigen::Vector<double, 6> d0;
    d0 << 0.1, 0.2, 0.3, 0.01, 0.02, 0.03;
    state->poses[0].applyDelta(d0);
    Eigen::Vector<double, 6> d1;
    d1 << -0.1, 0.0, 0.5, -0.01, 0.0, 0.01;
    state->poses[1].applyDelta(d1);

    Eigen::VectorXd delta;
    MargHelper::computeDelta(*state, delta);

    // size = (3-1)*6 = 12
    EXPECT_EQ(delta.size(), 2 * 6);
    // Frame 0 → delta[0..5]
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(delta(i), d0(i), 1e-14);
    // Frame 1 → delta[6..11]
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(delta(6 + i), d1(i), 1e-14);
}

TEST(ComputeDeltaTest, ExcludesNewestFrame) {
    auto state = MakeTestState(2, true);  // frames 0,1 linearized

    Eigen::Vector<double, 6> d0;
    d0 << 0.5, 0.0, 0.0, 0.0, 0.0, 0.0;
    state->poses[0].applyDelta(d0);
    Eigen::Vector<double, 6> d1;
    d1 << 0.1, 0.0, 0.0, 0.0, 0.0, 0.0;
    state->poses[1].applyDelta(d1);

    Eigen::VectorXd delta;
    MargHelper::computeDelta(*state, delta);

    // size = (2-1)*6 = 6, only kept frame 0
    EXPECT_EQ(delta.size(), 6);
    // Frame 0 is extracted; frame 1 (newest) is excluded
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(delta(i), d0(i), 1e-14);
}

// ── linearizeMargPrior ──────────────────────────────────────────────────────

TEST(LinearizeMargPriorTest, IdentitySqrtH) {
    // H = I ((n-1)*6 × (n-1)*6), b = 0, delta = 0
    int n = 2;  // 1 prior frame + 1 new
    auto cur_state = MakeTestState(n, true);

    int marg_cols = (n - 1) * 6;  // 6
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols);
    mld.b = Eigen::VectorXd::Zero(marg_cols);

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Zero(marg_cols, marg_cols);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Zero(marg_cols);
    double error = 999.0;

    MargHelper::linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    EXPECT_TRUE(abs_H.isApprox(Eigen::MatrixXd::Identity(marg_cols, marg_cols), 1e-14));
    EXPECT_TRUE(abs_b.isApproxToConstant(0, 1e-14));
    EXPECT_NEAR(error, 0.0, 1e-14);
}

TEST(LinearizeMargPriorTest, NonZeroDeltaCompensation) {
    // H = 2*I, b = 0, delta on kept frame 0 = [1,0,0,0,0,0]
    // (frame 1 is the newest, excluded from computeDelta)
    int n = 2;
    auto cur_state = MakeTestState(n, true);

    Eigen::Vector<double, 6> d0;
    d0 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    cur_state->poses[0].applyDelta(d0);

    int marg_cols = (n - 1) * 6;
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols) * 2.0;
    mld.b = Eigen::VectorXd::Zero(marg_cols);

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Zero(marg_cols, marg_cols);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Zero(marg_cols);
    double error = 0.0;

    MargHelper::linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    // abs_H: 4*I
    Eigen::MatrixXd expected_H = Eigen::MatrixXd::Identity(marg_cols, marg_cols) * 4.0;
    EXPECT_TRUE(abs_H.isApprox(expected_H, 1e-14));

    // abs_b: H^T * (b + H * delta) = 2*I * (0 + 2*I*delta) = 4*delta = [4,0,0,0,0,0]
    EXPECT_NEAR(abs_b(0), 4.0, 1e-14);
    for (int i = 1; i < marg_cols; ++i) EXPECT_NEAR(abs_b(i), 0.0, 1e-14);

    // error = delta^T * H^T * (0.5 * H * delta + b) = delta^T * 2I * (I*delta) = 2
    EXPECT_NEAR(error, 2.0, 1e-14);
}

TEST(LinearizeMargPriorTest, NonZeroB) {
    // H = I, b = [1,2,...,6], delta = 0
    int n = 2;
    auto cur_state = MakeTestState(n, true);

    int marg_cols = (n - 1) * 6;
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols);

    Eigen::VectorXd b_vec(marg_cols);
    for (int i = 0; i < marg_cols; ++i) b_vec(i) = static_cast<double>(i + 1);
    mld.b = b_vec;

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Zero(marg_cols, marg_cols);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Zero(marg_cols);
    double error = 0.0;

    MargHelper::linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    EXPECT_TRUE(abs_H.isApprox(Eigen::MatrixXd::Identity(marg_cols, marg_cols), 1e-14));
    EXPECT_TRUE(abs_b.isApprox(b_vec, 1e-14));
    EXPECT_NEAR(error, 0.0, 1e-14);
}

TEST(LinearizeMargPriorTest, Accumulation) {
    int n = 2;
    auto cur_state = MakeTestState(n, true);

    int marg_cols = (n - 1) * 6;
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols);
    mld.b = Eigen::VectorXd::Ones(marg_cols);

    Eigen::MatrixXd abs_H = Eigen::MatrixXd::Ones(marg_cols, marg_cols);
    Eigen::VectorXd abs_b = Eigen::VectorXd::Ones(marg_cols);
    double error = 0.0;

    MargHelper::linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);
    MargHelper::linearizeMargPrior(mld, *cur_state, abs_H, abs_b, error);

    // abs_H: identity added twice + initial ones
    Eigen::MatrixXd expected_H = 3.0 * Eigen::MatrixXd::Identity(marg_cols, marg_cols);
    for (int r = 0; r < marg_cols; ++r)
        for (int c = 0; c < marg_cols; ++c)
            if (r != c) expected_H(r, c) = 1.0;
    EXPECT_TRUE(abs_H.isApprox(expected_H, 1e-14));

    // abs_b: added twice + initial ones = vector of 3's
    for (int i = 0; i < marg_cols; ++i) EXPECT_NEAR(abs_b(i), 3.0, 1e-14);
}

// ── computeMargPriorError ───────────────────────────────────────────────────

TEST(ComputeMargPriorErrorTest, ZeroDeltaZeroB) {
    int n = 2;
    auto state = MakeTestState(n, true);

    int marg_cols = (n - 1) * 6;
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols);
    mld.b = Eigen::VectorXd::Zero(marg_cols);

    double error = 999.0;
    MargHelper::computeMargPriorError(mld, *state, error);

    EXPECT_NEAR(error, 0.0, 1e-14);
}

TEST(ComputeMargPriorErrorTest, NonZeroDelta) {
    int n = 2;
    auto state = MakeTestState(n, true);
    Eigen::Vector<double, 6> d0;
    d0 << 0.5, 0.0, 0.0, 0.0, 0.0, 0.0;
    state->poses[0].applyDelta(d0);  // kept frame 0, frame 1 is newest

    int marg_cols = (n - 1) * 6;
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols) * 2.0;
    mld.b = Eigen::VectorXd::Zero(marg_cols);

    double error = 0.0;
    MargHelper::computeMargPriorError(mld, *state, error);

    // error = delta^T * H^T * (0.5 * H * delta + b)
    //       = delta^T * 2I * (I * delta + 0)
    //       = 2 * ||delta||^2 = 2 * 0.5^2 = 0.5
    EXPECT_NEAR(error, 0.5, 1e-14);
}

TEST(ComputeMargPriorErrorTest, NonZeroB) {
    int n = 2;
    auto state = MakeTestState(n, true);

    int marg_cols = (n - 1) * 6;
    MargLinData mld;
    mld.H = Eigen::MatrixXd::Identity(marg_cols, marg_cols);
    Eigen::VectorXd b_vec(marg_cols);
    b_vec << 1, 2, 3, 4, 5, 6;
    mld.b = b_vec;

    double error = 0.0;
    MargHelper::computeMargPriorError(mld, *state, error);

    // error = 0^T * I * (0.5*I*0 + b) = 0
    EXPECT_NEAR(error, 0.0, 1e-14);
}

}  // namespace
}  // namespace tassel_core
