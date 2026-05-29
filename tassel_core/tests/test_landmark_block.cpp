#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <random>

#include "factor/landmark_block.h"

namespace tassel_core {
namespace {

constexpr double kQrTol = 1e-12;

class LandmarkBlockTest : public ::testing::Test {
protected:
    void SetUp() override { rng_.seed(42); }

    std::mt19937 rng_;
};

// ── Allocation ───────────────────────────────────────────────────────────

TEST_F(LandmarkBlockTest, AllocateSetsDimensions) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(3, 2, 6);

    EXPECT_EQ(lb.numRows(), 4);
    EXPECT_EQ(lb.keptRows(), 3);
    EXPECT_EQ(lb.paddingIdx(), 18);
}

TEST_F(LandmarkBlockTest, AllocateSingleFrame) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(1, 1, 6);

    EXPECT_EQ(lb.numRows(), 2);
    EXPECT_EQ(lb.keptRows(), 1);
    EXPECT_EQ(lb.paddingIdx(), 6);
}

TEST_F(LandmarkBlockTest, AllocateWithDifferentDim) {
    LandmarkBlock lb(15, nullptr);
    lb.allocate(2, 3, 15);

    EXPECT_EQ(lb.numRows(), 6);
    EXPECT_EQ(lb.keptRows(), 5);
    EXPECT_EQ(lb.paddingIdx(), 30);
}

// Verify storage size is correct
TEST_F(LandmarkBlockTest, AllocateStorageSize) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(5, 2, 6);

    const auto& s = lb.storage();
    EXPECT_EQ(s.rows(), 4);
    EXPECT_EQ(s.cols(), lb.residualIdx() + 1);
}

// ── QR: landmark column zeroed below row 0 ─────────────────────────────

TEST_F(LandmarkBlockTest, QRZerosLandmarkColumn) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(3, 3, 6);

    auto& s = lb.mutableStorage();
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int r = 0; r < s.rows(); ++r) {
        for (int c = 0; c < s.cols(); ++c) {
            s(r, c) = dist(rng_);
        }
    }
    s.col(lb.landmarkIdx()).setRandom();
    s(0, lb.landmarkIdx()) = 1.0;

    lb.performQR();

    int lm = lb.landmarkIdx();
    for (int r = 1; r < lb.numRows(); ++r) {
        EXPECT_NEAR(s(r, lm), 0.0, kQrTol) << "lm column not zeroed at row " << r;
    }
    EXPECT_NE(std::abs(s(0, lm)), 0.0) << "pivot should be non-zero";
}

TEST_F(LandmarkBlockTest, QRFrobeniusNormPreserved) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(2, 4, 6);

    auto& s = lb.mutableStorage();
    s.setRandom();

    double norm_before = s.norm();
    lb.performQR();
    double norm_after = s.norm();

    EXPECT_NEAR(norm_before, norm_after, 1e-10);
}

TEST_F(LandmarkBlockTest, QRZeroLandmarkColumnSkipsAllZeros) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(1, 3, 6);

    auto& s = lb.mutableStorage();
    s.setRandom();
    s.col(lb.landmarkIdx()).setZero();

    lb.performQR();

    for (int r = 0; r < lb.numRows(); ++r) {
        EXPECT_NEAR(s(r, lb.landmarkIdx()), 0.0, kQrTol);
    }
}

// ── QR: single observation pair (2 rows) ───────────────────────────────

TEST_F(LandmarkBlockTest, QRSingleObservation) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(2, 1, 6);

    auto& s = lb.mutableStorage();
    s.setRandom();

    lb.performQR();

    EXPECT_NEAR(s(1, lb.landmarkIdx()), 0.0, kQrTol);
}

// ── QR: verify the Givens rotation is correct on a known 2-row case ────

TEST_F(LandmarkBlockTest, QRGivensRotationExact) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(2, 1, 6);

    auto& s = lb.mutableStorage();
    s.setZero();

    // Set up known values: row0 = [1,2,3], row1 = [4,5,6]
    // landmark column is at lm_idx (after padding)
    int lm = lb.landmarkIdx();
    int res = lb.residualIdx();
    int pad = lb.paddingIdx();

    s(0, 0) = 1;
    s(0, lm) = 2;
    s(0, res) = 3;
    s(1, 0) = 4;
    s(1, lm) = 5;
    s(1, res) = 6;

    // Givens rotation to zero s(1, lm)=5 using s(0, lm)=2
    // G = [c  s; -s  c] such that -s*2 + c*5 = 0
    // c = 2/r, s = 5/r where r = sqrt(4+25) = sqrt(29)
    double r = std::sqrt(29.0);
    double c = 2.0 / r;
    double s_val = 5.0 / r;

    lb.performQR();

    EXPECT_NEAR(s(1, lm), 0.0, kQrTol);

    // row0_new = c * row0_old + s_val * row1_old
    EXPECT_NEAR(s(0, 0), c * 1 + s_val * 4, 1e-12);
    EXPECT_NEAR(s(0, res), c * 3 + s_val * 6, 1e-12);

    // row1_new = -s_val * row0_old + c * row1_old
    EXPECT_NEAR(s(1, 0), -s_val * 1 + c * 4, 1e-12);
    EXPECT_NEAR(s(1, res), -s_val * 3 + c * 6, 1e-12);
}

// ── QR: 3-row case, verify two Givens rotations zero both rows ─────────

TEST_F(LandmarkBlockTest, QRThreeRowCase) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(2, 2, 6);

    auto& s = lb.mutableStorage();
    s.setZero();
    int lm = lb.landmarkIdx();

    s(0, lm) = 1;
    s(1, lm) = 3;
    s(2, lm) = 4;
    s(3, lm) = 7;

    lb.performQR();

    for (int r = 1; r < lb.numRows(); ++r) {
        EXPECT_NEAR(s(r, lm), 0.0, kQrTol);
    }
    EXPECT_NE(std::abs(s(0, lm)), 0.0);
}

// ── get_dense_Q2Jp_Q2r ──────────────────────────────────────────────────

TEST_F(LandmarkBlockTest, GetDenseExtractsCorrectRows) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(2, 3, 6);

    auto& s = lb.mutableStorage();
    s.setRandom();

    lb.performQR();

    int kept = lb.keptRows();
    int pad = lb.paddingIdx();

    Eigen::MatrixXd Q2Jp(kept, pad);
    Eigen::VectorXd Q2r(kept);
    lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, 0);

    for (int r = 0; r < kept; ++r) {
        for (int c = 0; c < pad; ++c) {
            EXPECT_DOUBLE_EQ(Q2Jp(r, c), s(r + 1, c));
        }
        EXPECT_DOUBLE_EQ(Q2r(r), s(r + 1, lb.residualIdx()));
    }
}

TEST_F(LandmarkBlockTest, GetDenseWithOffsetPreservesPrefix) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(1, 2, 6);

    auto& s = lb.mutableStorage();
    s.setRandom();

    lb.performQR();

    int kept = lb.keptRows();
    int pad = lb.paddingIdx();
    int offset = 3;

    Eigen::MatrixXd Q2Jp(offset + kept, pad);
    Eigen::VectorXd Q2r(offset + kept);
    Q2Jp.setConstant(-1.0);
    Q2r.setConstant(-1.0);
    lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, offset);

    for (int r = 0; r < offset; ++r) {
        for (int c = 0; c < pad; ++c) {
            EXPECT_DOUBLE_EQ(Q2Jp(r, c), -1.0) << "offset prefix was overwritten";
        }
        EXPECT_DOUBLE_EQ(Q2r(r), -1.0);
    }
    for (int r = 0; r < kept; ++r) {
        for (int c = 0; c < pad; ++c) {
            EXPECT_DOUBLE_EQ(Q2Jp(offset + r, c), s(r + 1, c));
        }
        EXPECT_DOUBLE_EQ(Q2r(offset + r), s(r + 1, lb.residualIdx()));
    }
}

// ── QR: consistency check via linear system ────────────────────────────

// After QR, rows 1..n-1 give marginalized pose constraints:
//   Q2^T * Jp * Δpose = Q2^T * r
// Row 0 preserves the landmark coupling:
//   Jl' * Δlm + Jp' * Δpose = r'
//
// Build a consistent system (r = Jp * dx_true + Jl * dl_true), apply QR,
// then verify the true state satisfies the marginalized constraints.

TEST_F(LandmarkBlockTest, MarginalizedSystemConsistency) {
    LandmarkBlock lb(6, nullptr);
    lb.allocate(2, 3, 6);

    auto& s = lb.mutableStorage();
    int pad = lb.paddingIdx();
    int lm = lb.landmarkIdx();
    int res = lb.residualIdx();

    // Construct a consistent system: r = Jp * dx_true + Jl * dl_true
    Eigen::VectorXd dx_true = Eigen::VectorXd::Random(pad);
    double dl_true = 2.5;
    s.setRandom();
    s.col(res) = s.block(0, 0, lb.numRows(), pad) * dx_true + s.col(lm) * dl_true;

    lb.performQR();

    // The true state must satisfy the marginalized constraints (rows 1+)
    for (int r = 1; r < lb.numRows(); ++r) {
        double pred = (s.block(r, 0, 1, pad) * dx_true).value();
        double obs = s(r, res);
        EXPECT_NEAR(pred, obs, 1e-12) << "True state violated marginalized constraint at row " << r;
    }

    // Row 0 should also be consistent with the true state
    double row0_residual =
        (s.block(0, 0, 1, pad) * dx_true).value() + s(0, lm) * dl_true - s(0, res);
    EXPECT_NEAR(row0_residual, 0.0, 1e-12);

    // Verify landmark column is zeroed below row 0
    for (int r = 1; r < lb.numRows(); ++r) {
        EXPECT_NEAR(s(r, lm), 0.0, 1e-12);
    }
}

// ── Large random stress test ─────────────────────────────────────────────

TEST_F(LandmarkBlockTest, QRStressTest) {
    LandmarkBlock lb(6, nullptr);
    int num_obs = 20;
    lb.allocate(5, num_obs, 6);

    auto& s = lb.mutableStorage();
    std::normal_distribution<double> dist(0.0, 10.0);
    for (int r = 0; r < s.rows(); ++r) {
        for (int c = 0; c < s.cols(); ++c) {
            s(r, c) = dist(rng_);
        }
    }

    double norm_before = s.norm();
    lb.performQR();

    int lm = lb.landmarkIdx();
    for (int r = 1; r < lb.numRows(); ++r) {
        EXPECT_NEAR(s(r, lm), 0.0, kQrTol);
    }
    EXPECT_NEAR(s.norm(), norm_before, 1e-9);
}

}  // namespace
}  // namespace tassel_core
