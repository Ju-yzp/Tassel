#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "linearization/landmark_block.h"

namespace {

std::shared_ptr<tassel_core::State> makeState(int max_frame_count) {
    auto state = std::make_shared<tassel_core::State>(max_frame_count);
    state->Rs[0] = Eigen::Matrix3d::Identity();
    state->Ps[0] = Eigen::Vector3d(0, 0, 0);
    state->Rs[1] = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()).toRotationMatrix();
    state->Ps[1] = Eigen::Vector3d(0.5, 0.0, 0.0);
    state->Rs[2] = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    state->Ps[2] = Eigen::Vector3d(1.0, 0.1, 0.0);
    return state;
}

Eigen::Vector3d computeTargetUV(
    const Eigen::Vector3d& host_uv, double inv_depth, const Eigen::Matrix3d& host_R,
    const Eigen::Vector3d& host_P, const Eigen::Matrix3d& target_R,
    const Eigen::Vector3d& target_P) {
    Eigen::Vector3d pt_H = host_uv / inv_depth;
    Eigen::Vector3d pt_W = host_R * pt_H + host_P;
    return (target_R.transpose() * (pt_W - target_P)).normalized();
}

tassel_core::Feature makeFeature(
    size_t start_frame, double inv_depth, const Eigen::Vector3d& uv_host,
    const std::vector<Eigen::Vector3d>& uv_targets) {
    tassel_core::Feature feat(start_frame, uv_targets.size() + 1);
    feat.estimated_depth = 1.0 / inv_depth;
    feat.observations.resize(uv_targets.size() + 1);
    feat.observations[0].uv = uv_host;
    for (size_t i = 0; i < uv_targets.size(); ++i) {
        feat.observations[i + 1].uv = uv_targets[i];
    }
    return feat;
}

void expectNonZeroInColumns(
    const Eigen::MatrixXd& mat, int row_start, int row_count,
    const std::vector<int>& expected_nonzero_col_ranges, int total_cols) {
    std::vector<bool> col_nonzero(total_cols, false);
    for (int r = 0; r < row_count; ++r) {
        for (int c = 0; c < total_cols; ++c) {
            if (std::abs(mat(row_start + r, c)) > 1e-15) {
                col_nonzero[c] = true;
            }
        }
    }

    for (int c = 0; c < total_cols; ++c) {
        bool expected_nonzero = false;
        for (size_t i = 0; i < expected_nonzero_col_ranges.size(); i += 2) {
            if (c >= expected_nonzero_col_ranges[i] && c < expected_nonzero_col_ranges[i + 1]) {
                expected_nonzero = true;
                break;
            }
        }

        if (expected_nonzero && !col_nonzero[c]) {
            ADD_FAILURE() << "column " << c << " should be non-zero but is zero";
        }
        if (!expected_nonzero && col_nonzero[c]) {
            ADD_FAILURE() << "column " << c << " should be zero but has non-zero values";
        }
    }
}

}  // namespace

// ── Tests ────────────────────────────────────────────────────────────────────

constexpr int POSE_SIZE = 6;

// Host at frame 1, target at frame 2, max_frame_count=2 (3 frame slots)
// Layout: frame0(0-5), frame1/host(6-11), frame2/target(12-17), pad(18-19),
//         J_l(20), res(21)
TEST(LandmarkBlockTest, HostAtFrame1TargetAtFrame2) {
    const int max_frame_count = 2;
    auto state = makeState(max_frame_count);

    const double inv_depth = 0.5;
    const Eigen::Vector3d uv_host(0.1, 0.2, 1.0);
    Eigen::Vector3d uv_target =
        computeTargetUV(uv_host, inv_depth, state->Rs[1], state->Ps[1], state->Rs[2], state->Ps[2]);

    auto feat = makeFeature(/*start_frame=*/1, inv_depth, uv_host, {uv_target});

    tassel_core::LandmarkBlock block;
    block.allocate(&feat, state.get());

    ASSERT_EQ(block.getNumCols(), 22);
    ASSERT_EQ(block.getNumRows(), 3);  // 1 target obs * 2 + 1 damping
    ASSERT_EQ(block.getPaddingIdx(), 18);
    ASSERT_EQ(block.getLandmarkCol(), 20);

    double error = block.linearize();

    const auto& storage = block.getStorage();
    // Non-zero: host cols 6-11, target cols 12-17, J_l col 20 (res col 21 is ~0 with
    // ground-truth observations)
    expectNonZeroInColumns(storage, 0, 2, {6, 18, 20, 21}, block.getNumCols());

    // Residual should be ~0 (ground-truth projection used for observations)
    EXPECT_NEAR(error, 0.0, 1e-10);
}

// Host at frame 0, target at frame 1
// Layout: frame0/host(0-5), frame1/target(6-11), frame2(12-17), pad(18-19),
//         J_l(20), res(21)
TEST(LandmarkBlockTest, HostAtFrame0TargetAtFrame1) {
    const int max_frame_count = 2;
    auto state = makeState(max_frame_count);

    const double inv_depth = 0.5;
    const Eigen::Vector3d uv_host(0.1, 0.2, 1.0);
    Eigen::Vector3d uv_target =
        computeTargetUV(uv_host, inv_depth, state->Rs[0], state->Ps[0], state->Rs[1], state->Ps[1]);

    auto feat = makeFeature(/*start_frame=*/0, inv_depth, uv_host, {uv_target});

    tassel_core::LandmarkBlock block;
    block.allocate(&feat, state.get());

    ASSERT_EQ(block.getNumCols(), 22);
    ASSERT_EQ(block.getNumRows(), 3);

    double error = block.linearize();

    const auto& storage = block.getStorage();
    // Non-zero: host cols 0-5, target cols 6-11, J_l col 20 (res col 21 is ~0)
    expectNonZeroInColumns(storage, 0, 2, {0, 12, 20, 21}, block.getNumCols());

    EXPECT_NEAR(error, 0.0, 1e-10);
}

// Host at frame 0, observed at frames 1 and 2 (2 target obs)
// Layout: 3 frames * 6 = 18 pose cols + 2 pad + 1 J_l + 1 res = 22 cols
//         5 rows = 2 obs * 2 + 1 damping
TEST(LandmarkBlockTest, ThreeFrames) {
    const int max_frame_count = 2;
    auto state = makeState(max_frame_count);

    const double inv_depth = 0.5;
    const Eigen::Vector3d uv_host(0.1, 0.2, 1.0);
    Eigen::Vector3d uv_target1 =
        computeTargetUV(uv_host, inv_depth, state->Rs[0], state->Ps[0], state->Rs[1], state->Ps[1]);
    Eigen::Vector3d uv_target2 =
        computeTargetUV(uv_host, inv_depth, state->Rs[0], state->Ps[0], state->Rs[2], state->Ps[2]);

    auto feat = makeFeature(/*start_frame=*/0, inv_depth, uv_host, {uv_target1, uv_target2});

    tassel_core::LandmarkBlock block;
    block.allocate(&feat, state.get());

    ASSERT_EQ(block.getNumCols(), 22);
    ASSERT_EQ(block.getNumRows(), 5);  // 2 target obs * 2 + 1 damping

    double error = block.linearize();

    const auto& storage = block.getStorage();
    // All 3 frames contribute pose columns: 0-17 + J_l col 20 (res col 21 is ~0)
    expectNonZeroInColumns(storage, 0, 4, {0, 18, 20, 21}, block.getNumCols());

    EXPECT_NEAR(error, 0.0, 1e-10);
}

// QR → get_dense_Q2Jp_Q2r dimensions
TEST(LandmarkBlockTest, QRDenseExtractionDimensions) {
    auto state = makeState(2);

    const double inv_depth = 0.5;
    const Eigen::Vector3d uv_host(0.1, 0.2, 1.0);
    Eigen::Vector3d uv_target =
        computeTargetUV(uv_host, inv_depth, state->Rs[1], state->Ps[1], state->Rs[2], state->Ps[2]);

    auto feat = makeFeature(1, inv_depth, uv_host, {uv_target});

    tassel_core::LandmarkBlock block;
    block.allocate(&feat, state.get());
    block.linearize();
    block.performQR();

    const int nr = block.getNumRows() - 1;
    const int pose_dim = block.getPaddingIdx();

    Eigen::MatrixXd Q2Jp(nr, pose_dim);
    Eigen::VectorXd Q2r(nr);
    block.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, 0);

    // Q2Jp: (num_rows-1) x padding_idx = 2 x 18
    ASSERT_EQ(Q2Jp.rows(), nr);
    ASSERT_EQ(Q2Jp.cols(), pose_dim);

    // Q2r: num_rows-1 = 2
    ASSERT_EQ(Q2r.rows(), nr);

    // After QR with ground-truth obs, residual should be ~0
    EXPECT_NEAR(Q2r.squaredNorm(), 0.0, 1e-10);
}

// add_dense_H_b accumulates the reduced camera system
TEST(LandmarkBlockTest, DenseHbAccumulation) {
    auto state = makeState(2);

    const double inv_depth = 0.5;
    const Eigen::Vector3d uv_host(0.1, 0.2, 1.0);
    Eigen::Vector3d uv_target =
        computeTargetUV(uv_host, inv_depth, state->Rs[1], state->Ps[1], state->Rs[2], state->Ps[2]);

    auto feat = makeFeature(1, inv_depth, uv_host, {uv_target});

    tassel_core::LandmarkBlock block;
    block.allocate(&feat, state.get());
    block.linearize();
    block.performQR();

    const int pose_dim = block.getPaddingIdx();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(pose_dim, pose_dim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(pose_dim);

    block.add_dense_H_b(H, b);

    // H should be non-zero only at host(6-11) and target(12-17) pose blocks
    for (int i = 0; i < pose_dim; ++i) {
        for (int j = 0; j < pose_dim; ++j) {
            bool in_host = (i >= 6 && i < 12) || (i >= 12 && i < 18);
            bool in_target = (j >= 6 && j < 12) || (j >= 12 && j < 18);
            if (!in_host || !in_target) {
                EXPECT_NEAR(H(i, j), 0.0, 1e-15) << "H(" << i << "," << j << ") should be zero";
            }
        }
    }

    // b should only be non-zero at host and target pose indices
    for (int i = 0; i < pose_dim; ++i) {
        if (i < 6 || i >= 18) {
            EXPECT_NEAR(b(i), 0.0, 1e-10)
                << "b(" << i << ") should be zero for non-participating frame";
        }
    }

    // H should be symmetric
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
}
