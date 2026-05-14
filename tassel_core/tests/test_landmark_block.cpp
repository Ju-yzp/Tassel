#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <random>
#include <vector>

#include "frond_end/feature.h"
#include "linearization/landmark_block.h"
#include "loss_fuction/loss_fuction_base.h"
#include "state/state.h"
#include "tassel_tools/jacobian_checker.h"

namespace tassel_core {
namespace {

constexpr int POSE_SIZE = 6;

// ── helpers ───────────────────────────────────────────────────────────────────

std::shared_ptr<State> MakeState(int n) {
    auto s = std::make_shared<State>(n);
    for (int i = 0; i < n; ++i) {
        double x = i * 0.2;
        double y = (i % 2 == 0) ? 0.03 : -0.03;
        double z = 0.0;
        Eigen::Vector3d t(x, y, z);
        double angle = (i - 1) * 0.03;
        Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
        s->poses[i] = PoseStateWithLin(Sophus::SE3d(R, t));
    }
    return s;
}

Eigen::Vector3d Project(const Eigen::Vector3d& P_w, const State& s, int cam_i) {
    Sophus::SE3d T_wc = s.poses[cam_i].get_pose();
    Eigen::Vector3d P_c = T_wc.inverse() * P_w;
    return Eigen::Vector3d(P_c.x() / P_c.z(), P_c.y() / P_c.z(), 1.0);
}

Feature MakeFeature(int start_id, const std::vector<Eigen::Vector3d>& all_uvs, double depth) {
    Feature f;
    f.start_frame_id = start_id;
    f.estimated_depth = depth;
    f.observations.resize(all_uvs.size());
    for (size_t i = 0; i < all_uvs.size(); ++i) f.observations[i].uv = all_uvs[i];
    return f;
}

// ── construction ──────────────────────────────────────────────────────────────

TEST(LandmarkBlockTest, DefaultConstruction) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    EXPECT_EQ(lb.getNumRows(), 0);
    EXPECT_EQ(lb.getNumCols(), 0);
    EXPECT_EQ(lb.getPaddingIdx(), 0);
    EXPECT_EQ(lb.getLandmarkCol(), 0);
    EXPECT_FALSE(lb.hasLandmarkDamping());
}

// ── allocate ──────────────────────────────────────────────────────────────────

TEST(LandmarkBlockTest, AllocateSingleObs) {
    auto state = MakeState(4);
    Feature f = MakeFeature(0, {Eigen::Vector3d::UnitZ(), Eigen::Vector3d(0.1, 0.2, 1.0)}, 1.5);

    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&f, state.get(), TrivialLoss{});

    EXPECT_EQ(lb.getNumRows(), 3);
    EXPECT_EQ(lb.getPaddingIdx(), 24);
    EXPECT_EQ(lb.getLandmarkCol(), 24);
    EXPECT_EQ(lb.getNumCols(), 26);
}

TEST(LandmarkBlockTest, AllocateMultiObs) {
    auto state = MakeState(4);
    Feature f = MakeFeature(
        0,
        {Eigen::Vector3d::UnitZ(), Eigen::Vector3d(0.1, 0.2, 1.0), Eigen::Vector3d(0.2, 0.3, 1.0),
         Eigen::Vector3d(0.15, 0.25, 1.0)},
        1.5);

    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&f, state.get(), TrivialLoss{});

    EXPECT_EQ(lb.getNumRows(), 7);
    EXPECT_EQ(lb.getNumCols(), 26);
}

TEST(LandmarkBlockTest, AllocatePadding) {
    auto state = MakeState(5);
    Feature f = MakeFeature(0, {Eigen::Vector3d::UnitZ(), Eigen::Vector3d(0.1, 0.2, 1.0)}, 1.5);

    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&f, state.get(), TrivialLoss{});

    EXPECT_EQ(lb.getPaddingIdx(), 30);
    EXPECT_EQ(lb.getLandmarkCol(), 32);
    EXPECT_EQ(lb.getNumCols(), 34);
    EXPECT_EQ(lb.getNumCols() % 4, 2);
    EXPECT_EQ(lb.getLandmarkCol() % 4, 0);
}

// ── linearize ─────────────────────────────────────────────────────────────────

class LinearizeFixture : public ::testing::Test {
protected:
    void SetUp() override {
        state_ = MakeState(4);
        P_world_ = Eigen::Vector3d(0.4, -0.25, 2.0);
        Eigen::Vector3d P_c_host = state_->poses[0].get_pose().inverse() * P_world_;
        double true_depth = P_c_host.z();
        Eigen::Vector3d uv_host(P_c_host.x() / P_c_host.z(), P_c_host.y() / P_c_host.z(), 1.0);
        std::vector<Eigen::Vector3d> uvs = {
            uv_host,
            Project(P_world_, *state_, 1),
            Project(P_world_, *state_, 2),
            Project(P_world_, *state_, 3),
        };
        feat_ = MakeFeature(0, uvs, true_depth);
    }

    std::shared_ptr<State> state_;
    Eigen::Vector3d P_world_;
    Feature feat_;
};

TEST_F(LinearizeFixture, ResidualNearZero) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    double err = lb.linearize();
    EXPECT_NEAR(err, 0.0, 1e-8);
}

TEST_F(LinearizeFixture, ObsRowsSparsity) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();

    const auto& S = lb.getStorage();
    const int nr = lb.getNumRows();
    const int nc = lb.getNumCols();
    int n_obs = static_cast<int>(feat_.observations.size()) - 1;
    int start = feat_.start_frame_id;

    for (int off = 1; off <= n_obs; ++off) {
        int obs_idx = off - 1;
        int tid = start + off;
        int row0 = obs_idx * 2;
        for (int d = 0; d < 2; ++d) {
            int r = row0 + d;
            double host_norm = S.block<1, POSE_SIZE>(r, start * POSE_SIZE).norm();
            EXPECT_LT(1e-12, host_norm) << "obs row " << r << " host block zero";
            double targ_norm = S.block<1, POSE_SIZE>(r, tid * POSE_SIZE).norm();
            EXPECT_LT(1e-12, targ_norm)
                << "obs row " << r << " target block (frame " << tid << ") zero";
            for (int f = 0; f < 4; ++f) {
                if (f == start || f == tid) continue;
                for (int c = f * POSE_SIZE; c < (f + 1) * POSE_SIZE; ++c)
                    EXPECT_NEAR(S(r, c), 0.0, 1e-14) << "obs row " << r << " unrelated frame " << f
                                                     << " col " << c << " non-zero";
            }
        }
    }

    for (int c = 0; c < nc; ++c) EXPECT_NEAR(S(nr - 1, c), 0.0, 1e-14) << "damping row col " << c;
}

TEST_F(LinearizeFixture, JacobianColumnNonZero) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();

    const auto& S = lb.getStorage();

    int lc = lb.getLandmarkCol();
    for (int r = 0; r < lb.getNumRows() - 1; ++r)
        EXPECT_LT(1e-10, std::abs(S(r, lc))) << "Jl zero at row " << r;

    int rc = lb.getLandmarkCol() + 1;
    for (int r = 0; r < lb.getNumRows() - 1; ++r)
        EXPECT_NEAR(S(r, rc), 0.0, 1e-8) << "residual non-zero at obs row " << r;
}

// ── analytical vs numerical Jacobian ──────────────────────────────────────────

TEST_F(LinearizeFixture, JlNumericalDerivative) {
    const double eps = 1e-6;

    feat_.estimated_depth = P_world_.z() + eps;
    LandmarkBlock lb_plus(MIN_DISTANCE, MAX_DISTANCE);
    lb_plus.allocate(&feat_, state_.get(), TrivialLoss{});
    lb_plus.linearize();
    Eigen::MatrixXd S_plus = lb_plus.getStorage();

    feat_.estimated_depth = P_world_.z() - eps;
    LandmarkBlock lb_minus(MIN_DISTANCE, MAX_DISTANCE);
    lb_minus.allocate(&feat_, state_.get(), TrivialLoss{});
    lb_minus.linearize();
    Eigen::MatrixXd S_minus = lb_minus.getStorage();

    int lc = lb_plus.getLandmarkCol();
    int rc = lc + 1;
    Eigen::VectorXd Jl_num = (S_plus.col(rc).head(lb_plus.getNumRows() - 1) -
                              S_minus.col(rc).head(lb_minus.getNumRows() - 1)) /
                             (2 * eps);

    feat_.estimated_depth = P_world_.z();
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    Eigen::VectorXd Jl_ana = lb.getStorage().col(lc).head(lb.getNumRows() - 1);

    EXPECT_TRUE(Jl_ana.isApprox(Jl_num, 1e-4)) << "analytical Jl:\n"
                                               << Jl_ana.transpose() << "\nnumerical Jl:\n"
                                               << Jl_num.transpose();
}

TEST_F(LinearizeFixture, JpNumericalDerivative) {
    const double eps = 1e-6;

    feat_.estimated_depth = P_world_.z();
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    const auto& S_nom = lb.getStorage();
    int rc = lb.getLandmarkCol() + 1;
    int nr = lb.getNumRows() - 1;

    Sophus::SE3d T0 = state_->poses[0].get_pose();
    auto perturb = [](const Sophus::SE3d& T, double v) {
        Eigen::Matrix<double, 6, 1> delta;
        delta << v, 0, 0, 0, 0, 0;
        return T * Sophus::SE3d::exp(delta);
    };

    state_->poses[0] = PoseStateWithLin(perturb(T0, eps));
    LandmarkBlock lb_p(MIN_DISTANCE, MAX_DISTANCE);
    lb_p.allocate(&feat_, state_.get(), TrivialLoss{});
    lb_p.linearize();
    Eigen::VectorXd r_p = lb_p.getStorage().col(rc).head(nr);

    state_->poses[0] = PoseStateWithLin(perturb(T0, -eps));
    LandmarkBlock lb_m(MIN_DISTANCE, MAX_DISTANCE);
    lb_m.allocate(&feat_, state_.get(), TrivialLoss{});
    lb_m.linearize();
    Eigen::VectorXd r_m = lb_m.getStorage().col(rc).head(nr);

    state_->poses[0] = PoseStateWithLin(T0);

    Eigen::VectorXd Jp_num = (r_p - r_m) / (2 * eps);
    Eigen::VectorXd Jp_ana = S_nom.col(0).head(nr);

    EXPECT_TRUE(Jp_ana.isApprox(Jp_num, 1e-4)) << "analytical Jp col0:\n"
                                               << Jp_ana.transpose() << "\nnumerical Jp col0:\n"
                                               << Jp_num.transpose();
}

// ── performQR ─────────────────────────────────────────────────────────────────

TEST_F(LinearizeFixture, QRUpperTriangular) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int lc = lb.getLandmarkCol();
    const auto& S = lb.getStorage();

    double R_val = std::abs(S(0, lc));
    EXPECT_GT(R_val, 1e-8);
    for (int r = 1; r < lb.getNumRows(); ++r)
        EXPECT_NEAR(S(r, lc), 0.0, 1e-14) << "Jl non-zero at row " << r;
}

TEST_F(LinearizeFixture, QRRowZeroResidualMatchesQ1) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    const auto& S = lb.getStorage();
    int rc = lb.getLandmarkCol() + 1;

    EXPECT_NEAR(S(0, rc), 0.0, 1e-8);
}

// ── get_dense_Q2Jp_Q2r ───────────────────────────────────────────────────────

TEST_F(LinearizeFixture, Q2JpQ2rDimensions) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int nr = lb.getNumRows() - 1;
    int pdim = lb.getPaddingIdx();

    Eigen::MatrixXd Q2Jp(nr, pdim);
    Eigen::VectorXd Q2r(nr);
    lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, 0);

    EXPECT_EQ(Q2Jp.rows(), nr);
    EXPECT_EQ(Q2Jp.cols(), pdim);
    EXPECT_EQ(Q2r.size(), nr);
    EXPECT_NEAR(Q2r.squaredNorm(), 0.0, 1e-8);
}

TEST_F(LinearizeFixture, Q2JpQ2rOffset) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int nr = lb.getNumRows() - 1;
    int pdim = lb.getPaddingIdx();
    int total_nr = nr + 100;

    Eigen::MatrixXd Q2Jp = Eigen::MatrixXd::Zero(total_nr, pdim);
    Eigen::VectorXd Q2r = Eigen::VectorXd::Zero(total_nr);
    lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, 50);

    for (int i = 0; i < 50; ++i) {
        EXPECT_NEAR(Q2r(i), 0.0, 1e-14);
        EXPECT_NEAR(Q2Jp.row(i).squaredNorm(), 0.0, 1e-14);
    }
    for (int i = 50 + nr; i < total_nr; ++i) {
        EXPECT_NEAR(Q2r(i), 0.0, 1e-14);
        EXPECT_NEAR(Q2Jp.row(i).squaredNorm(), 0.0, 1e-14);
    }
}

// ── add_dense_H_b ─────────────────────────────────────────────────────────────

TEST_F(LinearizeFixture, DenseHbSymmetry) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int pdim = lb.getPaddingIdx();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(pdim, pdim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(pdim);
    lb.add_dense_H_b(H, b);

    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
    EXPECT_NEAR(b.norm(), 0.0, 1e-8);
}

TEST_F(LinearizeFixture, DenseHbPositiveSemidefinite) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int pdim = lb.getPaddingIdx();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(pdim, pdim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(pdim);
    lb.add_dense_H_b(H, b);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    for (int i = 0; i < pdim; ++i)
        EXPECT_GE(es.eigenvalues()[i], -1e-10) << "negative eigenvalue at " << i;
}

// ── addJp_diag2 ──────────────────────────────────────────────────────────────

TEST_F(LinearizeFixture, JpDiag2MatchesManual) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();

    int pdim = lb.getPaddingIdx();
    Eigen::VectorXd d2 = Eigen::VectorXd::Zero(pdim);
    lb.addJp_diag2(d2);

    Eigen::VectorXd d2_man = Eigen::VectorXd::Zero(pdim);
    const auto& S = lb.getStorage();
    int start = feat_.start_frame_id;
    for (int off = 1; off < static_cast<int>(feat_.observations.size()); ++off) {
        int obs_idx = off - 1;
        int row = obs_idx * 2;
        int tid = start + off;
        d2_man.segment<POSE_SIZE>(start * POSE_SIZE) +=
            S.block<2, POSE_SIZE>(row, start * POSE_SIZE).colwise().squaredNorm();
        d2_man.segment<POSE_SIZE>(tid * POSE_SIZE) +=
            S.block<2, POSE_SIZE>(row, tid * POSE_SIZE).colwise().squaredNorm();
    }

    EXPECT_TRUE(d2.isApprox(d2_man, 1e-12));
}

// ── scaleJl_cols ──────────────────────────────────────────────────────────────

TEST_F(LinearizeFixture, ScaleJlColsEffect) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();

    int lc = lb.getLandmarkCol();
    int nr = lb.getNumRows();
    double norm_before = lb.getStorage().col(lc).head(nr - 1).norm();

    lb.scaleJl_cols();

    double norm_after = lb.getStorage().col(lc).head(nr - 1).norm();
    EXPECT_LT(norm_after, 1.0);
    EXPECT_NEAR(norm_after, norm_before / (1e-6 + norm_before), 1e-12);
}

// ── scaleJp_cols ──────────────────────────────────────────────────────────────

TEST_F(LinearizeFixture, ScaleJpColsEffect) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int pdim = lb.getPaddingIdx();
    int nr = lb.getNumRows();
    Eigen::VectorXd scaling(pdim);
    scaling.setOnes();
    scaling *= 2.0;

    Eigen::MatrixXd before = lb.getStorage().topLeftCorner(nr - 1, pdim);
    lb.scaleJp_cols(scaling);
    Eigen::MatrixXd after = lb.getStorage().topLeftCorner(nr - 1, pdim);

    EXPECT_TRUE(after.isApprox(before * 2.0, 1e-12));
}

// ── setLandmarkDamping ───────────────────────────────────────────────────────

TEST_F(LinearizeFixture, DampingApplyUndo) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    auto S_before = lb.getStorage();

    lb.setLandmarkDamping(1.0);
    EXPECT_TRUE(lb.hasLandmarkDamping());

    lb.setLandmarkDamping(0.0);
    EXPECT_FALSE(lb.hasLandmarkDamping());

    EXPECT_TRUE(lb.getStorage().isApprox(S_before, 1e-12));
}

TEST_F(LinearizeFixture, DampingAddsDiagonal) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    int lc = lb.getLandmarkCol();
    double Q1Jl_before = lb.getStorage()(0, lc);

    double lambda = 4.0;
    lb.setLandmarkDamping(lambda);

    double Q1Jl_after = lb.getStorage()(0, lc);
    EXPECT_GT(std::abs(Q1Jl_after), std::abs(Q1Jl_before));
}

// ── backSubstitute ────────────────────────────────────────────────────────────

TEST_F(LinearizeFixture, BackSubZeroInc) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    double depth_before = feat_.estimated_depth;
    double l_diff = 0.0;
    Eigen::VectorXd inc = Eigen::VectorXd::Zero(lb.getPaddingIdx());

    lb.backSubstitute(inc, l_diff);

    EXPECT_NEAR(feat_.estimated_depth, depth_before, 1e-10);
    EXPECT_NEAR(l_diff, 0.0, 1e-10);
}

TEST_F(LinearizeFixture, BackSubModelCostPositive) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    Eigen::VectorXd inc(lb.getPaddingIdx());
    inc.setRandom();
    inc *= 0.01;

    double l_diff = 0.0;
    lb.backSubstitute(inc, l_diff);

    EXPECT_TRUE(std::isfinite(l_diff));
}

TEST_F(LinearizeFixture, BackSubDepthUpdate) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    lb.performQR();

    feat_.estimated_depth = P_world_.z() * 0.9;
    LandmarkBlock lb2(MIN_DISTANCE, MAX_DISTANCE);
    lb2.allocate(&feat_, state_.get(), TrivialLoss{});
    lb2.linearize();
    lb2.performQR();

    double depth_before = feat_.estimated_depth;
    Eigen::VectorXd inc = Eigen::VectorXd::Zero(lb2.getPaddingIdx());
    double l_diff = 0.0;

    lb2.backSubstitute(inc, l_diff);

    EXPECT_GT(feat_.estimated_depth, depth_before);
}

// ── multi-block assembly ─────────────────────────────────────────────────────

TEST(LandmarkBlockTest, MultiBlockAssembly) {
    auto state = MakeState(5);
    Eigen::Vector3d P0(0.1, 0.05, 1.2);
    Eigen::Vector3d P1(-0.08, 0.03, 1.8);

    auto make_feat = [&](const Eigen::Vector3d& P) {
        std::vector<Eigen::Vector3d> uvs;
        for (int i = 0; i < 5; ++i) uvs.push_back(Project(P, *state, i));
        Eigen::Vector3d P_c0 = state->poses[0].get_pose().inverse() * P;
        double true_depth = P_c0.z();
        double true_u = P_c0.x() / P_c0.z(), true_v = P_c0.y() / P_c0.z();
        uvs[0] = Eigen::Vector3d(true_u, true_v, 1.0);
        return MakeFeature(0, uvs, true_depth);
    };

    auto f0 = make_feat(P0), f1 = make_feat(P1);
    LandmarkBlock lb0(MIN_DISTANCE, MAX_DISTANCE), lb1(MIN_DISTANCE, MAX_DISTANCE);
    lb0.allocate(&f0, state.get(), TrivialLoss{});
    lb1.allocate(&f1, state.get(), TrivialLoss{});
    lb0.linearize();
    lb0.performQR();
    lb1.linearize();
    lb1.performQR();

    int pdim = lb0.getPaddingIdx();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(pdim, pdim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(pdim);
    lb0.add_dense_H_b(H, b);
    lb1.add_dense_H_b(H, b);

    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    int positive = 0;
    for (int i = 0; i < pdim; ++i)
        if (es.eigenvalues()[i] > 1e-8) positive++;
    EXPECT_GE(positive, 5);
}

// ── full pipeline order ──────────────────────────────────────────────────────

TEST_F(LinearizeFixture, FullPipelineOrder) {
    LandmarkBlock lb(MIN_DISTANCE, MAX_DISTANCE);
    lb.allocate(&feat_, state_.get(), TrivialLoss{});
    lb.linearize();
    Eigen::VectorXd d2(lb.getPaddingIdx());
    d2.setZero();
    lb.addJp_diag2(d2);

    lb.performQR();
    lb.setLandmarkDamping(0.5);
    EXPECT_TRUE(lb.hasLandmarkDamping());
    lb.setLandmarkDamping(0.0);
    EXPECT_FALSE(lb.hasLandmarkDamping());

    Eigen::VectorXd inc = Eigen::VectorXd::Zero(lb.getPaddingIdx());
    double l_diff = 0.0;
    lb.backSubstitute(inc, l_diff);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Jacobian checker — systematic numerical check of compute_feature_linearization_block
// ═══════════════════════════════════════════════════════════════════════════════

struct VisualJacobianChecker : public tassel_tools::JacobianChecker {
    VisualJacobianChecker()
        : JacobianChecker(
              /*num_residuals=*/2,
              /*tangent dims=*/std::vector<int>{6, 6, 1},
              /*storage dims=*/std::vector<int>{7, 7, 1}) {}

    Eigen::Vector3d uv_host;
    Eigen::Vector3d uv_target;

    bool plus(const double* x, const double* delta, double* x_out, int block_idx) const override {
        if (block_idx < 2) {
            Eigen::Map<const Eigen::Vector3d> t(x);
            Eigen::Map<const Eigen::Vector3d> rho(delta);
            Eigen::Map<const Eigen::Vector3d> phi(delta + 3);

            Eigen::Map<const Eigen::Quaterniond> q(x + 3);
            Eigen::Matrix3d R = q.toRotationMatrix();

            Eigen::Map<Eigen::Vector3d> t_out(x_out);
            t_out = t + R * rho;

            Eigen::Quaterniond dq = Sophus::SO3d::exp(phi).unit_quaternion();
            Eigen::Map<Eigen::Quaterniond> q_out(x_out + 3);
            q_out = (q * dq).normalized();
            return true;
        } else {
            x_out[0] = x[0] + delta[0];
            return true;
        }
    }

    void evaluate(
        const std::vector<double*>& params, double* residuals, double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> host_t(params[0]);
        Eigen::Map<const Eigen::Quaterniond> host_q(params[0] + 3);
        Eigen::Map<const Eigen::Vector3d> target_t(params[1]);
        Eigen::Map<const Eigen::Quaterniond> target_q(params[1] + 3);
        double depth = *params[2];

        Eigen::Matrix3d host_R = host_q.toRotationMatrix();
        Eigen::Matrix3d target_R = target_q.toRotationMatrix();

        auto tangent_base = compute_tangent_base(uv_target);

        Sophus::SE3d T_w_h(host_R, host_t);
        Sophus::SE3d T_w_t(target_R, target_t);

        Eigen::Matrix<double, 2, 6> J_host, J_target;
        Eigen::Matrix<double, 2, 1> J_inv, res;
        compute_feature_linearization_block(
            T_w_h, T_w_t, depth, uv_host, uv_target, tangent_base, J_host, J_target, J_inv, res);

        Eigen::Map<Eigen::Vector2d> r_out(residuals);
        r_out = res;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_h_out(jacobians[0]);
                J_h_out = J_host;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_t_out(jacobians[1]);
                J_t_out = J_target;
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Vector2d> J_l_out(jacobians[2]);
                J_l_out = J_inv;
            }
        }
    }
};

void packPose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q, double* dst) {
    dst[0] = p.x();
    dst[1] = p.y();
    dst[2] = p.z();
    dst[3] = q.x();
    dst[4] = q.y();
    dst[5] = q.z();
    dst[6] = q.w();
}

void computeObservations(
    const Eigen::Vector3d& host_P, const Eigen::Quaterniond& host_Q,
    const Eigen::Vector3d& target_P, const Eigen::Quaterniond& target_Q, double depth,
    Eigen::Vector3d& uv_host, Eigen::Vector3d& uv_target) {
    Eigen::Matrix3d host_R = host_Q.toRotationMatrix();

    uv_host = Eigen::Vector3d(0.1, 0.1, 1.0);

    Eigen::Vector3d pt_in_H = uv_host * depth;
    Eigen::Vector3d pt_in_W = host_R * pt_in_H + host_P;
    uv_target = (target_Q.conjugate() * (pt_in_W - target_P)).normalized();
}

TEST(VisualJacobianTest, AllBlocksPass) {
    Eigen::Vector3d host_P(0, 0, 0);
    Eigen::Quaterniond host_Q = Eigen::Quaterniond::Identity();

    Eigen::Vector3d target_P(0.5, 0.02, 0.0);
    Eigen::Quaterniond target_Q(
        Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()));

    double depth = 3.0;  // 1/0.3333

    double pose_i[7], pose_j[7];
    packPose(host_P, host_Q, pose_i);
    packPose(target_P, target_Q, pose_j);

    VisualJacobianChecker checker;
    computeObservations(
        host_P, host_Q, target_P, target_Q, depth, checker.uv_host, checker.uv_target);
    checker.set_params({pose_i, pose_j, &depth});
    ASSERT_TRUE(checker.check());
}

TEST(VisualJacobianTest, NonTrivialPose) {
    Eigen::Vector3d host_P(1.2, -0.3, 0.5);
    Eigen::Quaterniond host_Q(
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitX()));

    Eigen::Vector3d target_P(1.8, 0.1, 0.7);
    Eigen::Quaterniond target_Q(
        Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()));

    double depth = 2.0;  // 1/0.5

    double pose_i[7], pose_j[7];
    packPose(host_P, host_Q, pose_i);
    packPose(target_P, target_Q, pose_j);

    VisualJacobianChecker checker;
    computeObservations(
        host_P, host_Q, target_P, target_Q, depth, checker.uv_host, checker.uv_target);
    checker.set_params({pose_i, pose_j, &depth});
    ASSERT_TRUE(checker.check());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Robust loss function tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(LossFunctionTest, TrivialLoss) {
    TrivialLoss loss;
    EXPECT_EQ(loss.rho(0.0), 0.0);
    EXPECT_NEAR(loss.rho(2.0), 2.0, 1e-14);  // 0.5 * 4 = 2
    EXPECT_NEAR(loss.weight(0.0), 1.0, 1e-14);
    EXPECT_NEAR(loss.weight(100.0), 1.0, 1e-14);  // always 1.0
    EXPECT_NEAR(loss.rho_prime(3.0), 3.0, 1e-14);
}

TEST(LossFunctionTest, TrivialLossDispatch) {
    LossVariant v = TrivialLoss{};
    EXPECT_NEAR(computeRho(v, 2.0), 2.0, 1e-14);
    EXPECT_NEAR(computeWeight(v, 5.0), 1.0, 1e-14);
}

TEST(LossFunctionTest, HuberLossInlier) {
    HuberLoss loss(1.345);
    double s = 0.5;
    // Inlier: s <= delta → quadratic
    EXPECT_NEAR(loss.rho(s), 0.5 * s * s, 1e-14);
    EXPECT_NEAR(loss.rho_prime(s), s, 1e-14);
    EXPECT_NEAR(loss.weight(s), 1.0, 1e-14);
}

TEST(LossFunctionTest, HuberLossOutlier) {
    HuberLoss loss(1.345);
    double s = 3.0;
    // Outlier: s > delta → linear
    EXPECT_NEAR(loss.rho(s), 1.345 * (3.0 - 0.5 * 1.345), 1e-12);
    EXPECT_NEAR(loss.rho_prime(s), 1.345, 1e-12);
    EXPECT_NEAR(loss.weight(s), 1.345 / 3.0, 1e-12);
}

TEST(LossFunctionTest, HuberLossContinuity) {
    HuberLoss loss(1.345);
    double delta = 1.345;
    // At the transition point, rho and rho_prime should be continuous
    EXPECT_NEAR(loss.rho(delta - 1e-6), loss.rho(delta + 1e-6), 3e-6);
    EXPECT_NEAR(loss.rho_prime(delta - 1e-6), loss.rho_prime(delta + 1e-6), 3e-6);
}

TEST(LossFunctionTest, HuberLossDispatch) {
    LossVariant v = HuberLoss{1.345};
    EXPECT_NEAR(computeRho(v, 0.5), 0.125, 1e-14);  // 0.5 * 0.5^2
    EXPECT_NEAR(computeWeight(v, 3.0), 1.345 / 3.0, 1e-12);
}

TEST(LossFunctionTest, CauchyLoss) {
    CauchyLoss loss(2.3849);
    double c = 2.3849;
    double s = 1.0;
    double expected_rho = 0.5 * c * c * std::log1p(s * s / (c * c));
    EXPECT_NEAR(loss.rho(s), expected_rho, 1e-12);
    EXPECT_NEAR(loss.rho_prime(s), s / (1.0 + s * s / (c * c)), 1e-12);
    EXPECT_NEAR(loss.weight(s), 1.0 / (1.0 + s * s / (c * c)), 1e-12);

    // weight decreases as residual grows
    EXPECT_NEAR(loss.weight(0.0), 1.0, 1e-12);
    EXPECT_LT(loss.weight(10.0), 1.0);
    EXPECT_GT(loss.weight(10.0), 0.0);
}

TEST(LossFunctionTest, CauchyLossDispatch) {
    LossVariant v = CauchyLoss{2.3849};
    double c = 2.3849;
    double s = 1.0;
    EXPECT_NEAR(computeRho(v, s), 0.5 * c * c * std::log1p(s * s / (c * c)), 1e-12);
    EXPECT_NEAR(computeWeight(v, s), 1.0 / (1.0 + s * s / (c * c)), 1e-12);
}

TEST(LossFunctionTest, TukeyLoss) {
    TukeyLoss loss(4.6851);
    double c = 4.6851;

    // Inlier
    double s = 2.0;
    double t = s * s / (c * c);
    double expected_rho = (c * c / 6.0) * (1.0 - std::pow(1.0 - t, 3));
    EXPECT_NEAR(loss.rho(s), expected_rho, 1e-12);
    EXPECT_NEAR(loss.rho_prime(s), s * (1.0 - t) * (1.0 - t), 1e-12);
    EXPECT_NEAR(loss.weight(s), (1.0 - t) * (1.0 - t), 1e-12);

    // Outlier (completely rejected)
    double big = 2.0 * c;
    EXPECT_NEAR(loss.rho(big), c * c / 6.0, 1e-12);
    EXPECT_NEAR(loss.rho_prime(big), 0.0, 1e-12);
    EXPECT_NEAR(loss.weight(big), 0.0, 1e-12);
}

TEST(LossFunctionTest, TukeyLossWeightBoundary) {
    TukeyLoss loss(4.6851);
    // At s=c, weight should be 0
    EXPECT_NEAR(loss.weight(loss.c), 0.0, 1e-12);
    EXPECT_NEAR(loss.rho_prime(loss.c), 0.0, 1e-12);
    // At s=0, weight should be 1
    EXPECT_NEAR(loss.weight(0.0), 1.0, 1e-12);
}

TEST(LossFunctionTest, TukeyLossDispatch) {
    LossVariant v = TukeyLoss{4.6851};
    double c = 4.6851;
    double s = 2.0;
    double t = s * s / (c * c);
    EXPECT_NEAR(computeRho(v, s), (c * c / 6.0) * (1.0 - std::pow(1.0 - t, 3)), 1e-12);
    EXPECT_NEAR(computeWeight(v, s), (1.0 - t) * (1.0 - t), 1e-12);
    EXPECT_NEAR(computeWeight(v, 2.0 * c), 0.0, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loss integrated with LandmarkBlock
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(LinearizeFixture, HuberLossReducesWeights) {
    LandmarkBlock lb_trivial(MIN_DISTANCE, MAX_DISTANCE), lb_huber(MIN_DISTANCE, MAX_DISTANCE);

    // Extreme depth error to produce residuals well beyond Huber threshold
    feat_.estimated_depth = P_world_.z() * 0.01;  // 100x too small → huge residuals

    lb_trivial.allocate(&feat_, state_.get(), TrivialLoss{});
    double err_trivial = lb_trivial.linearize();
    double norm_trivial = lb_trivial.getStorage()
                              .block(0, 0, lb_trivial.getNumRows() - 1, lb_trivial.getPaddingIdx())
                              .norm();

    lb_huber.allocate(&feat_, state_.get(), HuberLoss{0.01});
    double err_huber = lb_huber.linearize();
    double norm_huber = lb_huber.getStorage()
                            .block(0, 0, lb_huber.getNumRows() - 1, lb_huber.getPaddingIdx())
                            .norm();

    // Huber should damp large residuals → lower error and smaller Jacobian norm
    EXPECT_LT(err_huber, err_trivial);
    EXPECT_LT(norm_huber, norm_trivial);
}

TEST(LossFunctionTest, VariantAssignment) {
    // Verify std::variant can hold and switch between loss types
    LossVariant v = TrivialLoss{};
    EXPECT_NEAR(computeWeight(v, 5.0), 1.0, 1e-14);

    v = HuberLoss{1.0};
    EXPECT_NEAR(computeWeight(v, 2.0), 0.5, 1e-12);  // delta/s = 1/2
    EXPECT_NEAR(computeWeight(v, 0.5), 1.0, 1e-12);

    v = CauchyLoss{2.0};
    EXPECT_NEAR(computeWeight(v, 0.0), 1.0, 1e-12);
    EXPECT_LT(computeWeight(v, 4.0), 1.0);

    v = TukeyLoss{4.6851};
    EXPECT_NEAR(computeWeight(v, 0.0), 1.0, 1e-12);
    EXPECT_NEAR(computeWeight(v, 10.0), 0.0, 1e-12);
}

}  // namespace
}  // namespace tassel_core
