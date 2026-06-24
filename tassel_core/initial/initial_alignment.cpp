#include "initial/initial_alignment.h"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cstddef>
#include <limits>
#include <sophus/so3.hpp>
#include <vector>

#include "factor/visual_factor.h"

namespace tassel_core {

bool linearAlignment(
    std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps,
    std::vector<Eigen::Vector3d>& Vs, const std::vector<Eigen::Vector3d>& delta_vs,
    const std::vector<Eigen::Vector3d>& delta_ps, const std::vector<double>& dts,
    Eigen::Vector3d& final_g, double& s, const Eigen::Matrix3d ric, const Eigen::Vector3d tic,
    double g_norm_thres, double target_g_norm) {
    int n_frames = static_cast<int>(Rs.size());
    int n_state = n_frames * 3 + 4;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_state, n_state);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n_state);
    Eigen::Matrix3d ricT = ric.transpose();

    for (int i = 0; i < n_frames - 1; ++i) {
        int j = i + 1;
        double dt = dts[i];
        Eigen::Vector3d delta_p = delta_ps[i];
        Eigen::Vector3d delta_v = delta_vs[i];

        Eigen::Matrix3d Ri = Rs[i];
        Eigen::Matrix3d Rj = Rs[j];
        Eigen::Vector3d Pi = Ps[i];
        Eigen::Vector3d Pj = Ps[j];

        int col_Vi = i * 3;

        Eigen::Matrix<double, 6, 10> tmp_A;
        tmp_A.setZero();
        Eigen::Matrix<double, 6, 1> tmp_b;
        tmp_b.setZero();

        tmp_A.block<3, 3>(0, 0) = -dt * ric * Rs[i].transpose() * ric.transpose();
        tmp_A.block<3, 3>(0, 6) = 0.5 * dt * dt * ric * Rs[i].transpose() * ric.transpose();
        tmp_A.block<3, 1>(0, 9) = ric * Rs[i].transpose() * (Pj - Pi) / 100.0;
        tmp_b.block<3, 1>(0, 0) =
            ric * Ri.transpose() * Rj * ric.transpose() * tic - ric * tic + delta_p;

        tmp_A.block<3, 3>(3, 0) = -ric * Rs[i].transpose() * ric.transpose();
        tmp_A.block<3, 3>(3, 3) = ric * Rs[i].transpose() * ric.transpose();
        tmp_A.block<3, 3>(3, 6) = ric * Rs[i].transpose() * ric.transpose() * dt;
        tmp_b.block<3, 1>(3, 0) = delta_v;

        Eigen::Matrix<double, 10, 10> r_A = tmp_A.transpose() * tmp_A;
        Eigen::Matrix<double, 10, 1> r_b = tmp_A.transpose() * tmp_b;

        A.block<6, 6>(col_Vi, col_Vi) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(col_Vi) += r_b.head<6>();
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();
        A.block<6, 4>(col_Vi, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, col_Vi) += r_A.bottomLeftCorner<4, 6>();
    }

    A = A * 1000.0;
    b = b * 1000.0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    const Eigen::VectorXd singular_values = svd.singularValues();
    const double max_singular = singular_values.size() > 0 ? singular_values(0) : 0.0;
    const double min_singular =
        singular_values.size() > 0 ? singular_values(singular_values.size() - 1) : 0.0;
    const double rank_tolerance =
        std::numeric_limits<double>::epsilon() * static_cast<double>(A.rows()) * max_singular;
    const int rank = static_cast<int>((singular_values.array() > rank_tolerance).count());
    if (rank < A.rows()) {
        spdlog::warn(
            "LinearAlignment: no unique solution, A is rank deficient rank={}/{} "
            "sigma_min={:.6e} sigma_max={:.6e} tol={:.6e}",
            rank, A.rows(), min_singular, max_singular, rank_tolerance);
    }

    Eigen::VectorXd x = A.ldlt().solve(b);

    s = x(n_state - 1) / 100.0;
    final_g = x.segment<3>(n_state - 4);

    spdlog::info(
        "LinearAlignment: |g|={:.4f} g=({:.3f},{:.3f},{:.3f}) s={:.4f}", final_g.norm(),
        final_g.x(), final_g.y(), final_g.z(), s);
    if (s <= 0 || std::abs(final_g.norm() - target_g_norm) > g_norm_thres) {
        return false;
    }

    for (int i = 0; i < n_frames; ++i) {
        Vs[i] = x.segment<3>(i * 3);
    }

    return true;
}

void refineGravitySpeeds(
    std::vector<Eigen::Vector3d>& Vs, const std::vector<Eigen::Matrix3d>& Rs,
    const std::vector<Eigen::Vector3d>& Ps, const std::vector<Eigen::Vector3d>& delta_vs,
    const std::vector<Eigen::Vector3d>& delta_ps, const std::vector<double>& dts,
    Eigen::Vector3d& G, double& s, const Eigen::Matrix3d ric, const Eigen::Vector3d tic,
    double g_mag) {
    int n_frames = static_cast<int>(Vs.size());
    int n_state = n_frames * 3 + 3;
    int col_dg = n_frames * 3;
    int col_s = n_frames * 3 + 2;

    Eigen::Vector3d g0_dir = G.normalized();

    for (int iter = 0; iter < 4; ++iter) {
        Eigen::Matrix<double, 2, 3> T = computeTangentBasis(g0_dir);
        Eigen::Matrix<double, 3, 2> L = g_mag * T.transpose();
        Eigen::Vector3d g0 = g_mag * g0_dir;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_state, n_state);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_state);

        for (int i = 0; i < n_frames - 1; ++i) {
            int j = i + 1;
            double dt = dts[i];
            double dt2 = 0.5 * dt * dt;

            Eigen::Matrix3d R = ric * Rs[i].transpose() * ric.transpose();
            Eigen::Matrix3d R_trans = ric * Rs[i].transpose();
            Eigen::Matrix<double, 3, 2> R_L = R * L;

            int col_Vi = i * 3;
            int col_Vj = j * 3;

            Eigen::Matrix<double, 6, 9> tmp_A;
            tmp_A.setZero();

            tmp_A.block<3, 3>(0, 0) = -dt * R;
            tmp_A.block<3, 2>(0, 6) = dt2 * R_L;
            tmp_A.block<3, 1>(0, 8) = R_trans * (Ps[j] - Ps[i]) / 100.0;

            tmp_A.block<3, 3>(3, 0) = -R;
            tmp_A.block<3, 3>(3, 3) = R;
            tmp_A.block<3, 2>(3, 6) = dt * R_L;

            Eigen::Matrix<double, 6, 1> tmp_b;
            tmp_b.block<3, 1>(0, 0) =
                delta_ps[i] + R_trans * Rs[j] * ric.transpose() * tic - ric * tic - dt2 * R * g0;
            tmp_b.block<3, 1>(3, 0) = delta_vs[i] - dt * R * g0;

            Eigen::Matrix<double, 9, 9> r_A = tmp_A.transpose() * tmp_A;
            Eigen::Matrix<double, 9, 1> r_b = tmp_A.transpose() * tmp_b;

            A.block<3, 3>(col_Vi, col_Vi) += r_A.block<3, 3>(0, 0);
            A.block<3, 3>(col_Vj, col_Vj) += r_A.block<3, 3>(3, 3);
            b.segment<3>(col_Vi) += r_b.segment<3>(0);
            b.segment<3>(col_Vj) += r_b.segment<3>(3);
            A.block<3, 3>(col_Vi, col_Vj) += r_A.block<3, 3>(0, 3);
            A.block<3, 3>(col_Vj, col_Vi) += r_A.block<3, 3>(3, 0);

            A.block<2, 2>(col_dg, col_dg) += r_A.block<2, 2>(6, 6);
            A(col_s, col_s) += r_A(8, 8);
            A.block<2, 1>(col_dg, col_s) += r_A.block<2, 1>(6, 8);
            A.block<1, 2>(col_s, col_dg) += r_A.block<1, 2>(8, 6);
            b.segment<2>(col_dg) += r_b.segment<2>(6);
            b(col_s) += r_b(8);

            A.block<3, 2>(col_Vi, col_dg) += r_A.block<3, 2>(0, 6);
            A.block<2, 3>(col_dg, col_Vi) += r_A.block<2, 3>(6, 0);
            A.block<3, 1>(col_Vi, col_s) += r_A.block<3, 1>(0, 8);
            A.block<1, 3>(col_s, col_Vi) += r_A.block<1, 3>(8, 0);
            A.block<3, 2>(col_Vj, col_dg) += r_A.block<3, 2>(3, 6);
            A.block<2, 3>(col_dg, col_Vj) += r_A.block<2, 3>(6, 3);
            A.block<3, 1>(col_Vj, col_s) += r_A.block<3, 1>(3, 8);
            A.block<1, 3>(col_s, col_Vj) += r_A.block<1, 3>(8, 3);
        }

        A = A * 1000.0;
        b = b * 1000.0;
        Eigen::VectorXd x = A.ldlt().solve(b);

        Eigen::Vector2d w(x[col_dg], x[col_dg + 1]);
        Eigen::Vector3d dg = T.transpose() * w;
        g0_dir = (g0_dir + dg).normalized();

        for (int i = 0; i < n_frames; ++i) Vs[i] = x.segment<3>(i * 3);
        s = x(col_s) / 100.0;
    }

    G = g_mag * g0_dir;
}

Eigen::Vector3d solveGyroBias(
    std::vector<Eigen::Matrix3d> Rs, std::vector<Eigen::Matrix3d> dq_dbgs,
    std::vector<Eigen::Matrix3d> delta_qs, Eigen::Matrix3d ric) {
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A.setZero();
    b.setZero();

    for (size_t i = 0; i < dq_dbgs.size(); ++i) {
        Eigen::Matrix3d R_Bi_Bj_vis = ric * Rs[i].transpose() * Rs[i + 1] * ric.transpose();
        Eigen::Quaterniond q_ij(R_Bi_Bj_vis);
        q_ij.normalize();
        R_Bi_Bj_vis = q_ij.toRotationMatrix();

        Eigen::Quaterniond q_delta(delta_qs[i]);
        q_delta.normalize();
        Eigen::Matrix3d delta_q = q_delta.toRotationMatrix();

        Eigen::Matrix3d R_diff = delta_q.transpose() * R_Bi_Bj_vis;
        Eigen::Quaterniond q_diff(R_diff);
        q_diff.normalize();
        Eigen::Vector3d phi = Sophus::SO3d(q_diff).log();

        A += dq_dbgs[i].transpose() * dq_dbgs[i];
        b += dq_dbgs[i].transpose() * phi;
    }

    Eigen::Vector3d bg = A.ldlt().solve(b);
    spdlog::info("Gyro bias: ({:.6f}, {:.6f}, {:.6f})", bg.x(), bg.y(), bg.z());
    return bg;
}

}  // namespace tassel_core
