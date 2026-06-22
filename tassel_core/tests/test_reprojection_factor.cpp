// =============================================================================
// test_reprojection_factor.cpp — ReprojectionFactor Jacobian verification
// =============================================================================
//
// 随机生成相机位姿 + 3D点 → 投影得到观测 →
// 对 q 使用 QuaternionManifold::Plus 在切空间做数值微分，
// 对 t、X 使用欧氏扰动，验证手推解析雅各比
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <random>

#include <ceres/ceres.h>

#include "factor/reprojection_factor.h"

namespace tassel_core {
namespace {

TEST(ReprojectionFactorTest, JacobianCheck) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    std::uniform_real_distribution<double> ddist(0.5, 8.0);

    const double eps = 1e-7;
    const double tol = 1e-5;
    int nbad = 0;

    ceres::QuaternionManifold q_man;

    for (int trial = 0; trial < 20; ++trial) {
        Eigen::Quaterniond q(ud(rng), ud(rng), ud(rng), ud(rng));
        q.normalize();
        Eigen::Vector3d t(ud(rng) * 2.0, ud(rng) * 2.0, ddist(rng));
        Eigen::Vector3d X(ud(rng) * 2.0, ud(rng) * 2.0, ddist(rng));

        Eigen::Vector3d pc = q * X + t;
        if (pc.z() < 0.2) continue;
        double obs_u = pc.x() / pc.z();
        double obs_v = pc.y() / pc.z();

        double q_arr[4] = {q.w(), q.x(), q.y(), q.z()};
        double t_arr[3] = {t.x(), t.y(), t.z()};
        double X_arr[3] = {X.x(), X.y(), X.z()};
        const double* params[] = {q_arr, t_arr, X_arr};

        ReprojectionFactor factor(obs_u, obs_v);

        // analytic Jacobians (ambient: 2x4, 2x3, 2x3)
        double r_an[2];
        double Jq_an[2 * 4], Jt_an[2 * 3], Jx_an[2 * 3];
        double* jacs_an[] = {Jq_an, Jt_an, Jx_an};
        factor.Evaluate(params, r_an, jacs_an);

        // numeric Jacobians on tangent space / Euclidean
        // quaternion: tangent dim=3, numeric on manifold
        Eigen::Matrix<double, 2, 3> Jq_num;
        for (int c = 0; c < 3; ++c) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(c) = eps;
            double q_plus[4], q_minus[4];
            q_man.Plus(q_arr, delta.data(), q_plus);
            q_man.Plus(q_arr, (-delta).eval().data(), q_minus);
            const double* pp[3] = {q_plus, t_arr, X_arr};
            const double* pm[3] = {q_minus, t_arr, X_arr};
            double rp[2], rm[2];
            factor.Evaluate(pp, rp, nullptr);
            factor.Evaluate(pm, rm, nullptr);
            Jq_num(0, c) = (rp[0] - rm[0]) / (2.0 * eps);
            Jq_num(1, c) = (rp[1] - rm[1]) / (2.0 * eps);
        }

        // compare: multiply analytic ambient Jq (2x4) by PlusJacobian (4x3)
        double J_plus[12];
        q_man.PlusJacobian(q_arr, J_plus);
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> Pj(J_plus);
        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> Jq_amb(Jq_an);
        Eigen::Matrix<double, 2, 3> Jq_tan = Jq_amb * Pj;

        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 3; ++c) {
                double an = Jq_tan(r, c);
                double num = Jq_num(r, c);
                double scale = std::max({std::abs(an), std::abs(num), 1e-8});
                if (std::abs(an - num) / scale > tol) ++nbad;
            }
        }

        // translation: Euclidean, compare directly
        for (int c = 0; c < 3; ++c) {
            double tp[3] = {t_arr[0], t_arr[1], t_arr[2]};
            double tm[3] = {t_arr[0], t_arr[1], t_arr[2]};
            tp[c] += eps;
            tm[c] -= eps;
            const double* pp[3] = {q_arr, tp, X_arr};
            const double* pm[3] = {q_arr, tm, X_arr};
            double rp[2], rm[2];
            factor.Evaluate(pp, rp, nullptr);
            factor.Evaluate(pm, rm, nullptr);
            for (int r = 0; r < 2; ++r) {
                double an = Jt_an[r * 3 + c];
                double num = (rp[r] - rm[r]) / (2.0 * eps);
                double scale = std::max({std::abs(an), std::abs(num), 1e-8});
                if (std::abs(an - num) / scale > tol) ++nbad;
            }
        }

        // point: Euclidean, compare directly
        for (int c = 0; c < 3; ++c) {
            double Xp[3] = {X_arr[0], X_arr[1], X_arr[2]};
            double Xm[3] = {X_arr[0], X_arr[1], X_arr[2]};
            Xp[c] += eps;
            Xm[c] -= eps;
            const double* pp[3] = {q_arr, t_arr, Xp};
            const double* pm[3] = {q_arr, t_arr, Xm};
            double rp[2], rm[2];
            factor.Evaluate(pp, rp, nullptr);
            factor.Evaluate(pm, rm, nullptr);
            for (int r = 0; r < 2; ++r) {
                double an = Jx_an[r * 3 + c];
                double num = (rp[r] - rm[r]) / (2.0 * eps);
                double scale = std::max({std::abs(an), std::abs(num), 1e-8});
                if (std::abs(an - num) / scale > tol) ++nbad;
            }
        }
    }

    EXPECT_EQ(nbad, 0) << nbad << " Jacobian elements exceed tolerance " << tol;
}

}  // namespace
}  // namespace tassel_core
