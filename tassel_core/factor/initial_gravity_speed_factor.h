#ifndef TASSEL_CORE_FACTOR_INITIAL_GRAVITY_SPEED_FACTOR_H_
#define TASSEL_CORE_FACTOR_INITIAL_GRAVITY_SPEED_FACTOR_H_

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace tassel_core {
namespace detail {

// Shared evaluator for the V+g initialization residual.
// 3 parameter blocks: V_i[3], V_j[3], w[2] → 6 residuals [rp, rv]
struct GravitySpeedEvaluator {
    Eigen::Vector3d dp_meas, dv_meas;
    Eigen::Vector3d P_i, P_j;
    Eigen::Matrix3d R_i_T;
    Eigen::Matrix<double, 2, 3> tangent_base;
    Eigen::Vector3d g0_dir;
    double g_mag, dt;

    GravitySpeedEvaluator(
        const Eigen::Vector3d& dp, const Eigen::Vector3d& dv, const Eigen::Vector3d& pi,
        const Eigen::Vector3d& pj, const Eigen::Matrix3d& Ri, const Eigen::Matrix<double, 2, 3>& tb,
        const Eigen::Vector3d& gd, double gm, double d)
        : dp_meas(dp),
          dv_meas(dv),
          P_i(pi),
          P_j(pj),
          R_i_T(Ri.transpose()),
          tangent_base(tb),
          g0_dir(gd),
          g_mag(gm),
          dt(d) {}

    bool operator()(
        const double* V_i, const double* V_j, const double* w, double* residuals,
        double** jacobians) const {
        Eigen::Map<const Eigen::Vector3d> v_i(V_i), v_j(V_j);
        Eigen::Vector2d w_vec(w[0], w[1]);

        // Gravity on sphere: g = g_mag * (g0_dir + T^T*w) / ||g0_dir + T^T*w||
        Eigen::Vector3d h = g0_dir + tangent_base.transpose() * w_vec;
        double h_norm = h.norm();
        Eigen::Vector3d g_hat = h / h_norm;
        Eigen::Vector3d g_vec = g_mag * g_hat;

        // dg/dw = g_mag / ||h|| * (I - g_hat*g_hat^T) * T^T   (3x2)
        Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - g_hat * g_hat.transpose();
        Eigen::Matrix<double, 3, 2> dg_dw = (g_mag / h_norm) * P * tangent_base.transpose();

        double dt2 = 0.5 * dt * dt;

        // Residuals: [rp, rv]
        Eigen::Map<Eigen::Matrix<double, 6, 1>> res(residuals);
        res.block<3, 1>(0, 0) = R_i_T * (P_j - P_i - v_i * dt - dt2 * g_vec) - dp_meas;
        res.block<3, 1>(3, 0) = R_i_T * (v_j - v_i - dt * g_vec) - dv_meas;

        if (jacobians) {
            Eigen::Matrix<double, 3, 2> J_dp_dw = -dt2 * R_i_T * dg_dw;
            Eigen::Matrix<double, 3, 2> J_dv_dw = -dt * R_i_T * dg_dw;

            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                J.block<3, 3>(0, 0) = -dt * R_i_T;
                J.block<3, 3>(3, 0) = -R_i_T;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                J.block<3, 3>(3, 0) = R_i_T;
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 6, 2, Eigen::RowMajor>> J(jacobians[2]);
                J.setZero();
                J.block<3, 2>(0, 0) = J_dp_dw;
                J.block<3, 2>(3, 0) = J_dv_dw;
            }
        }
        return true;
    }
};

}  // namespace detail

// ── InitialGravitySpeedFactor ──────────────────────────────────────────────
//
// SizedCostFunction<6, 3, 3, 2>: 6 residuals, V_i[3], V_j[3], w[2]
//
// 6 residuals [rp, rv]:
//   rp = R_i^T * (P_j - P_i - V_i*dt - 0.5*g*dt^2) - dp
//   rv = R_i^T * (V_j - V_i - g*dt) - dv
//
// Gravity is parameterized on a sphere to preserve magnitude:
//   g = g_mag * (g0_dir + T*w) / ||g0_dir + T*w||

class InitialGravitySpeedFactor : public ceres::SizedCostFunction<6, 3, 3, 2> {
public:
    InitialGravitySpeedFactor(
        const Eigen::Vector3d& dp, const Eigen::Vector3d& dv, const Eigen::Vector3d& pi,
        const Eigen::Vector3d& pj, const Eigen::Matrix3d& Ri, const Eigen::Matrix<double, 2, 3>& tb,
        const Eigen::Vector3d& gd, double gm, double d)
        : eval_(dp, dv, pi, pj, Ri, tb, gd, gm, d) {}

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        return eval_(parameters[0], parameters[1], parameters[2], residuals, jacobians);
    }

private:
    detail::GravitySpeedEvaluator eval_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_FACTOR_INITIAL_GRAVITY_SPEED_FACTOR_H_
