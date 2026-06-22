#ifndef TASSEL_CORE_FACTOR_REPROJECTION_FACTOR_H_
#define TASSEL_CORE_FACTOR_REPROJECTION_FACTOR_H_

#include <Eigen/Core>

#include <ceres/rotation.h>
#include <ceres/sized_cost_function.h>

namespace tassel_core {

class ReprojectionFactor : public ceres::SizedCostFunction<2, 4, 3, 3> {
public:
    ReprojectionFactor(double obs_u, double obs_v) : obs_u_(obs_u), obs_v_(obs_v) {}

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        const double* q = parameters[0];  // [w, x, y, z]
        const double* t = parameters[1];  // [tx, ty, tz]
        const double* X = parameters[2];  // [X, Y, Z]

        double p[3];
        ceres::QuaternionRotatePoint(q, X, p);
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        double inv_z = 1.0 / p[2];
        double u = p[0] * inv_z;
        double v = p[1] * inv_z;

        residuals[0] = u - obs_u_;
        residuals[1] = v - obs_v_;

        if (!jacobians) return true;

        // dr/dp: Jacobian of normalized projection w.r.t. camera-frame point (2x3)
        Eigen::Matrix<double, 2, 3> drdp;
        drdp << inv_z, 0.0, -p[0] * inv_z * inv_z, 0.0, inv_z, -p[1] * inv_z * inv_z;

        // ── dr/dt = dr/dp * I ──
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jt(jacobians[1]);
            Jt = drdp;
        }

        // ── dr/dX = dr/dp * R(q) ──
        if (jacobians[2]) {
            double R[9];
            ceres::QuaternionToRotation(q, R);
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jx(jacobians[2]);
            Jx(0, 0) = drdp(0, 0) * R[0] + drdp(0, 1) * R[3] + drdp(0, 2) * R[6];
            Jx(0, 1) = drdp(0, 0) * R[1] + drdp(0, 1) * R[4] + drdp(0, 2) * R[7];
            Jx(0, 2) = drdp(0, 0) * R[2] + drdp(0, 1) * R[5] + drdp(0, 2) * R[8];
            Jx(1, 0) = drdp(1, 0) * R[0] + drdp(1, 1) * R[3] + drdp(1, 2) * R[6];
            Jx(1, 1) = drdp(1, 0) * R[1] + drdp(1, 1) * R[4] + drdp(1, 2) * R[7];
            Jx(1, 2) = drdp(1, 0) * R[2] + drdp(1, 1) * R[5] + drdp(1, 2) * R[8];
        }

        // ── dr/dq = dr/dp * ∂(R(q)*X)/∂q ──
        if (jacobians[0]) {
            double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
            double Px = X[0], Py = X[1], Pz = X[2];

            // ∂(R*X)/∂q, 3×4 matrix, element-wise analytic derivatives
            double Jpq_00 = 2 * (qy * Pz - qz * Py);
            double Jpq_01 = 2 * (qy * Py + qz * Pz);
            double Jpq_02 = 2 * (-2 * qy * Px + qx * Py + qw * Pz);
            double Jpq_03 = 2 * (-2 * qz * Px - qw * Py + qx * Pz);

            double Jpq_10 = 2 * (qz * Px - qx * Pz);
            double Jpq_11 = 2 * (qy * Px - 2 * qx * Py - qw * Pz);
            double Jpq_12 = 2 * (qx * Px + qz * Pz);
            double Jpq_13 = 2 * (qw * Px - 2 * qz * Py + qy * Pz);

            double Jpq_20 = 2 * (qx * Py - qy * Px);
            double Jpq_21 = 2 * (qz * Px + qw * Py - 2 * qx * Pz);
            double Jpq_22 = 2 * (-qw * Px + qz * Py - 2 * qy * Pz);
            double Jpq_23 = 2 * (qx * Px + qy * Py);

            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> Jq(jacobians[0]);
            Jq(0, 0) = drdp(0, 0) * Jpq_00 + drdp(0, 1) * Jpq_10 + drdp(0, 2) * Jpq_20;
            Jq(0, 1) = drdp(0, 0) * Jpq_01 + drdp(0, 1) * Jpq_11 + drdp(0, 2) * Jpq_21;
            Jq(0, 2) = drdp(0, 0) * Jpq_02 + drdp(0, 1) * Jpq_12 + drdp(0, 2) * Jpq_22;
            Jq(0, 3) = drdp(0, 0) * Jpq_03 + drdp(0, 1) * Jpq_13 + drdp(0, 2) * Jpq_23;
            Jq(1, 0) = drdp(1, 0) * Jpq_00 + drdp(1, 1) * Jpq_10 + drdp(1, 2) * Jpq_20;
            Jq(1, 1) = drdp(1, 0) * Jpq_01 + drdp(1, 1) * Jpq_11 + drdp(1, 2) * Jpq_21;
            Jq(1, 2) = drdp(1, 0) * Jpq_02 + drdp(1, 1) * Jpq_12 + drdp(1, 2) * Jpq_22;
            Jq(1, 3) = drdp(1, 0) * Jpq_03 + drdp(1, 1) * Jpq_13 + drdp(1, 2) * Jpq_23;
        }

        return true;
    }

private:
    double obs_u_, obs_v_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_REPROJECTION_FACTOR_H_
