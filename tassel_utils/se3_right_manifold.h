#ifndef TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_
#define TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_

#include <ceres/manifold.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>

namespace tassel_core {

class SE3RightManifold : public ceres::Manifold {
public:
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        Eigen::Vector3d P(x[0], x[1], x[2]);
        Eigen::Vector3d phi(x[3], x[4], x[5]);
        Eigen::Vector3d dP(delta[0], delta[1], delta[2]);
        Eigen::Vector3d dphi(delta[3], delta[4], delta[5]);

        Sophus::SO3d R = Sophus::SO3d::exp(phi);
        Sophus::SO3d R_new = R * Sophus::SO3d::exp(dphi);
        Eigen::Quaterniond q_new = R_new.unit_quaternion();
        q_new.normalize();
        Eigen::Vector3d phi_new = Sophus::SO3d(q_new).log();
        Eigen::Vector3d P_new = P + dP;

        for (int i = 0; i < 3; ++i) x_plus_delta[i] = P_new(i);
        for (int i = 0; i < 3; ++i) x_plus_delta[3 + i] = phi_new(i);
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
        std::fill_n(jacobian, 36, 0.0);

        Eigen::Vector3d phi(x[3], x[4], x[5]);
        Eigen::Matrix3d Jr_inv = Sophus::SO3d::leftJacobianInverse(-phi);

        for (int i = 0; i < 3; ++i) jacobian[i * 6 + i] = 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) jacobian[(3 + i) * 6 + (3 + j)] = Jr_inv(i, j);
        return true;
    }

    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        Eigen::Vector3d P_y(y[0], y[1], y[2]);
        Eigen::Vector3d phi_y(y[3], y[4], y[5]);
        Eigen::Vector3d P_x(x[0], x[1], x[2]);
        Eigen::Vector3d phi_x(x[3], x[4], x[5]);

        Sophus::SO3d R_x = Sophus::SO3d::exp(phi_x);
        Sophus::SO3d R_y = Sophus::SO3d::exp(phi_y);
        Eigen::Vector3d dphi = (R_x.inverse() * R_y).log();
        Eigen::Vector3d dP = P_y - P_x;

        for (int i = 0; i < 3; ++i) y_minus_x[i] = dP(i);
        for (int i = 0; i < 3; ++i) y_minus_x[3 + i] = dphi(i);
        return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override {
        std::fill_n(jacobian, 36, 0.0);
        Eigen::Vector3d phi(x[3], x[4], x[5]);
        Eigen::Matrix3d Jr = Sophus::SO3d::leftJacobian(-phi);

        for (int i = 0; i < 3; ++i) jacobian[i * 6 + i] = 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) jacobian[(3 + i) * 6 + (3 + j)] = Jr(i, j);
        return true;
    }

    int AmbientSize() const override { return 6; }
    int TangentSize() const override { return 6; }
};

}  // namespace tassel_core
#endif /* TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_ */
