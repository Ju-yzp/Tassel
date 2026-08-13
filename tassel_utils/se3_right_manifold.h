#ifndef TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_
#define TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_

#include <ceres/manifold.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <stdexcept>

#include <sophus/so3.hpp>

namespace tassel_core {

// 统一先验切空间契约：姿态采用右扰动 R_new = R_old * Exp(delta_theta)。
// 返回 y ⊟ x = [p_y-p_x, Log(R_x^{-1}R_y)]，其旋转坐标在 x 的局部右切空间。
inline Eigen::Matrix<double, 6, 1> rightTangentDelta(
    const Eigen::Matrix<double, 6, 1>& x, const Eigen::Matrix<double, 6, 1>& y) {
    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = y.head<3>() - x.head<3>();
    delta.tail<3>() =
        (Sophus::SO3d::exp(x.tail<3>()).inverse() * Sophus::SO3d::exp(y.tail<3>())).log();
    return delta;
}

// 将旧线性化点的旋转切空间坐标转换到新线性化点：delta_old = T * delta_new。
inline Eigen::Matrix3d rightTangentTransport(
    const Eigen::Vector3d& old_phi, const Eigen::Vector3d& new_phi) {
    const Eigen::Vector3d delta =
        (Sophus::SO3d::exp(old_phi).inverse() * Sophus::SO3d::exp(new_phi)).log();
    if (!delta.allFinite() || delta.norm() >= 3.14159265358979323846 - 1e-6) {
        throw std::invalid_argument("Right tangent rotation transport is invalid");
    }
    return Sophus::SO3d::leftJacobianInverse(-delta);
}

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

        for (int i = 0; i < 3; ++i) {
            x_plus_delta[i] = P_new(i);
        }
        for (int i = 0; i < 3; ++i) {
            x_plus_delta[3 + i] = phi_new(i);
        }
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
        std::fill_n(jacobian, 36, 0.0);

        Eigen::Vector3d phi(x[3], x[4], x[5]);
        Eigen::Matrix3d Jr_inv = Sophus::SO3d::leftJacobianInverse(-phi);

        for (int i = 0; i < 3; ++i) {
            jacobian[i * 6 + i] = 1.0;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                jacobian[(3 + i) * 6 + (3 + j)] = Jr_inv(i, j);
            }
        }
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

        for (int i = 0; i < 3; ++i) {
            y_minus_x[i] = dP(i);
        }
        for (int i = 0; i < 3; ++i) {
            y_minus_x[3 + i] = dphi(i);
        }
        return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override {
        std::fill_n(jacobian, 36, 0.0);
        Eigen::Vector3d phi(x[3], x[4], x[5]);
        Eigen::Matrix3d Jr = Sophus::SO3d::leftJacobian(-phi);

        for (int i = 0; i < 3; ++i) {
            jacobian[i * 6 + i] = 1.0;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                jacobian[(3 + i) * 6 + (3 + j)] = Jr(i, j);
            }
        }
        return true;
    }

    int AmbientSize() const override { return 6; }
    int TangentSize() const override { return 6; }
};

}  // namespace tassel_core
#endif /* TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_ */
