#ifndef TASSEL_UTILS_ROTATION_H_
#define TASSEL_UTILS_ROTATION_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>

namespace tassel_utils {

inline Eigen::Matrix3d normalizeRot(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q(R);
    q.normalize();
    return q.toRotationMatrix();
}

inline Eigen::Vector3d rotDiff(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) {
    Eigen::Matrix3d dR = normalizeRot(R1).transpose() * normalizeRot(R2);
    return Sophus::SO3d(dR).log();
}

inline Eigen::Matrix<double, 2, 3> tangentBasis(const Eigen::Vector3d& direction) {
    const Eigen::Vector3d axis = direction.normalized();
    Eigen::Vector3d reference = Eigen::Vector3d::UnitZ();
    if (axis.isApprox(reference)) {
        reference = Eigen::Vector3d::UnitX();
    }
    const Eigen::Vector3d b1 = (reference - axis * axis.dot(reference)).normalized();
    Eigen::Matrix<double, 2, 3> basis;
    basis.row(0) = b1.transpose();
    basis.row(1) = axis.cross(b1).transpose();
    return basis;
}

}  // namespace tassel_utils

#endif  // TASSEL_UTILS_ROTATION_H_
