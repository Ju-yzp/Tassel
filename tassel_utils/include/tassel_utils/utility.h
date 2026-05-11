#ifndef TASSEL_UTILS_UTILITY_H_
#define TASSEL_UTILS_UTILITY_H_

#include <Eigen/Dense>

namespace tassel_utils {
inline Eigen::Quaterniond deltaQ(const Eigen::Vector3d& theta) {
    Eigen::Quaterniond dq;
    Eigen::Vector3d half_theta = theta;
    half_theta /= 2.0;
    dq.w() = 1.0;
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

inline Eigen::Matrix<double, 3, 3> skew_x(const Eigen::Matrix<double, 3, 1>& w) {
    Eigen::Matrix<double, 3, 3> w_x;
    w_x << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return w_x;
}

inline Eigen::Matrix<double, 3, 3> exp_so3(const Eigen::Matrix<double, 3, 1>& w) {
    // get theta
    Eigen::Matrix<double, 3, 3> w_x = skew_x(w);
    double theta = w.norm();
    // Handle small angle values
    double A, B;
    if (theta < 1e-7) {
        A = 1;
        B = 0.5;
    } else {
        A = sin(theta) / theta;
        B = (1 - cos(theta)) / (theta * theta);
    }
    // compute so(3) rotation
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (theta > 1e-7) {
        R += A * w_x + B * w_x * w_x;
    }
    return R;
}

inline Eigen::Matrix<double, 3, 1> log_so3(const Eigen::Matrix<double, 3, 3>& R) {
    // note switch to base 1
    double R11 = R(0, 0), R12 = R(0, 1), R13 = R(0, 2);
    double R21 = R(1, 0), R22 = R(1, 1), R23 = R(1, 2);
    double R31 = R(2, 0), R32 = R(2, 1), R33 = R(2, 2);

    // Get trace(R)
    const double tr = R.trace();
    Eigen::Vector3d omega;

    // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    // we do something special
    if (tr + 1.0 < 1e-10) {
        if (std::abs(R33 + 1.0) > 1e-5)
            omega = (M_PI / sqrt(2.0 + 2.0 * R33)) * Eigen::Vector3d(R13, R23, 1.0 + R33);
        else if (std::abs(R22 + 1.0) > 1e-5)
            omega = (M_PI / sqrt(2.0 + 2.0 * R22)) * Eigen::Vector3d(R12, 1.0 + R22, R32);
        else
            // if(std::abs(R.r1_.x()+1.0) > 1e-5)  This is implicit
            omega = (M_PI / sqrt(2.0 + 2.0 * R11)) * Eigen::Vector3d(1.0 + R11, R21, R31);
    } else {
        double magnitude;
        const double tr_3 = tr - 3.0;  // always negative
        if (tr_3 < -1e-7) {
            double theta = acos((tr - 1.0) / 2.0);
            magnitude = theta / (2.0 * sin(theta));
        } else {
            // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            // see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0;
        }
        omega = magnitude * Eigen::Vector3d(R32 - R23, R13 - R31, R21 - R12);
    }

    return omega;
}

inline Eigen::Matrix4d exp_se3(Eigen::Matrix<double, 6, 1> vec) {
    // Precompute our values
    Eigen::Vector3d w = vec.head(3);
    Eigen::Vector3d u = vec.tail(3);
    double theta = sqrt(w.dot(w));
    Eigen::Matrix3d wskew;
    wskew << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

    // Handle small angle values
    double A, B, C;
    if (theta < 1e-7) {
        A = 1;
        B = 0.5;
        C = 1.0 / 6.0;
    } else {
        A = sin(theta) / theta;
        B = (1 - cos(theta)) / (theta * theta);
        C = (1 - A) / (theta * theta);
    }

    // Matrices we need V and Identity
    Eigen::Matrix3d I_33 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d V = I_33 + B * wskew + C * wskew * wskew;

    // Get the final matrix to return
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat.block(0, 0, 3, 3) = I_33 + A * wskew + B * wskew * wskew;
    mat.block(0, 3, 3, 1) = V * u;
    mat(3, 3) = 1;
    return mat;
}

inline Eigen::Quaterniond positify(const Eigen::Quaterniond& q) {
    return q.w() >= 0.0 ? q : Eigen::Quaterniond(-q.w(), -q.x(), -q.y(), -q.z());
}
}  // namespace tassel_utils
#endif
