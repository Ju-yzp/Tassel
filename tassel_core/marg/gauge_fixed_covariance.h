#ifndef TASSEL_CORE_MARG_GAUGE_FIXED_COVARIANCE_H_
#define TASSEL_CORE_MARG_GAUGE_FIXED_COVARIANCE_H_

#include <Eigen/Core>

#include <vector>

namespace tassel_core {

struct GaugeFixedBasis {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // full_delta = tangent * reduced_delta；position 和全局 yaw 四个 gauge 自由度被显式删除。
    Eigen::MatrixXd tangent;
    std::vector<int> full_to_reduced;
};

struct GaugeFixedCovariance {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::MatrixXd reduced;
    Eigen::MatrixXd lifted;
};

GaugeFixedBasis buildPositionYawGaugeBasis(
    int state_size, int anchor_pose_column, const Eigen::Matrix3d& anchor_rotation);

GaugeFixedCovariance covarianceInFixedGauge(
    const Eigen::MatrixXd& sqrt_information, const GaugeFixedBasis& basis);

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_GAUGE_FIXED_COVARIANCE_H_
