#ifndef TASSEL_CORE_MARG_IMU_COVARIANCE_PROPAGATION_H_
#define TASSEL_CORE_MARG_IMU_COVARIANCE_PROPAGATION_H_

#include <Eigen/Core>

namespace tassel_core {

struct ImuCovariancePropagation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<double, 15, 15> F;
    Eigen::Matrix<double, 15, 15> Q;
    Eigen::Matrix<double, 15, 1> offset;
};

// 白化 IMU 因子满足 Ji * dxi + Jj * dxj + residual = noise。
ImuCovariancePropagation buildImuCovariancePropagation(
    const Eigen::Matrix<double, 15, 15>& jacobian_i,
    const Eigen::Matrix<double, 15, 15>& jacobian_j, const Eigen::Matrix<double, 15, 1>& residual);

// prior_covariance 的 parent_state_column 指向连续 15 维 [p, theta, v, ba, bg] 块；
// 输出在矩阵末尾追加传播后的子状态，并同步生成它与全部已有变量的交叉协方差。
Eigen::MatrixXd appendImuPropagatedCovariance(
    const Eigen::MatrixXd& prior_covariance, int parent_state_column,
    const ImuCovariancePropagation& propagation);

Eigen::MatrixXd insertImuPropagatedCovariance(
    const Eigen::MatrixXd& prior_covariance, int parent_state_column, int child_state_column,
    const ImuCovariancePropagation& propagation);

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_IMU_COVARIANCE_PROPAGATION_H_
