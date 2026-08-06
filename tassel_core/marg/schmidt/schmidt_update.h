#ifndef TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_UPDATE_H_
#define TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_UPDATE_H_

#include <Eigen/Core>

#include <vector>

namespace tassel_core {

struct SchmidtUpdateResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::VectorXd active_delta;
    Eigen::MatrixXd active_covariance;
    Eigen::MatrixXd active_schmidt_cross_covariance;
    Eigen::MatrixXd schmidt_covariance;
};

struct SchmidtSqrtPrior {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::MatrixXd H;
    Eigen::VectorXd b;
};

struct SchmidtJointUpdateResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::VectorXd delta;
    Eigen::MatrixXd covariance;
};

// 对一次线性高斯观测执行 Schmidt 更新；Schmidt 均值和边缘协方差不接收观测更新。
// active_delta 是主动状态均值增量，交叉协方差必须随本次创新同步更新。
SchmidtUpdateResult applySchmidtUpdate(
    const Eigen::MatrixXd& active_covariance,
    const Eigen::MatrixXd& active_schmidt_cross_covariance,
    const Eigen::MatrixXd& schmidt_covariance, const Eigen::MatrixXd& active_jacobian,
    const Eigen::MatrixXd& schmidt_jacobian, const Eigen::MatrixXd& measurement_covariance,
    const Eigen::VectorXd& residual);

SchmidtJointUpdateResult applySchmidtUpdateToJointCovariance(
    const Eigen::MatrixXd& covariance, const Eigen::MatrixXd& jacobian,
    const Eigen::MatrixXd& measurement_covariance, const Eigen::VectorXd& residual,
    const std::vector<int>& schmidt_indices);

// 在同一满秩先验上吸收一次线性似然，并把指定变量作为 Schmidt 状态。
// 残差约定为 H * delta + b；输出仍以相同坐标表达，Schmidt 均值和边缘协方差保持先验值。
SchmidtSqrtPrior buildSchmidtUpdatedSqrtPrior(
    const Eigen::MatrixXd& prior_sqrt_information, const Eigen::VectorXd& prior_residual,
    const Eigen::MatrixXd& likelihood_sqrt_information, const Eigen::VectorXd& likelihood_residual,
    const std::vector<int>& schmidt_indices);

Eigen::MatrixXd covarianceFromFullRankSqrtInformation(const Eigen::MatrixXd& sqrt_information);

Eigen::MatrixXd sqrtInformationFromPositiveDefiniteCovariance(const Eigen::MatrixXd& covariance);

// 下面两个接口只计算 Moore-Penrose 伪逆。信息零空间在结果中为零，不能解释为
// 概率协方差，也不能用于 Schmidt/consider 更新。
Eigen::MatrixXd pseudoCovarianceFromSqrtInformation(const Eigen::MatrixXd& sqrt_information);

Eigen::MatrixXd sqrtInformationFromPseudoCovariance(const Eigen::MatrixXd& pseudo_covariance);

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_UPDATE_H_
