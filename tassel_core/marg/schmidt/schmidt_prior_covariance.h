#ifndef TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_PRIOR_COVARIANCE_H_
#define TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_PRIOR_COVARIANCE_H_

#include <Eigen/Core>

#include <array>
#include <vector>

#include "marg/imu_covariance_propagation.h"
#include "marg/marg_lin_data.h"
#include "marg/window_action.h"

namespace tassel_core {

struct SchmidtPriorCovariance {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // covariance 始终使用对应 MargLinData 的紧凑列布局；固定的四维 gauge 在该矩阵中为零方差。
    Eigen::MatrixXd covariance;
    int state_count = 0;
};

SchmidtPriorCovariance buildGaugeFixedPriorCovariance(
    const MargLinData& prior, const Eigen::Matrix3d& anchor_rotation);

bool priorHasFiniteGaugeFixedCovariance(
    const MargLinData& prior, const Eigen::Matrix3d& anchor_rotation);

void recenterPriorCovariance(
    SchmidtPriorCovariance& prior_covariance, const MargLinData& prior,
    const std::vector<std::array<double, 6>>& new_poses);

void transformPriorCovarianceGauge(
    SchmidtPriorCovariance& prior_covariance, const Eigen::Matrix3d& rotation);

SchmidtPriorCovariance propagatePriorCovarianceWithImu(
    const SchmidtPriorCovariance& prior_covariance, int parent_frame_index,
    const ImuCovariancePropagation& propagation);

SchmidtPriorCovariance retainMarginalizedPriorCovariance(
    const SchmidtPriorCovariance& current_covariance, RetainedHostAction action);

std::vector<std::array<int, 3>> priorBaColumns(const SchmidtPriorCovariance& prior_covariance);

SchmidtPriorCovariance propagateAndUpdateSchmidtPrior(
    const SchmidtPriorCovariance& prior_covariance, const ImuCovariancePropagation& propagation,
    const Eigen::MatrixXd& visual_jacobian_in_window, const Eigen::VectorXd& visual_residual,
    int window_state_count, RetainedHostAction action);

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_SCHMIDT_SCHMIDT_PRIOR_COVARIANCE_H_
