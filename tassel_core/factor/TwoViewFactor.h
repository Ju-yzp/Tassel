#ifndef TASSEL_CORE_FACTOR_TWO_VIEW_FACTOR_H_
#define TASSEL_CORE_FACTOR_TWO_VIEW_FACTOR_H_

#include <ceres/sized_cost_function.h>

#include <Eigen/Core>

namespace tassel_core {
class TwoViewFactor : public ceres::SizedCostFunction<1, 6, 6, 1> {
public:
    explicit TwoViewFactor(
        const Eigen::Vector3d& uv_i_, const Eigen::Vector3d& uv_j_, const Eigen::Matrix3d& ric_,
        const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
        const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
        const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_,
        const double* ba_i_lin_, const double* ba_j_lin_, const double sqrt_info_);

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override;

private:
    Eigen::Vector3d uv_i, uv_j;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    Eigen::Vector3d w_i, w_j;
    Eigen::Vector3d a_i, a_j;
    const double *v_i, *v_j;
    const double *bg_i_lin, *bg_j_lin;
    const double *ba_i_lin, *ba_j_lin;
    double sqrt_info;
};

}  // namespace tassel_core

#endif
