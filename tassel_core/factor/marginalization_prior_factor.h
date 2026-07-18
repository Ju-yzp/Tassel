#ifndef TASSEL_CORE_FACTOR_MARGINALIZATION_PRIOR_FACTOR_H_
#define TASSEL_CORE_FACTOR_MARGINALIZATION_PRIOR_FACTOR_H_

#include <ceres/cost_function.h>
#include <Eigen/Core>

#include <array>
#include <vector>

#include "marg/marg_lin_data.h"

namespace tassel_core {

class MarginalizationPriorFactor : public ceres::CostFunction {
public:
    explicit MarginalizationPriorFactor(const MargLinData& data);

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override;

private:
    Eigen::MatrixXd H_;
    Eigen::VectorXd b_;
    std::vector<std::array<double, 6>> lin_poses_;
    std::vector<std::array<double, 9>> lin_speed_bias_;
    double lin_delay_time_;
    int num_kept_;
    bool has_speed_bias_;
    bool has_delay_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARGINALIZATION_PRIOR_FACTOR_H_
