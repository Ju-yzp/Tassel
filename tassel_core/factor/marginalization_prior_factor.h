#ifndef TASSEL_CORE_FACTOR_MARGINALIZATION_PRIOR_FACTOR_H_
#define TASSEL_CORE_FACTOR_MARGINALIZATION_PRIOR_FACTOR_H_

#include <ceres/cost_function.h>
#include <Eigen/Core>

#include <array>
#include <vector>

namespace tassel_core {

// Ceres CostFunction：将 sqrt 先验 H*(x - x_lin) + b 加入优化
class MarginalizationPriorFactor : public ceres::CostFunction {
public:
    MarginalizationPriorFactor(
        const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
        std::vector<std::array<double, 6>> linearization_poses);

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override;

private:
    Eigen::MatrixXd H_;
    Eigen::VectorXd b_;
    std::vector<std::array<double, 6>> lin_poses_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARGINALIZATION_PRIOR_FACTOR_H_
