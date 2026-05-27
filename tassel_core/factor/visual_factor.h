#ifndef TASSEL_CORE_VISUAL_FACTOR_H_
#define TASSEL_CORE_VISUAL_FACTOR_H_

#include <ceres/sized_cost_function.h>
#include <Eigen/Core>
#include <sophus/so3.hpp>

namespace tassel_core {
class VisualFactor : public ceres::SizedCostFunction<2, 6, 6, 1> {
public:
    VisualFactor(
        const Eigen::Vector3d& uv_i_, const Eigen::Vector3d& uv_j_, const Eigen::Matrix3d& ric_,
        const Eigen::Vector3d& tic_, double min_depth = 0.5);

    virtual bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const;

private:
    Eigen::Vector3d uv_i, uv_j;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    double min_depth_;
    Eigen::Vector3d w_G_, v_G_, a_B_;
    double dt_;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_VISUAL_FACTOR_H_ */
