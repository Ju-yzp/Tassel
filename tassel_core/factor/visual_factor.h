#ifndef TASSEL_CORE_VISUAL_FACTOR_H_
#define TASSEL_CORE_VISUAL_FACTOR_H_

#include <Eigen/Core>

#include <ceres/sized_cost_function.h>
#include <sophus/so3.hpp>

#include "cam/camera_base.h"

namespace tassel_core {

inline Eigen::Matrix<double, 2, 3> computeTangentBasis(const Eigen::Vector3d& uv) {
    Eigen::Vector3d a = uv.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp) {
        tmp << 1, 0, 0;
    }
    Eigen::Vector3d b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    Eigen::Vector3d b2 = a.cross(b1);
    Eigen::Matrix<double, 2, 3> tangent_base;
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
    return tangent_base;
}

class VisualFactor : public ceres::SizedCostFunction<2, 6, 6, 1, 1> {
public:
    explicit VisualFactor(
        const Eigen::Vector3d& uv_i_, const Eigen::Vector2d& pt_j_, const Eigen::Matrix3d& ric_,
        const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
        const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
        const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_,
        const double* ba_i_lin_, const double* ba_j_lin_, const Eigen::Matrix2d& sqrt_info_,
        const CameraBase* camera);

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override;

private:
    Eigen::Vector3d uv_i;
    Eigen::Vector2d pt_j;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    Eigen::Vector3d w_i, w_j;
    Eigen::Vector3d a_i, a_j;
    const double *v_i, *v_j;
    const double *bg_i_lin, *bg_j_lin;
    const double *ba_i_lin, *ba_j_lin;
    Eigen::Matrix2d sqrt_info;
    const CameraBase* camera;
};

}  // namespace tassel_core
#endif /* TASSEL_CORE_VISUAL_FACTOR_H_ */
