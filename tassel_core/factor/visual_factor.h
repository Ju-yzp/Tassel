#ifndef TASSEL_CORE_VISUAL_FACTOR_H_
#define TASSEL_CORE_VISUAL_FACTOR_H_

#include <ceres/sized_cost_function.h>
#include <Eigen/Core>
#include <sophus/so3.hpp>

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
        const Eigen::Vector3d& uv_i_, const Eigen::Vector3d& uv_j_, const Eigen::Matrix3d& ric_,
        const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
        const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
        const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_,
        const Eigen::Matrix2d& sqrt_info_);

    virtual bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const;

private:
    // 相机坐标系下的归一化坐标
    Eigen::Vector3d uv_i, uv_j;
    // 外参
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    // 相机在imu(t)时刻的imu体坐标系下的角速度和加速度(帧i,j)
    Eigen::Vector3d w_i, w_j;
    Eigen::Vector3d a_i, a_j;
    // imu(t)时刻的在世界坐标系下imu的线速度，指向ceres优化变量线速度(帧i,j)
    const double *v_i, *v_j;
    // imu(t)时刻的在世界坐标系下imu的偏置，指向ceres优化变量偏置(帧i,j)
    const double *bg_i_lin, *bg_j_lin;
    Eigen::Matrix<double, 2, 3> tangent_base;
    // 信息矩阵，是像素协方差
    Eigen::Matrix2d sqrt_info;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_VISUAL_FACTOR_H_ */
