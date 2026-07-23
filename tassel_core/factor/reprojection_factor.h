#ifndef TASSEL_CORE_REPROJECTION_FACTOR_H_
#define TASSEL_CORE_REPROJECTION_FACTOR_H_

#include <Eigen/Core>

#include <ceres/sized_cost_function.h>
#include <sophus/so3.hpp>

#include "cam/camera_base.h"

namespace tassel_core {

// 参数块依次为 [pose_i(6), pose_j(6), delay_time(1), inverse_depth(1)]。
// pose 中 R_k 将 IMU 体坐标系向量旋转到世界坐标系，ric/tic 将相机点变换到 IMU 系。
// 残差定义为 sqrt_info * (project(p_C_j) - pixel_j)。
class ReprojectionFactor : public ceres::SizedCostFunction<2, 6, 6, 1, 1> {
public:
    explicit ReprojectionFactor(
        const Eigen::Vector3d& uv_i_, const Eigen::Vector2d& pt_j_, const Eigen::Matrix3d& ric_,
        const Eigen::Vector3d& tic_, const Eigen::Vector3d& w_i_, const Eigen::Vector3d& w_j_,
        const Eigen::Vector3d& a_i_, const Eigen::Vector3d& a_j_, const double* v_i_,
        const double* v_j_, const double* bg_i_lin_, const double* bg_j_lin_,
        const double* ba_i_lin_, const double* ba_j_lin_, const Eigen::Matrix2d& sqrt_info_,
        const CameraBase* camera_, double sync_delay_i_ = 0.0, double sync_delay_j_ = 0.0);

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override;

private:
    Eigen::Vector3d uv_i;
    Eigen::Vector2d pt_j;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    Eigen::Vector3d w_i, w_j;
    Eigen::Vector3d a_i, a_j;
    Eigen::Vector3d v_i, v_j;
    Eigen::Vector3d bg_i, bg_j;
    Eigen::Vector3d ba_i, ba_j;
    Eigen::Matrix2d sqrt_info;
    const CameraBase* camera;
    double sync_delay_i, sync_delay_j;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_REPROJECTION_FACTOR_H_
