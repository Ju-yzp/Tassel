#ifndef TASSEL_CORE_MARG_IMU_BLOCK_H_
#define TASSEL_CORE_MARG_IMU_BLOCK_H_

#include <array>
#include <memory>
#include <vector>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "tassel_utils/macros.h"

namespace tassel_core {
template <typename Derived>
class IMUBlock {
public:
    void allocate(IntegratorBase<Derived>* integrator) {
        TASSEL_ASSERT(integrator != nullptr);
        auto pint_ptr =
            std::shared_ptr<IntegratorBase<Derived>>(integrator, [](IntegratorBase<Derived>*) {});
        imu_factor_ = std::make_unique<IMUFactor<Derived>>(pint_ptr);
    }

    void linearize(
        Eigen::Vector3d V_i, Eigen::Vector3d V_j, Eigen::Vector3d P_i, Eigen::Vector3d P_j,
        Eigen::Vector3d Q_i, Eigen::Vector3d Q_j, Eigen::Vector3d Ba_i, Eigen::Vector3d Ba_j,
        Eigen::Vector3d Bg_i, Eigen::Vector3d Bg_j) {
        std::array<double, 6> param_pose_i, param_pose_j;
        std::array<double, 9> param_speed_bias_i, param_speed_bias_j;
        param_pose_i = {P_i.x(), P_i.y(), P_i.z(), Q_i.x(), Q_i.y(), Q_i.z()};
        param_speed_bias_i = {V_i.x(),  V_i.y(),  V_i.z(),  Ba_i.x(), Ba_i.y(),
                              Ba_i.z(), Bg_i.x(), Bg_i.y(), Bg_i.z()};
        param_pose_j = {P_j.x(), P_j.y(), P_j.z(), Q_j.x(), Q_j.y(), Q_j.z()};
        param_speed_bias_j = {V_j.x(),  V_j.y(),  V_j.z(),  Ba_j.x(), Ba_j.y(),
                              Ba_j.z(), Bg_j.x(), Bg_j.y(), Bg_j.z()};

        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i, jacobian_pose_j;
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speed_bias_i, jacobian_speed_bias_j;
        Eigen::Matrix<double, 15, 1> residual;

        std::vector<double*> parameters;
        std::vector<double*> jacobians;
        parameters.push_back(param_pose_i.data());
        parameters.push_back(param_speed_bias_i.data());
        parameters.push_back(param_pose_j.data());
        parameters.push_back(param_speed_bias_j.data());

        jacobians.push_back(jacobian_pose_i.data());
        jacobians.push_back(jacobian_speed_bias_i.data());
        jacobians.push_back(jacobian_pose_j.data());
        jacobians.push_back(jacobian_speed_bias_j.data());

        TASSEL_ASSERT(imu_factor_->Evaluate(parameters.data(), residual.data(), jacobians.data()));
        // IMUFactor 对旋转向量求导；边缘化系统使用当前姿态处的右扰动切空间。
        jacobian_pose_i.block<15, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(-Q_i);
        jacobian_pose_j.block<15, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(-Q_j);

        b_ = residual;
        // 固定列布局为 [pose_i(6), speed_bias_i(9), pose_j(6), speed_bias_j(9)]。
        Jp_.template block<15, 6>(0, 0) = jacobian_pose_i;
        Jp_.template block<15, 9>(0, 6) = jacobian_speed_bias_i;
        Jp_.template block<15, 6>(0, 15) = jacobian_pose_j;
        Jp_.template block<15, 9>(0, 21) = jacobian_speed_bias_j;
    }

    void get_dense_Jp_b(Eigen::MatrixXd& Jp, Eigen::VectorXd& b, int start_row, int start_col) {
        TASSEL_ASSERT(Jp.rows() == b.rows());
        Jp.template block<15, 30>(start_row, start_col) = Jp_;
        b.template block<15, 1>(start_row, 0) = b_;
    }

private:
    std::unique_ptr<IMUFactor<Derived>> imu_factor_;
    Eigen::Matrix<double, 15, 30> Jp_;
    Eigen::Matrix<double, 15, 1> b_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_IMU_BLOCK_H_
