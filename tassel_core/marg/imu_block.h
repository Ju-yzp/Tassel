#ifndef TASSEL_CORE_MARG_IMU_BLOCK_H_
#define TASSEL_CORE_MARG_IMU_BLOCK_H_

#include <array>
#include <memory>

#include "factor/imu_factor.h"
#include "factor/midpoint_integrator.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
class IMUBlock {
public:
    void allocate(MidPointIntegrator* integrator) {
        TASSEL_ASSERT(integrator != nullptr);
        auto pint_ptr = std::shared_ptr<MidPointIntegrator>(integrator, [](MidPointIntegrator*) {});
        imu_factor_ = std::make_unique<IMUFactor>(pint_ptr);
    }

    void linearize(
        const std::array<double, 6>& current_pose_i,
        const std::array<double, 9>& current_speed_bias_i,
        const std::array<double, 6>& current_pose_j,
        const std::array<double, 9>& current_speed_bias_j,
        const std::array<double, 6>& linearized_pose_i,
        const std::array<double, 9>& linearized_speed_bias_i,
        const std::array<double, 6>& linearized_pose_j,
        const std::array<double, 9>& linearized_speed_bias_j) {
        std::array<double, 6> current_pose_i_copy = current_pose_i;
        std::array<double, 9> current_speed_bias_i_copy = current_speed_bias_i;
        std::array<double, 6> current_pose_j_copy = current_pose_j;
        std::array<double, 9> current_speed_bias_j_copy = current_speed_bias_j;
        std::array<double, 6> linearized_pose_i_copy = linearized_pose_i;
        std::array<double, 9> linearized_speed_bias_i_copy = linearized_speed_bias_i;
        std::array<double, 6> linearized_pose_j_copy = linearized_pose_j;
        std::array<double, 9> linearized_speed_bias_j_copy = linearized_speed_bias_j;

        const double* current_parameters[] = {
            current_pose_i_copy.data(), current_speed_bias_i_copy.data(),
            current_pose_j_copy.data(), current_speed_bias_j_copy.data()};
        const double* linearized_parameters[] = {
            linearized_pose_i_copy.data(), linearized_speed_bias_i_copy.data(),
            linearized_pose_j_copy.data(), linearized_speed_bias_j_copy.data()};
        Eigen::Matrix<double, 15, 1> current_residual;
        Eigen::Matrix<double, 15, 1> unused_linearized_residual;
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i;
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speed_bias_i;
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speed_bias_j;
        double* jacobians[] = {
            jacobian_pose_i.data(), jacobian_speed_bias_i.data(), jacobian_pose_j.data(),
            jacobian_speed_bias_j.data()};
        TASSEL_ASSERT(imu_factor_->Evaluate(current_parameters, current_residual.data(), nullptr));
        TASSEL_ASSERT(imu_factor_->Evaluate(
            linearized_parameters, unused_linearized_residual.data(), jacobians));

        jacobian_pose_i.block<15, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(
            -Eigen::Vector3d(linearized_pose_i[3], linearized_pose_i[4], linearized_pose_i[5]));
        jacobian_pose_j.block<15, 3>(0, 3) *= Sophus::SO3d::leftJacobianInverse(
            -Eigen::Vector3d(linearized_pose_j[3], linearized_pose_j[4], linearized_pose_j[5]));

        const Eigen::Map<const Eigen::Matrix<double, 6, 1>> lin_pose_i(linearized_pose_i.data());
        const Eigen::Map<const Eigen::Matrix<double, 6, 1>> cur_pose_i(current_pose_i.data());
        const Eigen::Map<const Eigen::Matrix<double, 6, 1>> lin_pose_j(linearized_pose_j.data());
        const Eigen::Map<const Eigen::Matrix<double, 6, 1>> cur_pose_j(current_pose_j.data());
        const Eigen::Matrix<double, 6, 1> delta_pose_i = rightTangentDelta(lin_pose_i, cur_pose_i);
        const Eigen::Matrix<double, 6, 1> delta_pose_j = rightTangentDelta(lin_pose_j, cur_pose_j);
        Eigen::Matrix<double, 30, 1> delta = Eigen::Matrix<double, 30, 1>::Zero();
        delta.segment<6>(0) = delta_pose_i;
        delta.segment<9>(6) =
            Eigen::Map<const Eigen::Matrix<double, 9, 1>>(current_speed_bias_i.data()) -
            Eigen::Map<const Eigen::Matrix<double, 9, 1>>(linearized_speed_bias_i.data());
        delta.segment<6>(15) = delta_pose_j;
        delta.segment<9>(21) =
            Eigen::Map<const Eigen::Matrix<double, 9, 1>>(current_speed_bias_j.data()) -
            Eigen::Map<const Eigen::Matrix<double, 9, 1>>(linearized_speed_bias_j.data());
        Jp_.template block<15, 6>(0, 0) = jacobian_pose_i;
        Jp_.template block<15, 9>(0, 6) = jacobian_speed_bias_i;
        Jp_.template block<15, 6>(0, 15) = jacobian_pose_j;
        Jp_.template block<15, 9>(0, 21) = jacobian_speed_bias_j;
        b_ = current_residual - Jp_ * delta;
    }

    void get_dense_Jp_b(Eigen::MatrixXd& Jp, Eigen::VectorXd& b, int start_row, int start_col) {
        TASSEL_ASSERT(Jp.rows() == b.rows());
        Jp.template block<15, 30>(start_row, start_col) = Jp_;
        b.template block<15, 1>(start_row, 0) = b_;
    }

    // 线性化后保持 [state_i(15), state_j(15)] 列布局，供协方差传播使用。
    const Eigen::Matrix<double, 15, 30>& jacobian() const { return Jp_; }
    const Eigen::Matrix<double, 15, 1>& residual() const { return b_; }

private:
    std::unique_ptr<IMUFactor> imu_factor_;
    Eigen::Matrix<double, 15, 30> Jp_;
    Eigen::Matrix<double, 15, 1> b_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_IMU_BLOCK_H_
