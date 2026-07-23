#include "marginalization_prior_factor.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace tassel_core {

MarginalizationPriorFactor::MarginalizationPriorFactor(const MargLinData& data)
    : H_(data.H),
      b_(data.b),
      lin_poses_(data.linearization_poses),
      lin_speed_bias_(data.linearization_speed_bias),
      lin_delay_time_(data.linearization_delay_time) {
    num_kept_ = static_cast<int>(lin_poses_.size());
    const int pose_only_cols = num_kept_ * 6;
    const int full_state_cols = num_kept_ * 15;
    has_speed_bias_ = static_cast<int>(lin_speed_bias_.size()) == num_kept_ &&
                      (H_.cols() == full_state_cols || H_.cols() == full_state_cols + 1);
    has_delay_ = H_.cols() == (has_speed_bias_ ? full_state_cols + 1 : pose_only_cols + 1);
    set_num_residuals(static_cast<int>(b_.size()));
    if (!has_speed_bias_) {
        for (int i = 0; i < num_kept_; ++i) {
            mutable_parameter_block_sizes()->push_back(6);
        }
    } else {
        for (int i = 0; i < num_kept_; ++i) {
            mutable_parameter_block_sizes()->push_back(6);  // pose
            mutable_parameter_block_sizes()->push_back(9);  // speed_bias
        }
    }
    if (has_delay_) {
        mutable_parameter_block_sizes()->push_back(1);
    }
}

bool MarginalizationPriorFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    // 先验残差始终定义为 r = H * delta + b；旋转增量采用线性化姿态到当前姿态的右扰动。
    Eigen::VectorXd delta(H_.cols());
    if (!has_speed_bias_) {
        for (int i = 0; i < num_kept_; ++i) {
            const double* pose = parameters[i];
            Eigen::Vector3d P(pose[0], pose[1], pose[2]);
            Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
            Eigen::Vector3d P_lin(lin_poses_[i][0], lin_poses_[i][1], lin_poses_[i][2]);
            Eigen::Vector3d phi_lin(lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);

            Sophus::SO3d R = Sophus::SO3d::exp(phi);
            Sophus::SO3d R_lin = Sophus::SO3d::exp(phi_lin);
            Sophus::SO3d dR = R_lin.inverse() * R;

            delta.segment<3>(i * 6) = P - P_lin;
            delta.segment<3>(i * 6 + 3) = dR.log();
        }
    } else {
        for (int i = 0; i < num_kept_; ++i) {
            const double* pose = parameters[2 * i];
            const double* sb = parameters[2 * i + 1];

            Eigen::Vector3d P(pose[0], pose[1], pose[2]);
            Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
            Eigen::Vector3d P_lin(lin_poses_[i][0], lin_poses_[i][1], lin_poses_[i][2]);
            Eigen::Vector3d phi_lin(lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);

            Sophus::SO3d R = Sophus::SO3d::exp(phi);
            Sophus::SO3d R_lin = Sophus::SO3d::exp(phi_lin);
            Sophus::SO3d dR = R_lin.inverse() * R;

            delta.segment<3>(i * 15) = P - P_lin;
            delta.segment<3>(i * 15 + 3) = dR.log();

            for (int d = 0; d < 3; ++d) {
                delta(i * 15 + 6 + d) = sb[d] - lin_speed_bias_[i][d];          // 速度
                delta(i * 15 + 9 + d) = sb[3 + d] - lin_speed_bias_[i][3 + d];  // 加速度计偏置
                delta(i * 15 + 12 + d) = sb[6 + d] - lin_speed_bias_[i][6 + d];  // 陀螺仪偏置
            }
        }
    }
    if (has_delay_) {
        delta(delta.size() - 1) =
            parameters[has_speed_bias_ ? 2 * num_kept_ : num_kept_][0] - lin_delay_time_;
    }

    Eigen::Map<Eigen::VectorXd> r(residuals, b_.size());
    r = H_ * delta + b_;

    if (jacobians) {
        for (int i = 0; i < num_kept_; ++i) {
            if (!has_speed_bias_) {
                if (jacobians[i]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> J(
                        jacobians[i], b_.size(), 6);
                    J = H_.block(0, i * 6, b_.size(), 6);
                    const double* pose = parameters[i];
                    Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
                    Eigen::Vector3d phi_lin(lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);
                    Eigen::Vector3d delta =
                        (Sophus::SO3d::exp(phi_lin).inverse() * Sophus::SO3d::exp(phi)).log();
                    J.rightCols(3) *= Sophus::SO3d::leftJacobianInverse(-delta) *
                                      Sophus::SO3d::leftJacobian(-phi);
                }
            } else {
                if (jacobians[2 * i]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> J_pose(
                        jacobians[2 * i], b_.size(), 6);
                    J_pose = H_.block(0, i * 15, b_.size(), 6);
                    const double* pose = parameters[2 * i];
                    Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
                    Eigen::Vector3d phi_lin(lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);
                    Eigen::Vector3d delta =
                        (Sophus::SO3d::exp(phi_lin).inverse() * Sophus::SO3d::exp(phi)).log();
                    J_pose.rightCols(3) *= Sophus::SO3d::leftJacobianInverse(-delta) *
                                           Sophus::SO3d::leftJacobian(-phi);
                }
                if (jacobians[2 * i + 1]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> J_sb(
                        jacobians[2 * i + 1], b_.size(), 9);
                    J_sb = H_.block(0, i * 15 + 6, b_.size(), 9);
                }
            }
        }
        if (has_delay_) {
            const int delay_param_idx = has_speed_bias_ ? 2 * num_kept_ : num_kept_;
            if (jacobians[delay_param_idx]) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> J_delay(
                    jacobians[delay_param_idx], b_.size(), 1);
                J_delay = H_.rightCols(1);
            }
        }
    }

    return true;
}

}  // namespace tassel_core
