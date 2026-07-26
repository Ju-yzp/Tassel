#include "marginalization_prior_factor.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

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
    const int mixed_state_cols = num_kept_ > 0 ? 6 + (num_kept_ - 1) * 15 : 0;
    has_speed_bias_ = static_cast<int>(lin_speed_bias_.size()) == num_kept_ &&
                      (H_.cols() == full_state_cols || H_.cols() == full_state_cols + 1);
    has_pose_only_host_ = static_cast<int>(lin_speed_bias_.size()) == num_kept_ &&
                          (H_.cols() == mixed_state_cols || H_.cols() == mixed_state_cols + 1);
    has_delay_ = H_.cols() == (has_speed_bias_       ? full_state_cols + 1
                               : has_pose_only_host_ ? mixed_state_cols + 1
                                                     : pose_only_cols + 1);
    set_num_residuals(static_cast<int>(b_.size()));
    if (has_pose_only_host_) {
        mutable_parameter_block_sizes()->push_back(6);  // retained host pose
        for (int i = 1; i < num_kept_; ++i) {
            mutable_parameter_block_sizes()->push_back(6);  // active pose
            mutable_parameter_block_sizes()->push_back(9);  // active speed_bias
        }
    } else if (!has_speed_bias_) {
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
    // 残差使用当前状态相对历史线性化点的增量，局部雅各比始终保持为边缘化时的 H。
    Eigen::VectorXd delta(H_.cols());
    if (has_pose_only_host_) {
        int parameter_index = 0;
        int column_index = 0;
        for (int i = 0; i < num_kept_; ++i) {
            const double* pose = parameters[parameter_index++];
            Eigen::Vector3d P(pose[0], pose[1], pose[2]);
            Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
            Eigen::Vector3d P_lin(lin_poses_[i][0], lin_poses_[i][1], lin_poses_[i][2]);
            Eigen::Vector3d phi_lin(lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);

            Sophus::SO3d R = Sophus::SO3d::exp(phi);
            Sophus::SO3d R_lin = Sophus::SO3d::exp(phi_lin);
            Sophus::SO3d dR = R_lin.inverse() * R;

            delta.segment<3>(column_index) = P - P_lin;
            delta.segment<3>(column_index + 3) = dR.log();
            column_index += 6;

            if (i == 0) {
                continue;
            }
            const double* sb = parameters[parameter_index++];
            for (int d = 0; d < 3; ++d) {
                delta(column_index + d) = sb[d] - lin_speed_bias_[i][d];  // 速度
                delta(column_index + 3 + d) =
                    sb[3 + d] - lin_speed_bias_[i][3 + d];  // 加速度计偏置
                delta(column_index + 6 + d) = sb[6 + d] - lin_speed_bias_[i][6 + d];  // 陀螺仪偏置
            }
            column_index += 9;
        }
    } else if (!has_speed_bias_) {
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
        const int delay_param_idx = has_speed_bias_       ? 2 * num_kept_
                                    : has_pose_only_host_ ? 2 * num_kept_ - 1
                                                          : num_kept_;
        delta(delta.size() - 1) = parameters[delay_param_idx][0] - lin_delay_time_;
    }

    Eigen::Map<Eigen::VectorXd> r(residuals, b_.size());
    r = H_ * delta + b_;

    if (jacobians) {
        if (has_pose_only_host_) {
            int parameter_index = 0;
            int column_index = 0;
            for (int i = 0; i < num_kept_; ++i) {
                if (jacobians[parameter_index]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> J_pose(
                        jacobians[parameter_index], b_.size(), 6);
                    double minus_data[36];
                    SE3RightManifold manifold;
                    TASSEL_ASSERT(manifold.MinusJacobian(parameters[parameter_index], minus_data));
                    Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(
                        minus_data);
                    J_pose = H_.block(0, column_index, b_.size(), 6) * minus;
                }
                ++parameter_index;
                column_index += 6;

                if (i == 0) {
                    continue;
                }
                if (jacobians[parameter_index]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> J_sb(
                        jacobians[parameter_index], b_.size(), 9);
                    J_sb = H_.block(0, column_index, b_.size(), 9);
                }
                ++parameter_index;
                column_index += 9;
            }
        } else {
            for (int i = 0; i < num_kept_; ++i) {
                if (!has_speed_bias_) {
                    if (jacobians[i]) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> J(
                            jacobians[i], b_.size(), 6);
                        double minus_data[36];
                        SE3RightManifold manifold;
                        TASSEL_ASSERT(manifold.MinusJacobian(parameters[i], minus_data));
                        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(
                            minus_data);
                        J = H_.block(0, i * 6, b_.size(), 6) * minus;
                    }
                } else {
                    if (jacobians[2 * i]) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>>
                            J_pose(jacobians[2 * i], b_.size(), 6);
                        double minus_data[36];
                        SE3RightManifold manifold;
                        TASSEL_ASSERT(manifold.MinusJacobian(parameters[2 * i], minus_data));
                        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(
                            minus_data);
                        J_pose = H_.block(0, i * 15, b_.size(), 6) * minus;
                    }
                    if (jacobians[2 * i + 1]) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> J_sb(
                            jacobians[2 * i + 1], b_.size(), 9);
                        J_sb = H_.block(0, i * 15 + 6, b_.size(), 9);
                    }
                }
            }
        }
        if (has_delay_) {
            const int delay_param_idx = has_speed_bias_       ? 2 * num_kept_
                                        : has_pose_only_host_ ? 2 * num_kept_ - 1
                                                              : num_kept_;
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
