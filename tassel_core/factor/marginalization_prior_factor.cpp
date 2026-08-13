#include "marginalization_prior_factor.h"

#include <Eigen/Core>
#include <stdexcept>

#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {

MarginalizationPriorFactor::MarginalizationPriorFactor(const MargLinData& data)
    : H_(data.H),
      b_(data.b),
      lin_poses_(data.linearization_poses),
      lin_speed_bias_(data.linearization_speed_bias),
      lin_delay_time_(data.linearization_delay_time),
      num_kept_(data.stateCount()),
      delay_column_(-1) {
    data.validate();
    delay_column_ = data.delayColumn();
    set_num_residuals(static_cast<int>(b_.size()));
    pose_columns_.reserve(static_cast<size_t>(num_kept_));
    speed_bias_columns_.reserve(static_cast<size_t>(num_kept_ - 1));
    for (int i = 0; i < num_kept_; ++i) {
        pose_columns_.push_back(data.poseColumn(i));
        mutable_parameter_block_sizes()->push_back(6);  // pose
        if (i > 0) {
            speed_bias_columns_.push_back(data.speedBiasColumn(i));
            mutable_parameter_block_sizes()->push_back(9);  // speed_bias
        }
    }
    mutable_parameter_block_sizes()->push_back(1);
}

bool MarginalizationPriorFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    // 残差使用当前状态相对历史线性化点的增量，局部雅各比始终保持为边缘化时的 H。
    Eigen::VectorXd delta(H_.cols());
    int parameter_index = 0;
    for (int i = 0; i < num_kept_; ++i) {
        const int pose_column = pose_columns_[static_cast<size_t>(i)];
        const double* pose = parameters[parameter_index++];
        const Eigen::Vector3d position(pose[0], pose[1], pose[2]);
        const Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
        const Eigen::Vector3d linearization_position(
            lin_poses_[i][0], lin_poses_[i][1], lin_poses_[i][2]);
        const Eigen::Vector3d linearization_phi(
            lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);
        Eigen::Matrix<double, 6, 1> linearization_pose;
        linearization_pose << linearization_position, linearization_phi;
        Eigen::Matrix<double, 6, 1> current_pose;
        current_pose << position, phi;
        delta.segment<6>(pose_column) = rightTangentDelta(linearization_pose, current_pose);

        if (i > 0) {
            const int speed_bias_column = speed_bias_columns_[static_cast<size_t>(i - 1)];
            const double* speed_bias = parameters[parameter_index++];
            for (int d = 0; d < 9; ++d) {
                delta(speed_bias_column + d) = speed_bias[d] - lin_speed_bias_[i][d];
            }
        }
    }
    delta(delay_column_) = parameters[parameter_index++][0] - lin_delay_time_;

    Eigen::Map<Eigen::VectorXd> r(residuals, b_.size());
    r = H_ * delta + b_;

    if (jacobians) {
        parameter_index = 0;
        for (int i = 0; i < num_kept_; ++i) {
            const int pose_column = pose_columns_[static_cast<size_t>(i)];
            if (jacobians[parameter_index]) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> pose_jacobian(
                    jacobians[parameter_index], b_.size(), 6);
                double minus_data[36];
                SE3RightManifold manifold;
                TASSEL_ASSERT(manifold.MinusJacobian(parameters[parameter_index], minus_data));
                Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> minus(minus_data);
                pose_jacobian = H_.block(0, pose_column, b_.size(), 6) * minus;
            }
            ++parameter_index;
            if (i > 0) {
                if (jacobians[parameter_index]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>>
                        speed_bias_jacobian(jacobians[parameter_index], b_.size(), 9);
                    speed_bias_jacobian =
                        H_.block(0, speed_bias_columns_[static_cast<size_t>(i - 1)], b_.size(), 9);
                }
                ++parameter_index;
            }
        }
        if (jacobians[parameter_index]) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> J_delay(
                jacobians[parameter_index], b_.size(), 1);
            J_delay = H_.col(delay_column_);
        }
    }

    return true;
}

}  // namespace tassel_core
