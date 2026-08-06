#include "marginalization_prior_factor.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

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
      num_kept_(static_cast<int>(data.linearization_poses.size())),
      layout_(num_kept_, static_cast<int>(data.H.cols())) {
    if (H_.rows() != b_.size()) {
        throw std::invalid_argument("Marginalization prior Jacobian and residual rows differ");
    }
    for (int i = 0; i < num_kept_; ++i) {
        if (layout_.hasSpeedBias(i) && static_cast<int>(lin_speed_bias_.size()) != num_kept_) {
            throw std::invalid_argument("Marginalization prior speed-bias states are incomplete");
        }
    }
    set_num_residuals(static_cast<int>(b_.size()));
    for (int i = 0; i < num_kept_; ++i) {
        mutable_parameter_block_sizes()->push_back(6);  // pose
        if (layout_.hasSpeedBias(i)) {
            mutable_parameter_block_sizes()->push_back(9);  // speed_bias
        }
    }
    if (layout_.hasDelay()) {
        mutable_parameter_block_sizes()->push_back(1);
    }
}

bool MarginalizationPriorFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    // 残差使用当前状态相对历史线性化点的增量，局部雅各比始终保持为边缘化时的 H。
    Eigen::VectorXd delta(H_.cols());
    int parameter_index = 0;
    for (int i = 0; i < num_kept_; ++i) {
        const int pose_column = layout_.poseColumn(i);
        const double* pose = parameters[parameter_index++];
        const Eigen::Vector3d position(pose[0], pose[1], pose[2]);
        const Eigen::Vector3d phi(pose[3], pose[4], pose[5]);
        const Eigen::Vector3d linearization_position(
            lin_poses_[i][0], lin_poses_[i][1], lin_poses_[i][2]);
        const Eigen::Vector3d linearization_phi(
            lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);
        const Sophus::SO3d rotation_delta =
            Sophus::SO3d::exp(linearization_phi).inverse() * Sophus::SO3d::exp(phi);
        delta.segment<3>(pose_column) = position - linearization_position;
        delta.segment<3>(pose_column + 3) = rotation_delta.log();

        if (layout_.hasSpeedBias(i)) {
            const int speed_bias_column = layout_.speedBiasColumn(i);
            const double* speed_bias = parameters[parameter_index++];
            for (int d = 0; d < 9; ++d) {
                delta(speed_bias_column + d) = speed_bias[d] - lin_speed_bias_[i][d];
            }
        }
    }
    if (layout_.hasDelay()) {
        delta(layout_.delayColumn()) = parameters[parameter_index++][0] - lin_delay_time_;
    }

    Eigen::Map<Eigen::VectorXd> r(residuals, b_.size());
    r = H_ * delta + b_;

    if (jacobians) {
        parameter_index = 0;
        for (int i = 0; i < num_kept_; ++i) {
            const int pose_column = layout_.poseColumn(i);
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
            if (layout_.hasSpeedBias(i)) {
                if (jacobians[parameter_index]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>>
                        speed_bias_jacobian(jacobians[parameter_index], b_.size(), 9);
                    speed_bias_jacobian = H_.block(0, layout_.speedBiasColumn(i), b_.size(), 9);
                }
                ++parameter_index;
            }
        }
        if (layout_.hasDelay()) {
            if (jacobians[parameter_index]) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> J_delay(
                    jacobians[parameter_index], b_.size(), 1);
                J_delay = H_.col(layout_.delayColumn());
            }
        }
    }

    return true;
}

}  // namespace tassel_core
