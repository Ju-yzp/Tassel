#include "marginalization_prior_factor.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace tassel_core {

MarginalizationPriorFactor::MarginalizationPriorFactor(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
    std::vector<std::array<double, 6>> linearization_poses)
    : H_(H), b_(b), lin_poses_(std::move(linearization_poses)) {
    int num_kept = static_cast<int>(lin_poses_.size());
    set_num_residuals(static_cast<int>(b_.size()));
    for (int i = 0; i < num_kept; ++i) {
        mutable_parameter_block_sizes()->push_back(6);
    }
}

bool MarginalizationPriorFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    int num_kept = static_cast<int>(lin_poses_.size());

    Eigen::VectorXd delta(num_kept * 6);
    for (int i = 0; i < num_kept; ++i) {
        Eigen::Vector3d phi(parameters[i][0], parameters[i][1], parameters[i][2]);
        Eigen::Vector3d P(parameters[i][3], parameters[i][4], parameters[i][5]);
        Eigen::Vector3d phi_lin(lin_poses_[i][0], lin_poses_[i][1], lin_poses_[i][2]);
        Eigen::Vector3d P_lin(lin_poses_[i][3], lin_poses_[i][4], lin_poses_[i][5]);

        Sophus::SO3d R = Sophus::SO3d::exp(phi);
        Sophus::SO3d R_lin = Sophus::SO3d::exp(phi_lin);

        Sophus::SO3d dR = R_lin.inverse() * R;
        delta.segment<3>(i * 6) = dR.log();
        delta.segment<3>(i * 6 + 3) = P - P_lin;
    }

    Eigen::Map<Eigen::VectorXd> r(residuals, b_.size());
    r = H_ * delta + b_;

    if (jacobians) {
        for (int i = 0; i < num_kept; ++i) {
            if (jacobians[i]) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> J(
                    jacobians[i], b_.size(), 6);
                J = H_.block(0, i * 6, b_.size(), 6);
            }
        }
    }

    return true;
}

}  // namespace tassel_core
