#include "imu_covariance_propagation.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tassel_core {
namespace {

void requireCovariance(const Eigen::MatrixXd& covariance, const char* name) {
    if (covariance.rows() != covariance.cols() || !covariance.allFinite() ||
        !covariance.isApprox(covariance.transpose(), 1e-10)) {
        throw std::invalid_argument(std::string(name) + " is not a finite symmetric matrix");
    }
    if (covariance.rows() == 0) {
        return;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(std::string(name) + " eigendecomposition failed");
    }
    const double scale = std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
    if (solver.eigenvalues().minCoeff() < -1e-11 * scale) {
        throw std::invalid_argument(std::string(name) + " is not positive semidefinite");
    }
}

}  // namespace

ImuCovariancePropagation buildImuCovariancePropagation(
    const Eigen::Matrix<double, 15, 15>& jacobian_i,
    const Eigen::Matrix<double, 15, 15>& jacobian_j, const Eigen::Matrix<double, 15, 1>& residual) {
    if (!jacobian_i.allFinite() || !jacobian_j.allFinite() || !residual.allFinite()) {
        throw std::invalid_argument("IMU propagation factor contains non-finite values");
    }
    Eigen::FullPivLU<Eigen::Matrix<double, 15, 15>> factor(jacobian_j);
    factor.setThreshold(1e-12);
    if (factor.rank() != 15) {
        throw std::invalid_argument("IMU child-state Jacobian is rank deficient");
    }
    const Eigen::Matrix<double, 15, 15> inverse_j =
        factor.solve(Eigen::Matrix<double, 15, 15>::Identity());
    if (!inverse_j.allFinite()) {
        throw std::runtime_error("IMU child-state Jacobian solve failed");
    }

    ImuCovariancePropagation propagation;
    propagation.F = -inverse_j * jacobian_i;
    propagation.Q = inverse_j * inverse_j.transpose();
    propagation.Q = 0.5 * (propagation.Q + propagation.Q.transpose()).eval();
    propagation.offset = -inverse_j * residual;
    Eigen::LLT<Eigen::Matrix<double, 15, 15>> covariance_factor(propagation.Q);
    if (covariance_factor.info() != Eigen::Success || !propagation.F.allFinite() ||
        !propagation.offset.allFinite()) {
        throw std::runtime_error("IMU propagation covariance is not positive definite");
    }
    return propagation;
}

Eigen::MatrixXd appendImuPropagatedCovariance(
    const Eigen::MatrixXd& prior_covariance, int parent_state_column,
    const ImuCovariancePropagation& propagation) {
    return insertImuPropagatedCovariance(
        prior_covariance, parent_state_column, static_cast<int>(prior_covariance.cols()),
        propagation);
}

Eigen::MatrixXd insertImuPropagatedCovariance(
    const Eigen::MatrixXd& prior_covariance, int parent_state_column, int child_state_column,
    const ImuCovariancePropagation& propagation) {
    constexpr int state_size = 15;
    requireCovariance(prior_covariance, "Prior covariance");
    requireCovariance(propagation.Q, "IMU process covariance");
    if (parent_state_column < 0 || parent_state_column + state_size > prior_covariance.cols() ||
        child_state_column < 0 || child_state_column > prior_covariance.cols() ||
        !propagation.F.allFinite() || !propagation.offset.allFinite()) {
        throw std::invalid_argument("IMU covariance propagation layout is invalid");
    }

    const Eigen::Index prior_size = prior_covariance.rows();
    Eigen::MatrixXd appended =
        Eigen::MatrixXd::Zero(prior_size + state_size, prior_size + state_size);
    appended.topLeftCorner(prior_size, prior_size) = prior_covariance;
    const Eigen::MatrixXd prior_child_cross =
        prior_covariance.middleCols(parent_state_column, state_size) * propagation.F.transpose();
    appended.topRightCorner(prior_size, state_size) = prior_child_cross;
    appended.bottomLeftCorner(state_size, prior_size) = prior_child_cross.transpose();
    appended.bottomRightCorner(state_size, state_size) =
        propagation.F *
            prior_covariance.block(
                parent_state_column, parent_state_column, state_size, state_size) *
            propagation.F.transpose() +
        propagation.Q;
    Eigen::MatrixXd propagated(appended.rows(), appended.cols());
    std::vector<int> source_columns;
    source_columns.reserve(static_cast<size_t>(appended.cols()));
    for (int column = 0; column < child_state_column; ++column) {
        source_columns.push_back(column);
    }
    for (int d = 0; d < state_size; ++d) {
        source_columns.push_back(static_cast<int>(prior_size) + d);
    }
    for (int column = child_state_column; column < prior_size; ++column) {
        source_columns.push_back(column);
    }
    for (int row = 0; row < propagated.rows(); ++row) {
        for (int column = 0; column < propagated.cols(); ++column) {
            propagated(row, column) = appended(
                source_columns[static_cast<size_t>(row)],
                source_columns[static_cast<size_t>(column)]);
        }
    }
    propagated = 0.5 * (propagated + propagated.transpose()).eval();
    requireCovariance(propagated, "Propagated covariance");
    return propagated;
}

}  // namespace tassel_core
