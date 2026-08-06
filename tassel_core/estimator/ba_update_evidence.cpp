#include "ba_update_evidence.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tassel_core {

BaUpdateEvidence computeBaUpdateEvidence(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& residual,
    const std::array<int, 3>& ba_columns) {
    if (jacobian.rows() != residual.size() || jacobian.cols() < 3 || !jacobian.allFinite() ||
        !residual.allFinite()) {
        throw std::invalid_argument("Invalid Ba evidence linear system");
    }
    std::vector<bool> is_ba(static_cast<size_t>(jacobian.cols()), false);
    for (const int col : ba_columns) {
        if (col < 0 || col >= jacobian.cols() || is_ba[static_cast<size_t>(col)]) {
            throw std::invalid_argument("Invalid Ba evidence column mapping");
        }
        is_ba[static_cast<size_t>(col)] = true;
    }

    Eigen::MatrixXd candidate_basis = Eigen::MatrixXd::Zero(jacobian.cols(), 3);
    Eigen::MatrixXd nuisance_basis = Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.cols() - 3);
    int x_col = 0;
    int b_col = 0;
    for (Eigen::Index col = 0; col < jacobian.cols(); ++col) {
        if (is_ba[static_cast<size_t>(col)]) {
            candidate_basis(col, b_col++) = 1.0;
        } else {
            nuisance_basis(col, x_col++) = 1.0;
        }
    }
    return computeBaUpdateEvidence(jacobian, residual, candidate_basis, nuisance_basis);
}

BaUpdateEvidence computeBaUpdateEvidence(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& candidate_basis, const Eigen::MatrixXd& nuisance_basis) {
    if (jacobian.rows() != residual.size() || candidate_basis.rows() != jacobian.cols() ||
        candidate_basis.cols() != 3 || nuisance_basis.rows() != jacobian.cols() ||
        !jacobian.allFinite() || !residual.allFinite() || !candidate_basis.allFinite() ||
        !nuisance_basis.allFinite()) {
        throw std::invalid_argument("Invalid Ba evidence tangent mapping");
    }
    const Eigen::MatrixXd Jx = jacobian * nuisance_basis;
    const Eigen::MatrixXd Jb = jacobian * candidate_basis;

    Eigen::JacobiSVD<Eigen::MatrixXd> x_svd(Jx, Eigen::ComputeFullU);
    const Eigen::VectorXd x_values = x_svd.singularValues();
    const double x_max = x_values.size() > 0 ? x_values.maxCoeff() : 0.0;
    const double x_threshold = std::sqrt(std::numeric_limits<double>::epsilon()) *
                               static_cast<double>(std::max(Jx.rows(), Jx.cols())) * x_max;
    int x_rank = 0;
    for (Eigen::Index i = 0; i < x_values.size(); ++i) {
        if (x_values[i] > x_threshold) {
            ++x_rank;
        }
    }

    const Eigen::MatrixXd null_basis = x_svd.matrixU().rightCols(jacobian.rows() - x_rank);
    const Eigen::MatrixXd conditional_jacobian = null_basis.transpose() * Jb;
    const Eigen::VectorXd conditional_residual = null_basis.transpose() * residual;

    BaUpdateEvidence evidence;
    evidence.information = conditional_jacobian.transpose() * conditional_jacobian;
    evidence.gradient = conditional_jacobian.transpose() * conditional_residual;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(evidence.information);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
        throw std::runtime_error("Failed to decompose Ba conditional information");
    }
    const double max_value = std::max(0.0, solver.eigenvalues().maxCoeff());
    const double threshold = std::sqrt(std::numeric_limits<double>::epsilon()) * max_value;
    double min_value = std::numeric_limits<double>::infinity();
    Eigen::Vector3d inverse_values = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        if (solver.eigenvalues()[i] > threshold) {
            inverse_values[i] = 1.0 / solver.eigenvalues()[i];
            min_value = std::min(min_value, solver.eigenvalues()[i]);
            ++evidence.rank;
        }
    }
    const Eigen::Matrix3d information_inverse =
        solver.eigenvectors() * inverse_values.asDiagonal() * solver.eigenvectors().transpose();
    evidence.increment = -information_inverse * evidence.gradient;
    evidence.cost_reduction = evidence.gradient.dot(information_inverse * evidence.gradient);
    if (evidence.rank == 3) {
        evidence.condition = max_value / min_value;
    }
    return evidence;
}

}  // namespace tassel_core
