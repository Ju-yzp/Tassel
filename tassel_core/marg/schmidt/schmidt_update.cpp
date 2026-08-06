#include "marg/schmidt/schmidt_update.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tassel_core {
namespace {

void requireFiniteMatrix(const Eigen::MatrixXd& matrix, const char* name) {
    if (!matrix.allFinite()) {
        throw std::invalid_argument(std::string(name) + " contains non-finite values");
    }
}

void requireFiniteVector(const Eigen::VectorXd& vector, const char* name) {
    if (!vector.allFinite()) {
        throw std::invalid_argument(std::string(name) + " contains non-finite values");
    }
}

Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposeSymmetric(
    const Eigen::MatrixXd& matrix, const char* name) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument(std::string(name) + " must be square");
    }
    requireFiniteMatrix(matrix, name);
    const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(std::string(name) + " eigendecomposition failed");
    }
    return solver;
}

double eigenvalueThreshold(const Eigen::VectorXd& eigenvalues) {
    return 1e-12 * std::max(1.0, eigenvalues.cwiseAbs().maxCoeff());
}

}  // namespace

Eigen::MatrixXd covarianceFromFullRankSqrtInformation(const Eigen::MatrixXd& sqrt_information) {
    if (!sqrt_information.allFinite()) {
        throw std::invalid_argument("Square-root information contains non-finite values");
    }
    const Eigen::Index state_size = sqrt_information.cols();
    if (state_size == 0) {
        return Eigen::MatrixXd(0, 0);
    }
    const Eigen::MatrixXd information = sqrt_information.transpose() * sqrt_information;
    const auto solver = decomposeSymmetric(information, "Square-root information");
    const double threshold = eigenvalueThreshold(solver.eigenvalues());
    Eigen::VectorXd inverse_eigenvalues(state_size);
    for (Eigen::Index i = 0; i < state_size; ++i) {
        const double eigenvalue = solver.eigenvalues()[i];
        if (eigenvalue <= threshold) {
            throw std::invalid_argument("Square-root information is not full column rank");
        }
        inverse_eigenvalues[i] = 1.0 / eigenvalue;
    }
    return solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
           solver.eigenvectors().transpose();
}

Eigen::MatrixXd sqrtInformationFromPositiveDefiniteCovariance(const Eigen::MatrixXd& covariance) {
    if (covariance.rows() != covariance.cols() || !covariance.allFinite()) {
        throw std::invalid_argument("Covariance has invalid dimensions or non-finite values");
    }
    const Eigen::Index state_size = covariance.rows();
    if (state_size == 0) {
        return Eigen::MatrixXd(0, 0);
    }
    const auto solver = decomposeSymmetric(covariance, "Covariance");
    const double threshold = eigenvalueThreshold(solver.eigenvalues());
    Eigen::VectorXd inverse_eigenvalues(state_size);
    for (Eigen::Index i = 0; i < state_size; ++i) {
        if (solver.eigenvalues()[i] <= threshold) {
            throw std::invalid_argument("Covariance is not positive definite");
        }
        inverse_eigenvalues[i] = 1.0 / solver.eigenvalues()[i];
    }
    return inverse_eigenvalues.cwiseSqrt().asDiagonal() * solver.eigenvectors().transpose();
}

Eigen::MatrixXd pseudoCovarianceFromSqrtInformation(const Eigen::MatrixXd& sqrt_information) {
    if (!sqrt_information.allFinite()) {
        throw std::invalid_argument("Square-root information contains non-finite values");
    }
    const Eigen::Index state_size = sqrt_information.cols();
    if (state_size == 0) {
        return Eigen::MatrixXd(0, 0);
    }
    const Eigen::MatrixXd information = sqrt_information.transpose() * sqrt_information;
    const auto solver = decomposeSymmetric(information, "Square-root information");
    const double threshold = eigenvalueThreshold(solver.eigenvalues());
    Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(state_size);
    for (Eigen::Index i = 0; i < state_size; ++i) {
        if (solver.eigenvalues()[i] < -threshold) {
            throw std::invalid_argument("Square-root information is not positive semidefinite");
        }
        if (solver.eigenvalues()[i] > threshold) {
            inverse_eigenvalues[i] = 1.0 / solver.eigenvalues()[i];
        }
    }
    return solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
           solver.eigenvectors().transpose();
}

Eigen::MatrixXd sqrtInformationFromPseudoCovariance(const Eigen::MatrixXd& pseudo_covariance) {
    if (pseudo_covariance.rows() != pseudo_covariance.cols() || !pseudo_covariance.allFinite()) {
        throw std::invalid_argument(
            "Pseudo-covariance has invalid dimensions or non-finite values");
    }
    const Eigen::Index state_size = pseudo_covariance.rows();
    if (state_size == 0) {
        return Eigen::MatrixXd(0, 0);
    }
    const auto solver = decomposeSymmetric(pseudo_covariance, "Pseudo-covariance");
    const double threshold = eigenvalueThreshold(solver.eigenvalues());
    std::vector<Eigen::Index> positive_indices;
    for (Eigen::Index i = 0; i < state_size; ++i) {
        if (solver.eigenvalues()[i] < -threshold) {
            throw std::invalid_argument("Pseudo-covariance is not positive semidefinite");
        }
        if (solver.eigenvalues()[i] > threshold) {
            positive_indices.push_back(i);
        }
    }
    Eigen::MatrixXd sqrt_information(
        static_cast<Eigen::Index>(positive_indices.size()), state_size);
    for (Eigen::Index row = 0; row < sqrt_information.rows(); ++row) {
        const Eigen::Index index = positive_indices[static_cast<size_t>(row)];
        sqrt_information.row(row) =
            solver.eigenvectors().col(index).transpose() / std::sqrt(solver.eigenvalues()[index]);
    }
    return sqrt_information;
}

SchmidtUpdateResult applySchmidtUpdate(
    const Eigen::MatrixXd& active_covariance,
    const Eigen::MatrixXd& active_schmidt_cross_covariance,
    const Eigen::MatrixXd& schmidt_covariance, const Eigen::MatrixXd& active_jacobian,
    const Eigen::MatrixXd& schmidt_jacobian, const Eigen::MatrixXd& measurement_covariance,
    const Eigen::VectorXd& residual) {
    const Eigen::Index active_size = active_covariance.rows();
    const Eigen::Index schmidt_size = schmidt_covariance.rows();
    const Eigen::Index measurement_size = residual.size();
    if (active_covariance.cols() != active_size || schmidt_covariance.cols() != schmidt_size ||
        active_schmidt_cross_covariance.rows() != active_size ||
        active_schmidt_cross_covariance.cols() != schmidt_size ||
        active_jacobian.rows() != measurement_size || active_jacobian.cols() != active_size ||
        schmidt_jacobian.rows() != measurement_size || schmidt_jacobian.cols() != schmidt_size ||
        measurement_covariance.rows() != measurement_size ||
        measurement_covariance.cols() != measurement_size) {
        throw std::invalid_argument("Schmidt update dimensions are inconsistent");
    }
    requireFiniteMatrix(active_covariance, "active covariance");
    requireFiniteMatrix(active_schmidt_cross_covariance, "active-Schmidt cross covariance");
    requireFiniteMatrix(schmidt_covariance, "Schmidt covariance");
    requireFiniteMatrix(active_jacobian, "active Jacobian");
    requireFiniteMatrix(schmidt_jacobian, "Schmidt Jacobian");
    requireFiniteMatrix(measurement_covariance, "measurement covariance");
    requireFiniteVector(residual, "residual");

    const Eigen::MatrixXd active_cross =
        active_covariance * active_jacobian.transpose() +
        active_schmidt_cross_covariance * schmidt_jacobian.transpose();
    const Eigen::MatrixXd schmidt_cross =
        active_schmidt_cross_covariance.transpose() * active_jacobian.transpose() +
        schmidt_covariance * schmidt_jacobian.transpose();
    const Eigen::MatrixXd innovation_covariance =
        active_jacobian * active_covariance * active_jacobian.transpose() +
        active_jacobian * active_schmidt_cross_covariance * schmidt_jacobian.transpose() +
        schmidt_jacobian * active_schmidt_cross_covariance.transpose() *
            active_jacobian.transpose() +
        schmidt_jacobian * schmidt_covariance * schmidt_jacobian.transpose() +
        measurement_covariance;

    Eigen::LDLT<Eigen::MatrixXd> factor(innovation_covariance);
    if (factor.info() != Eigen::Success || !factor.isPositive()) {
        throw std::runtime_error("Schmidt innovation covariance factorization failed");
    }

    const Eigen::MatrixXd gain = factor.solve(active_cross.transpose()).transpose();
    SchmidtUpdateResult result;
    result.active_delta = gain * residual;
    result.active_covariance = active_covariance - gain * active_cross.transpose();
    result.active_covariance =
        0.5 * (result.active_covariance + result.active_covariance.transpose());
    result.active_schmidt_cross_covariance =
        active_schmidt_cross_covariance - gain * schmidt_cross.transpose();
    result.schmidt_covariance = schmidt_covariance;
    return result;
}

SchmidtJointUpdateResult applySchmidtUpdateToJointCovariance(
    const Eigen::MatrixXd& covariance, const Eigen::MatrixXd& jacobian,
    const Eigen::MatrixXd& measurement_covariance, const Eigen::VectorXd& residual,
    const std::vector<int>& schmidt_indices) {
    const Eigen::Index state_size = covariance.rows();
    if (covariance.cols() != state_size || jacobian.cols() != state_size ||
        jacobian.rows() != residual.size() || measurement_covariance.rows() != residual.size() ||
        measurement_covariance.cols() != residual.size()) {
        throw std::invalid_argument("Joint Schmidt update dimensions are inconsistent");
    }
    std::vector<bool> is_schmidt(static_cast<size_t>(state_size), false);
    for (int index : schmidt_indices) {
        if (index < 0 || index >= state_size || is_schmidt[static_cast<size_t>(index)]) {
            throw std::invalid_argument("Joint Schmidt index is invalid or duplicated");
        }
        is_schmidt[static_cast<size_t>(index)] = true;
    }
    if (schmidt_indices.empty() || schmidt_indices.size() == static_cast<size_t>(state_size)) {
        throw std::invalid_argument("Joint Schmidt update requires both variable subsets");
    }
    std::vector<int> active_indices;
    active_indices.reserve(static_cast<size_t>(state_size) - schmidt_indices.size());
    for (int index = 0; index < state_size; ++index) {
        if (!is_schmidt[static_cast<size_t>(index)]) {
            active_indices.push_back(index);
        }
    }

    Eigen::MatrixXd paa(active_indices.size(), active_indices.size());
    Eigen::MatrixXd pas(active_indices.size(), schmidt_indices.size());
    Eigen::MatrixXd pss(schmidt_indices.size(), schmidt_indices.size());
    Eigen::MatrixXd ha(jacobian.rows(), active_indices.size());
    Eigen::MatrixXd hs(jacobian.rows(), schmidt_indices.size());
    for (size_t i = 0; i < active_indices.size(); ++i) {
        ha.col(static_cast<Eigen::Index>(i)) = jacobian.col(active_indices[i]);
        for (size_t j = 0; j < active_indices.size(); ++j) {
            paa(i, j) = covariance(active_indices[i], active_indices[j]);
        }
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            pas(i, j) = covariance(active_indices[i], schmidt_indices[j]);
        }
    }
    for (size_t i = 0; i < schmidt_indices.size(); ++i) {
        hs.col(static_cast<Eigen::Index>(i)) = jacobian.col(schmidt_indices[i]);
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            pss(i, j) = covariance(schmidt_indices[i], schmidt_indices[j]);
        }
    }
    const SchmidtUpdateResult update =
        applySchmidtUpdate(paa, pas, pss, ha, hs, measurement_covariance, residual);

    SchmidtJointUpdateResult result;
    result.delta = Eigen::VectorXd::Zero(state_size);
    result.covariance = covariance;
    for (size_t i = 0; i < active_indices.size(); ++i) {
        result.delta[active_indices[i]] = update.active_delta[static_cast<Eigen::Index>(i)];
        for (size_t j = 0; j < active_indices.size(); ++j) {
            result.covariance(active_indices[i], active_indices[j]) =
                update.active_covariance(i, j);
        }
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            result.covariance(active_indices[i], schmidt_indices[j]) =
                update.active_schmidt_cross_covariance(i, j);
            result.covariance(schmidt_indices[j], active_indices[i]) =
                update.active_schmidt_cross_covariance(i, j);
        }
    }
    for (size_t i = 0; i < schmidt_indices.size(); ++i) {
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            result.covariance(schmidt_indices[i], schmidt_indices[j]) =
                update.schmidt_covariance(i, j);
        }
    }
    return result;
}

SchmidtSqrtPrior buildSchmidtUpdatedSqrtPrior(
    const Eigen::MatrixXd& prior_sqrt_information, const Eigen::VectorXd& prior_residual,
    const Eigen::MatrixXd& likelihood_sqrt_information, const Eigen::VectorXd& likelihood_residual,
    const std::vector<int>& schmidt_indices) {
    const Eigen::Index state_size = prior_sqrt_information.cols();
    if (prior_sqrt_information.rows() != prior_residual.size() ||
        likelihood_sqrt_information.cols() != state_size ||
        likelihood_sqrt_information.rows() != likelihood_residual.size()) {
        throw std::invalid_argument("Schmidt square-root update dimensions are inconsistent");
    }
    requireFiniteVector(prior_residual, "prior residual");
    requireFiniteMatrix(likelihood_sqrt_information, "likelihood square-root information");
    requireFiniteVector(likelihood_residual, "likelihood residual");

    std::vector<bool> is_schmidt(static_cast<size_t>(state_size), false);
    for (int index : schmidt_indices) {
        if (index < 0 || index >= state_size || is_schmidt[static_cast<size_t>(index)]) {
            throw std::invalid_argument("Schmidt index is invalid or duplicated");
        }
        is_schmidt[static_cast<size_t>(index)] = true;
    }
    if (schmidt_indices.empty() || schmidt_indices.size() == static_cast<size_t>(state_size)) {
        throw std::invalid_argument("Schmidt update requires active and Schmidt variables");
    }

    std::vector<int> active_indices;
    active_indices.reserve(static_cast<size_t>(state_size) - schmidt_indices.size());
    for (int index = 0; index < state_size; ++index) {
        if (!is_schmidt[static_cast<size_t>(index)]) {
            active_indices.push_back(index);
        }
    }

    const Eigen::MatrixXd prior_covariance =
        covarianceFromFullRankSqrtInformation(prior_sqrt_information);
    const Eigen::VectorXd prior_mean =
        -prior_covariance * prior_sqrt_information.transpose() * prior_residual;
    Eigen::MatrixXd paa(active_indices.size(), active_indices.size());
    Eigen::MatrixXd pas(active_indices.size(), schmidt_indices.size());
    Eigen::MatrixXd pss(schmidt_indices.size(), schmidt_indices.size());
    Eigen::MatrixXd ha(likelihood_sqrt_information.rows(), active_indices.size());
    Eigen::MatrixXd hs(likelihood_sqrt_information.rows(), schmidt_indices.size());
    for (size_t i = 0; i < active_indices.size(); ++i) {
        ha.col(static_cast<Eigen::Index>(i)) = likelihood_sqrt_information.col(active_indices[i]);
        for (size_t j = 0; j < active_indices.size(); ++j) {
            paa(i, j) = prior_covariance(active_indices[i], active_indices[j]);
        }
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            pas(i, j) = prior_covariance(active_indices[i], schmidt_indices[j]);
        }
    }
    for (size_t i = 0; i < schmidt_indices.size(); ++i) {
        hs.col(static_cast<Eigen::Index>(i)) = likelihood_sqrt_information.col(schmidt_indices[i]);
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            pss(i, j) = prior_covariance(schmidt_indices[i], schmidt_indices[j]);
        }
    }

    const Eigen::VectorXd innovation =
        -(likelihood_sqrt_information * prior_mean + likelihood_residual);
    const SchmidtUpdateResult update = applySchmidtUpdate(
        paa, pas, pss, ha, hs,
        Eigen::MatrixXd::Identity(
            likelihood_sqrt_information.rows(), likelihood_sqrt_information.rows()),
        innovation);

    Eigen::VectorXd posterior_mean = prior_mean;
    Eigen::MatrixXd posterior_covariance = prior_covariance;
    for (size_t i = 0; i < active_indices.size(); ++i) {
        posterior_mean[active_indices[i]] += update.active_delta[static_cast<Eigen::Index>(i)];
        for (size_t j = 0; j < active_indices.size(); ++j) {
            posterior_covariance(active_indices[i], active_indices[j]) =
                update.active_covariance(i, j);
        }
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            posterior_covariance(active_indices[i], schmidt_indices[j]) =
                update.active_schmidt_cross_covariance(i, j);
            posterior_covariance(schmidt_indices[j], active_indices[i]) =
                update.active_schmidt_cross_covariance(i, j);
        }
    }
    for (size_t i = 0; i < schmidt_indices.size(); ++i) {
        for (size_t j = 0; j < schmidt_indices.size(); ++j) {
            posterior_covariance(schmidt_indices[i], schmidt_indices[j]) =
                update.schmidt_covariance(i, j);
        }
    }

    SchmidtSqrtPrior result;
    result.H = sqrtInformationFromPositiveDefiniteCovariance(posterior_covariance);
    result.b = -result.H * posterior_mean;
    return result;
}

}  // namespace tassel_core
