#include "solver/linearized_system.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tassel_core {

LinearizedSystem::LinearizedSystem(Eigen::MatrixXd jacobian, Eigen::VectorXd residual)
    : jacobian_(std::move(jacobian)), residual_(std::move(residual)) {
    if (jacobian_.rows() <= 0 || jacobian_.cols() <= 0 ||
        jacobian_.rows() != residual_.rows()) {
        throw std::invalid_argument("Linearized system dimensions are invalid");
    }
    if (!jacobian_.allFinite() || !residual_.allFinite()) {
        throw std::invalid_argument("Linearized system contains non-finite values");
    }
}

double LinearizedSystem::cost() const {
    return 0.5 * residual_.squaredNorm();
}

double LinearizedSystem::costAt(const Eigen::VectorXd& delta) const {
    if (delta.size() != jacobian_.cols() || !delta.allFinite()) {
        throw std::invalid_argument("Linearized system delta is invalid");
    }
    return 0.5 * (residual_ + jacobian_ * delta).squaredNorm();
}

Eigen::MatrixXd LinearizedSystem::hessian() const {
    return jacobian_.transpose() * jacobian_;
}

Eigen::VectorXd LinearizedSystem::gradient() const {
    return jacobian_.transpose() * residual_;
}

LinearStep solveDampedNormalStep(
    const LinearizedSystem& system, const DiagonalDamping& damping) {
    if (!std::isfinite(damping.lambda) || !std::isfinite(damping.min_diagonal) ||
        damping.lambda < 0.0 || damping.min_diagonal < 0.0) {
        throw std::invalid_argument("Linearized system damping is invalid");
    }

    const Eigen::MatrixXd hessian = system.hessian();
    const Eigen::VectorXd gradient = system.gradient();
    Eigen::VectorXd damping_diagonal(hessian.rows());
    for (Eigen::Index i = 0; i < hessian.rows(); ++i) {
        damping_diagonal[i] =
            std::max(damping.lambda * hessian(i, i), damping.min_diagonal);
    }

    Eigen::MatrixXd damped_hessian = hessian;
    damped_hessian.diagonal() += damping_diagonal;
    const Eigen::LLT<Eigen::MatrixXd> factor(damped_hessian);
    if (factor.info() != Eigen::Success) {
        throw std::runtime_error("Damped normal system is not positive definite");
    }

    LinearStep result;
    result.delta = factor.solve(-gradient);
    if (factor.info() != Eigen::Success || !result.delta.allFinite()) {
        throw std::runtime_error("Damped normal system solve failed");
    }
    result.damping_diagonal = std::move(damping_diagonal);
    result.initial_cost = system.cost();
    result.model_cost = system.costAt(result.delta);
    result.predicted_reduction = result.initial_cost - result.model_cost;
    if (!std::isfinite(result.predicted_reduction)) {
        throw std::runtime_error("Linearized model reduction is non-finite");
    }
    return result;
}

}  // namespace tassel_core
