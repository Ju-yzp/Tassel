#include "lm_optimizer.h"

#include <Eigen/Cholesky>
#include <cmath>

namespace tassel_core {

LMOptimizer::LMOptimizer(const LMOptions& options) : options_(options) {}

void LMOptimizer::optimize(LinearizationAbsQR* linearization) {
    double lambda = options_.lambda_initial;
    double lambda_vee = options_.initial_vee;

    for (int iter = 0; iter < options_.max_iterations; iter++) {
        // ---- Step 1: linearize at current point ----
        double error_0 = linearization->linearizeProbelm();
        linearization->performQR();

        // ---- Inner backtracking loop ----
        for (int j = 0; iter < options_.max_iterations; j++) {
            // ---- Step 2: build reduced Hessian ----
            Eigen::MatrixXd H;
            Eigen::VectorXd b;
            linearization->get_dense_H_b(H, b);

            // ---- Step 3: apply LM damping and solve ----
            Eigen::VectorXd Hdiag_lambda = (H.diagonal() * lambda).cwiseMax(options_.lambda_min);
            Eigen::MatrixXd H_aug = H;
            H_aug.diagonal() += Hdiag_lambda;

            Eigen::LDLT<Eigen::MatrixXd> ldlt(H_aug);
            if (ldlt.info() != Eigen::Success) {
                lambda = std::min(lambda_vee * lambda, options_.lambda_max);
                lambda_vee *= options_.vee_factor;
                continue;
            }

            Eigen::VectorXd inc_raw = ldlt.solve(b);  // inc_raw = H_aug^{-1} * J^T r = -dx
            Eigen::VectorXd inc = -inc_raw;           // actual parameter increment dx

            if (!inc.array().isFinite().all()) {
                lambda = std::min(lambda_vee * lambda, options_.lambda_max);
                lambda_vee *= options_.vee_factor;
                continue;
            }

            // ---- Step 4: save state, apply increment, back-substitute ----
            linearization->saveState();
            linearization->applyPoseInc(inc);

            int pose_dim = linearization->getPoseDim();
            double l_diff = linearization->backSubstitute(inc.head(pose_dim));

            // ---- Step 5: compute new error and check acceptance ----
            double error_new = linearization->computeError();
            double f_diff = error_0 - error_new;
            double relative_decrease = f_diff / l_diff;

            if (l_diff > 0 && relative_decrease > 0) {
                // Accept
                lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * relative_decrease - 1.0, 3));
                lambda = std::max(options_.lambda_min, lambda);
                lambda_vee = options_.initial_vee;

                if (f_diff < 1e-6) {
                    return;  // converged (function tolerance)
                }
                break;
            } else {
                // Reject: restore state and increase lambda
                linearization->restoreState();

                lambda = std::min(lambda_vee * lambda, options_.lambda_max);
                lambda_vee *= options_.vee_factor;

                if (lambda >= options_.lambda_max) {
                    return;  // diverged, give up
                }
                // continue inner loop with larger lambda
            }
        }
    }
}

}  // namespace tassel_core
