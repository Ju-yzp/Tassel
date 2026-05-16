#include "optimizer/lm_optimizer.h"

#include <spdlog/spdlog.h>
#include <Eigen/Cholesky>
#include <cmath>

#include "optimizer/summary.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/timer.h"

namespace tassel_core {

LMOptimizer::LMOptimizer(const LMOptions& options) : options_(options) {}

void LMOptimizer::optimize(LinearizationAbsQR* linearization) {
    double lambda = options_.lambda_initial;
    double lambda_vee = options_.initial_vee;

    int num_iter = 0;
    int num_rejected = 0;
    bool converged = false;
    bool terminated = false;
    bool first_linearization = true;

    tassel_utils::Timer timer("optimizer");
    timer.start();
    for (; num_iter < options_.max_iterations && !terminated;) {
        // ── Phase 1: linearize residuals and marginalize landmarks via QR ──
        bool numerically_valid = false;
        double error_0 = linearization->linearizeProbelm(&numerically_valid);
        if (first_linearization) {
            summary_.init_cost = error_0;
            first_linearization = false;
        }
        TASSEL_ASSERT(numerically_valid && "numerical failure during linearization");
        linearization->performQR();

        // ── Phase 2: inner LM backtracking loop ──
        for (int j = 0; num_iter < options_.max_iterations && !terminated; j++) {
            // Build reduced camera system
            Eigen::MatrixXd H;
            Eigen::VectorXd b;
            linearization->get_dense_H_b(H, b);

            // Apply LM damping and solve, retry up to 3x on non-finite result
            Eigen::VectorXd inc;
            {
                bool inc_valid = false;
                int retry = 0;
                constexpr int kMaxRetry = 3;

                while (retry < kMaxRetry && !inc_valid) {
                    Eigen::VectorXd Hdiag =
                        (H.diagonal().array() * lambda).cwiseMax(options_.lambda_min);
                    Eigen::MatrixXd H_damped = H;
                    H_damped.diagonal() += Hdiag;

                    Eigen::LDLT<Eigen::MatrixXd> ldlt(H_damped);
                    if (ldlt.info() == Eigen::Success) {
                        inc = ldlt.solve(b);
                        if (inc.array().isFinite().all()) {
                            inc_valid = true;
                        }
                    }

                    if (!inc_valid) {
                        lambda = lambda_vee * lambda;
                        lambda_vee *= options_.vee_factor;
                        retry++;
                    }
                }

                if (!inc_valid) {
                    spdlog::warn(
                        "LM iter={} inner={}: inc still invalid after {} retries, "
                        "giving up",
                        num_iter, j, kMaxRetry);
                    num_iter++;
                    num_rejected++;
                    terminated = true;
                    break;
                }
            }

            if (terminated) break;

            // Backup state, negate increment, back-substitute landmarks
            linearization->saveState();
            inc = -inc;
            int pose_dim = linearization->getPoseDim();
            double l_diff = linearization->backSubstitute(inc.head(pose_dim));

            // Apply increment to poses
            linearization->applyPoseInc(inc.head(pose_dim));
            double step_norminf = inc.head(pose_dim).array().abs().maxCoeff();

            double error_new = linearization->computeError();
            double f_diff = error_0 - error_new;

            // Step quality assessment
            bool step_is_valid = l_diff > 0;
            double relative_decrease = step_is_valid ? f_diff / l_diff : 0;
            bool step_is_successful = step_is_valid && relative_decrease > 0;

            PerIterationSummary pis;
            pis.cost = error_new;
            pis.lambda = lambda;
            pis.step = inc.head(pose_dim);

            // spdlog::debug(
            //     "LM iter={} inner={} lambda={:.2e} err0={:.6f} err1={:.6f} "
            //     "f_diff={:.6f} l_diff={:.6f} rel={:.4f} step_norminf={:.4f} {}",
            //     num_iter, j, lambda, error_0, error_new, f_diff, l_diff,
            //     relative_decrease, step_norminf,
            //     step_is_successful ? "accepted" : "rejected");

            pis.rejected = !step_is_successful;
            summary_.summaries.push_back(pis);
            if (step_is_successful) {
                // // Absorb accumulated delta into the linearization point so the
                // // next outer iteration linearizes Jacobians at the current
                // // (correct) pose, rather than the frozen pre-optimization pose.
                // for (auto& p : linearization->getState()->poses) {
                //     p.updateLinearizationPoint();
                // }

                lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * relative_decrease - 1.0, 3));
                lambda = std::max(options_.lambda_min, lambda);
                lambda_vee = options_.initial_vee;

                num_iter++;

                if ((f_diff > 0 && f_diff < 1e-6) || step_norminf < 1e-4) {
                    converged = true;
                    terminated = true;
                }
                break;  // exit inner loop → re-linearize at next outer iteration
            } else {
                linearization->restoreState();

                lambda = std::min(lambda_vee * lambda, options_.lambda_max);
                lambda_vee *= options_.vee_factor;

                num_iter++;
                num_rejected++;

                if (lambda >= options_.lambda_max) {
                    spdlog::warn("LM giving up: lambda reached max ({:.2e})", lambda);
                    terminated = true;
                }
            }
        }
    }

    double final_error = linearization->computeError();
    double spend_time = timer.end();
    summary_.actual_iterations = num_iter;
    summary_.num_rejected = num_rejected;
    summary_.final_cost = final_error;
    summary_.spend_time = spend_time;
    summary_.converged = converged;
}

}  // namespace tassel_core
