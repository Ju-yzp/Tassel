#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_

#include "loss_fuction/loss_fuction_base.h"

namespace tassel_core {

struct EstimatorOption {
    // ── optimization ────────────────────────────────────────────────────────
    bool optimize_enabled = true;
    int num_iterations = 10;
    double lambda_initial = 1e-3;

    // ── marginalization ─────────────────────────────────────────────────────
    bool marginalization_enabled = true;

    // ── robust loss (reprojection) ──────────────────────────────────────────
    LossVariant reprojection_loss = TrivialLoss{};

    // ── depth robust kernel ─────────────────────────────────────────────────
    DepthLoss depth_loss = DepthLoss::none();

    double min_depth = 0.05;
    double max_depth = 10.0;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_
