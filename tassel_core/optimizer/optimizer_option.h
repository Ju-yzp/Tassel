#ifndef TASSEL_CORE_OPTIMIZER_OPTIMIZER_OPTION_H_
#define TASSEL_CORE_OPTIMIZER_OPTIMIZER_OPTION_H_

namespace tassel_core {

struct LMOptions {
    int max_iterations = 10;
    double lambda_initial = 1e-3;
    double lambda_min = 1e-6;
    double lambda_max = 1e6;
    double vee_factor = 2.0;
    double initial_vee = 2.0;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_OPTIMIZER_OPTIMIZER_OPTION_H_
