#ifndef TASSEL_CORE_OPTIMIZER_OPTIMIZER_H_
#define TASSEL_CORE_OPTIMIZER_OPTIMIZER_H_

#include "linearization/linearization_abs_qr.h"

namespace tassel_core {

struct OptimizerResult {
    int num_iterations = 0;
    int num_rejected = 0;
    double final_error = 0;
    bool converged = false;
    double spend_time = 0.0;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual OptimizerResult optimize(LinearizationAbsQR* linearization) = 0;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_OPTIMIZER_OPTIMIZER_H_
