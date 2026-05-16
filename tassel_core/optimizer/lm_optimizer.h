#ifndef TASSEL_CORE_OPTIMIZER_LM_OPTIMIZER_H_
#define TASSEL_CORE_OPTIMIZER_LM_OPTIMIZER_H_

#include "optimizer/optimizer.h"
#include "optimizer/optimizer_option.h"

namespace tassel_core {

class LMOptimizer : public Optimizer {
public:
    LMOptimizer(const LMOptions& options = LMOptions());

    void optimize(LinearizationAbsQR* linearization) override;

private:
    LMOptions options_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_OPTIMIZER_LM_OPTIMIZER_H_
