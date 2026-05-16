#ifndef TASSEL_CORE_OPTIMIZER_OPTIMIZER_H_
#define TASSEL_CORE_OPTIMIZER_OPTIMIZER_H_

#include "linearization/linearization_abs_qr.h"
#include "summary.h"

namespace tassel_core {

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void optimize(LinearizationAbsQR* linearization) = 0;

    void log_summary(LogType type = LogType::CONCISE) { summary_.log(type); }

protected:
    Summary summary_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_OPTIMIZER_OPTIMIZER_H_
