#ifndef TASSEL_CORE_SOLVER_PARAMETER_BLOCK_HOLD_CALLBACK_H_
#define TASSEL_CORE_SOLVER_PARAMETER_BLOCK_HOLD_CALLBACK_H_

#include <ceres/ceres.h>

#include <vector>

namespace tassel_core {

struct HeldParameterBlock {
    double* data = nullptr;
    int size = 0;
};

class ParameterBlockHoldCallback final : public ceres::IterationCallback {
public:
    explicit ParameterBlockHoldCallback(std::vector<HeldParameterBlock> blocks);

    ceres::CallbackReturnType operator()(const ceres::IterationSummary&) override;
    void restore() const;

private:
    // data 由 Ceres Problem 中的参数块持有，生命周期必须覆盖回调和最终 restore。
    std::vector<HeldParameterBlock> blocks_;
    std::vector<std::vector<double>> snapshots_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_SOLVER_PARAMETER_BLOCK_HOLD_CALLBACK_H_
