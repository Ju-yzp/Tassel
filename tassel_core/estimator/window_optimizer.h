#ifndef TASSEL_CORE_ESTIMATOR_WINDOW_OPTIMIZER_H_
#define TASSEL_CORE_ESTIMATOR_WINDOW_OPTIMIZER_H_

#include <memory>
#include <utility>
#include <vector>

#include <ceres/ceres.h>

#include "factor/integrator_base.h"
#include "frond_end/feature.h"

namespace tassel_tools {
struct Parameters;
}

namespace tassel_core {

struct MargLinData;
struct State;

struct WindowOptimizationResult {
    std::vector<std::pair<int, double>> feature_depths;
    std::vector<int> visual_factors_per_frame;
};

class WindowOptimizer {
public:
    WindowOptimizer(const tassel_tools::Parameters& params, std::shared_ptr<State> state);

    // features 是独立值快照；求解器写入 State 的参数缓存，并按快照 ID 返回深度。
    // Ba 始终参与联合求解；hold_accel_bias 为 true 时仅恢复其求解前均值。
    WindowOptimizationResult solve(
        const std::vector<std::pair<int, Feature>>& features,
        std::vector<MidPointIntegrator>& preintegrators, const MargLinData* prior,
        bool hold_accel_bias);

    WindowOptimizationResult solve(
        const std::vector<std::pair<int, Feature>>& features,
        std::vector<EulerIntegrator>& preintegrators, const MargLinData* prior,
        bool hold_accel_bias);

private:
    template <typename Integrator>
    WindowOptimizationResult solveImpl(
        const std::vector<std::pair<int, Feature>>& features,
        std::vector<Integrator>& preintegrators, const MargLinData* prior, bool hold_accel_bias);

    const tassel_tools::Parameters& params_;
    std::shared_ptr<State> state_;
    std::unique_ptr<ceres::Context> ceres_context_{ceres::Context::Create()};
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_ESTIMATOR_WINDOW_OPTIMIZER_H_
