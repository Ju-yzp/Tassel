#ifndef TASSEL_CORE_INITIAL_DYNAMIC_INITIALIZER_H_
#define TASSEL_CORE_INITIAL_DYNAMIC_INITIALIZER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "factor/midpoint_integrator.h"
#include "parameters/parameters.h"

namespace tassel_core {

class FeatureManager;
struct State;

class DynamicInitializer {
public:
    using PreintegratorStorage = std::vector<MidPointIntegrator>;

    DynamicInitializer(
        const tassel_tools::Parameters& params, std::shared_ptr<State> state,
        std::shared_ptr<FeatureManager> feature_manager, PreintegratorStorage& preintegrators,
        const Eigen::Matrix<double, 18, 18>& noise);

    bool initialize();

private:
    const tassel_tools::Parameters& params_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> feature_manager_;
    PreintegratorStorage& preintegrators_;
    Eigen::Matrix<double, 18, 18> noise_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_INITIAL_DYNAMIC_INITIALIZER_H_
