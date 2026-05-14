#ifndef TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_

#include <Eigen/Core>
#include <memory>
#include <unordered_map>

#include "estimator/estimator_option.h"
#include "frond_end/feature_manager.h"
#include "marginalization/marg_linearized_data.h"
#include "state/state.h"

namespace tassel_core {

class VoEstimator {
public:
    VoEstimator(
        const EstimatorOption& option, std::shared_ptr<State> state,
        std::shared_ptr<FeatureManager> fm,
        const Eigen::Matrix3d& ric = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& tic = Eigen::Vector3d::Zero(),
        const Eigen::Matrix3d& ric1 = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& tic1 = Eigen::Vector3d::Zero());

    void processMeasurement(
        double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame);

    std::shared_ptr<State> getState() const { return state_; }
    std::shared_ptr<FeatureManager> getFeatureManager() const { return feature_manager_; }

private:
    void optimize();
    void marginalizeOldestFrame();
    void slideWindow();

    void initializeMargPrior();

    EstimatorOption option_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> feature_manager_;
    std::shared_ptr<MargLinData> cur_marg_lin_data_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Matrix3d ric1_;
    Eigen::Vector3d tic1_;

    double init_ts_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
