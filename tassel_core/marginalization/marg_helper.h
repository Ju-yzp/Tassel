#ifndef TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_
#define TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_

#include "marg_linearized_data.h"
#include "state/state.h"

namespace tassel_core {
class MargHelper {
public:
    static void computeDelta(const State& state, Eigen::VectorXd& delta);

    static void linearizeMargPrior(
        const MargLinData& mld, const State& cur_state, Eigen::MatrixXd& abs_H,
        Eigen::VectorXd& abs_b, double& marg_prior_error);

    static void computeMargPriorError(
        const MargLinData& mld, const State& cur_state, double& marg_prior_error);

    static void marginalizeOldest(
        size_t marg_size, size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r,
        Eigen::MatrixXd& marg_sqrt_H, Eigen::VectorXd& marg_sqrt_b);

    static double computeMargPriorModelCostChange(
        const MargLinData& mld, const State& cur_state, const Eigen::VectorXd& marg_scaling,
        const Eigen::VectorXd& marg_pose_inc);
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_ */
