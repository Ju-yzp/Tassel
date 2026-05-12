#ifndef TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_
#define TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_

#include "marg_linearized_data.h"
#include "state/state.h"

namespace tassel_core {
class MargHelper {
public:
    void computeDelta(const State& state, Eigen::VectorXd& delta) const;

    void linearizeMargPrior(
        const MargLinData& mld, const State& cur_state, Eigen::MatrixXd& abs_H,
        Eigen::VectorXd& abs_b, double& marg_prior_error);

    void computeMargPriorError(const MargLinData& mld, double& marg_prior_error) const;

    // 使用sqrt策略,默认边缘化第一帧也就是最老帧
    void marginalizeOldest(
        size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, Eigen::MatrixXd& marg_sqrt_H,
        Eigen::VectorXd& marg_sqrt_b);
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_ */
