#ifndef TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_
#define TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_

#include <unordered_map>
#include "abs_order_map.h"
#include "marg_linearized_data.h"
#include "pose_state_with_lin.h"

namespace tassel_core {
class MargHelper {
public:
    void computeDelta(const AbsOrderMap& aom, Eigen::VectorXd& delta) const;

    void linearizeMargPrior(
        const MargLinData& mld, const AbsOrderMap& aom, Eigen::MatrixXd& abs_H,
        Eigen::VectorXd& abs_b, double& marg_prior_error);

    void computeMargPriorError(const MargLinData& mld, double& marg_prior_error) const;

    // 使用sqrt策略,默认边缘化第一帧也就是最老帧
    void marginalizeOldest(
        size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, Eigen::MatrixXd& marg_sqrt_H,
        Eigen::VectorXd& marg_sqrt_b);

private:
    std::unordered_map<int64_t, PoseStateWithLin> frame_poses_;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_MARGINALIZATION_MARG_HELPER_H_ */
