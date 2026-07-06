#ifndef TASSEL_CORE_MARG_MARG_HELPER_H_
#define TASSEL_CORE_MARG_MARG_HELPER_H_

#include <Eigen/Core>
#include <cstddef>

namespace tassel_core {

class MargHelper {
public:
    // Householder QR 压缩 sqrt 形式 [Q2Jp | Q2r] → 新 sqrt 先验
    // 列布局: [marg (丢弃) | keep (保留)]
    static void marginalizeSqrtToSqrt(
        size_t marg_size, size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r,
        Eigen::MatrixXd& marg_sqrt_H, Eigen::VectorXd& marg_sqrt_b);
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARG_HELPER_H_
