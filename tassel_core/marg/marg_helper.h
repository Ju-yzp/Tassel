#ifndef TASSEL_CORE_MARG_MARG_HELPER_H_
#define TASSEL_CORE_MARG_MARG_HELPER_H_

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <vector>

#include "marg_lin_data.h"

namespace tassel_core {

class MargHelper {
public:
    // 在当前状态的右扰动切空间中重新线性化已有平方根先验。
    static MargLinData rebasePrior(
        const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
        const std::vector<std::array<double, 9>>& speed_bias, double delay_time);

    // Householder QR 压缩 sqrt 形式 [Q2Jp | Q2r] → 新 sqrt 先验
    // 列布局: [marg (丢弃) | keep (保留)]
    static void marginalizeSqrtToSqrt(
        size_t marg_size, size_t keep_size, Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r,
        Eigen::MatrixXd& marg_sqrt_H, Eigen::VectorXd& marg_sqrt_b);
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARG_HELPER_H_
