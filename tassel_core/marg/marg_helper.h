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
    // 将固定先验传输到当前状态的右扰动切空间，不重新线性化原始因子。
    static MargLinData transportPriorToCurrentTangent(
        const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
        const std::vector<std::array<double, 9>>& speed_bias, double delay_time);

    // 使用秩揭示 Householder QR 消去平方根系统前部的参数列。
    // 列顺序为 [待消元参数 | 保留参数]。
    static void eliminateSquareRootSystem(
        size_t eliminated_size, size_t retained_size, Eigen::MatrixXd& jacobian,
        Eigen::VectorXd& residual, Eigen::MatrixXd& prior_jacobian,
        Eigen::VectorXd& prior_residual);
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARG_HELPER_H_
