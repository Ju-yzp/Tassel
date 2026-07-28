#ifndef TASSEL_CORE_MARG_MARG_HELPER_H_
#define TASSEL_CORE_MARG_MARG_HELPER_H_

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <vector>

#include "marg_lin_data.h"

namespace tassel_core {

enum class RetainedHostAction {
    Create,
    Keep,
    Replace,
};

class MargHelper {
public:
    static constexpr int kPoseSize = 6;
    static constexpr int kSpeedBiasSize = 9;
    static constexpr int kFullStateSize = 15;

    // 在当前状态计算固定线性化先验的残差；先验雅各比保持在原线性化点。
    static Eigen::VectorXd evaluatePriorResidual(
        const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
        const std::vector<std::array<double, 9>>& speed_bias, double delay_time);

    // 将先验的一阶模型迁移到当前状态，并同步更新局部雅各比。
    static void recenterPrior(
        MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
        const std::vector<std::array<double, 9>>& speed_bias, double delay_time);

    // 对先验的世界系位置、姿态和速度施加统一的 gauge 变换。
    static void transformPriorGauge(
        MargLinData& prior, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);

    // 使用秩揭示 Householder QR 边缘化平方根系统前部的参数列。
    // 列顺序为 [待边缘化参数 | 保留参数]。
    static void marginalizeSquareRootSystem(
        size_t marginalized_size, size_t retained_size, Eigen::MatrixXd& jacobian,
        Eigen::VectorXd& residual, Eigen::MatrixXd& prior_jacobian,
        Eigen::VectorXd& prior_residual);

    static Eigen::MatrixXd reorderForMarginalization(
        const Eigen::MatrixXd& jacobian, RetainedHostAction action);
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARG_HELPER_H_
