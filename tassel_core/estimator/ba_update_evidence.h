#ifndef TASSEL_CORE_ESTIMATOR_BA_UPDATE_EVIDENCE_H_
#define TASSEL_CORE_ESTIMATOR_BA_UPDATE_EVIDENCE_H_

#include <Eigen/Core>

#include <array>
#include <limits>

namespace tassel_core {

struct BaUpdateEvidence {
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    Eigen::Vector3d increment = Eigen::Vector3d::Zero();
    double condition = std::numeric_limits<double>::infinity();
    double cost_reduction = 0.0;
    int rank = 0;
};

// 在残差空间消去其余状态，返回新数据对 Ba 的条件增量证据。
BaUpdateEvidence computeBaUpdateEvidence(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& residual,
    const std::array<int, 3>& ba_columns);

// candidate_basis 将统一 Ba 增量映射到系统切空间；nuisance_basis 描述允许解释残差的方向。
BaUpdateEvidence computeBaUpdateEvidence(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& candidate_basis, const Eigen::MatrixXd& nuisance_basis);

}  // namespace tassel_core

#endif  // TASSEL_CORE_ESTIMATOR_BA_UPDATE_EVIDENCE_H_
