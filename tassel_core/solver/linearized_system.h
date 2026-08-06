#ifndef TASSEL_CORE_SOLVER_LINEARIZED_SYSTEM_H_
#define TASSEL_CORE_SOLVER_LINEARIZED_SYSTEM_H_

#include <Eigen/Core>

namespace tassel_core {

class LinearizedSystem {
public:
    LinearizedSystem(Eigen::MatrixXd jacobian, Eigen::VectorXd residual);

    const Eigen::MatrixXd& jacobian() const { return jacobian_; }
    const Eigen::VectorXd& residual() const { return residual_; }
    int variableSize() const { return static_cast<int>(jacobian_.cols()); }

    double cost() const;
    double costAt(const Eigen::VectorXd& delta) const;
    Eigen::MatrixXd hessian() const;
    Eigen::VectorXd gradient() const;

private:
    // 统一约定为 r(delta) = residual + jacobian * delta。
    Eigen::MatrixXd jacobian_;
    Eigen::VectorXd residual_;
};

struct DiagonalDamping {
    double lambda = 0.0;
    double min_diagonal = 0.0;
};

struct LinearStep {
    Eigen::VectorXd delta;
    Eigen::VectorXd damping_diagonal;
    double initial_cost = 0.0;
    double model_cost = 0.0;
    double predicted_reduction = 0.0;
};

// 求解 (J^T J + D) delta = -J^T r，其中 D_i=max(lambda*H_ii,min_diagonal)。
// 未提供足够阻尼的缺秩系统属于求解错误，不使用伪逆静默补充不可观信息。
LinearStep solveDampedNormalStep(
    const LinearizedSystem& system, const DiagonalDamping& damping = {});

}  // namespace tassel_core

#endif  // TASSEL_CORE_SOLVER_LINEARIZED_SYSTEM_H_
