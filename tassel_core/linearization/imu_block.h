#ifndef TASSEL_CORE_IMU_BLOCK_H_
#define TASSEL_CORE_IMU_BLOCK_H_

#include <Eigen/Dense>
#include <cstddef>

#include "preintegrator.h"
#include "state/state.h"

namespace tassel_core {

class IMUBlock {
public:
    explicit IMUBlock();

    void add_dense_H_b(
        size_t start_idx, size_t end_idx, Eigen::MatrixXd& jacobian,
        Eigen::VectorXd& residual) const;

    void restore();

    void update(const Eigen::VectorXd inc);

private:
    PreIntegrator* preintegrator_;
    State* state_;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_IMU_BLOCK_H_ */
