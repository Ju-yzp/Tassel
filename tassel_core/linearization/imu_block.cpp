#include "imu_block.h"

namespace tassel_core {
void IMUBlock::add_dense_H_b(
    size_t start_idx, size_t end_idx, Eigen::MatrixXd& jacobian, Eigen::VectorXd& residual) const {}
}  // namespace tassel_core
