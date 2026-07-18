#ifndef TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_
#define TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_

#include <Eigen/Core>

#include <array>
#include <vector>

namespace tassel_core {

struct MargLinData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    std::vector<std::array<double, 6>> linearization_poses;
    std::vector<std::array<double, 9>> linearization_speed_bias;
    double linearization_delay_time = 0.0;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_
