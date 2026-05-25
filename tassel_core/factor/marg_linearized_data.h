#ifndef TASSEL_CORE_FACTOR_MARG_LINEARIZED_DATA_H_
#define TASSEL_CORE_FACTOR_MARG_LINEARIZED_DATA_H_

#include <Eigen/Dense>

namespace tassel_core {

struct MargLinData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARG_LINEARIZED_DATA_H_
