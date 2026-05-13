#ifndef TASSEL_CORE_MARGINALIZATION_MARG_LINEARIZED_DATA_H_
#define TASSEL_CORE_MARGINALIZATION_MARG_LINEARIZED_DATA_H_

#include <Eigen/Dense>

#include "state/state.h"

namespace tassel_core {
struct MargLinData {
    MargLinData() {}
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H;
    Eigen::Matrix<double, Eigen::Dynamic, 1> b;
};
}  // namespace tassel_core

#endif /* TASSEL_CORE_MARGINALIZATION_MARG_DATA_H_ */
