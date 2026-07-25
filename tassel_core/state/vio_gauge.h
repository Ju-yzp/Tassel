#ifndef TASSEL_CORE_STATE_VIO_GAUGE_H_
#define TASSEL_CORE_STATE_VIO_GAUGE_H_

#include <Eigen/Core>

#include "state.h"

namespace tassel_core {

// 恢复优化前的全局位置和航向规范，不约束可观的横滚角与俯仰角。
void restoreVioGauge(
    State& state, int first_frame_index, const Eigen::Matrix3d& reference_rotation,
    const Eigen::Vector3d& reference_position);

}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_VIO_GAUGE_H_
