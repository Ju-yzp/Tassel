#ifndef TASSEL_CORE_ESTIMATOR_IMU_INTERPOLATION_H_
#define TASSEL_CORE_ESTIMATOR_IMU_INTERPOLATION_H_

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_core {

inline void interpolateBodyImu(
    const std::vector<tassel_utils::IMUMeasurement>& measurements, double timestamp,
    Eigen::Vector3d& body_gyro, Eigen::Vector3d& body_acc) {
    body_gyro.setZero();
    body_acc.setZero();
    if (measurements.empty()) {
        return;
    }

    const size_t count = measurements.size();
    if (count < 3) {
        // 样本不足以进行二次插值时，退化为区间内最新的体坐标系测量。
        body_gyro = measurements.back().gyro;
        body_acc = measurements.back().acc;
        return;
    }

    size_t right = 0;
    while (right < count && measurements[right].timestamp < timestamp) {
        ++right;
    }
    size_t first = right > 1 ? right - 2 : 0;
    if (first + 3 > count) {
        first = count - 3;
    }

    // 使用查询时刻附近三个样本的二次拉格朗日插值，陀螺仪和加速度计共用权重。
    for (size_t i = 0; i < 3; ++i) {
        const double sample_time = measurements[first + i].timestamp;
        double weight = 1.0;
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                continue;
            }
            const double other_time = measurements[first + j].timestamp;
            const double denominator = sample_time - other_time;
            if (std::abs(denominator) < 1e-12) {
                // 重复采样时刻无法构造插值基函数，保持与样本不足时一致的退化行为。
                body_gyro = measurements.back().gyro;
                body_acc = measurements.back().acc;
                return;
            }
            weight *= (timestamp - other_time) / denominator;
        }
        body_gyro += weight * measurements[first + i].gyro;
        body_acc += weight * measurements[first + i].acc;
    }
}

}  // namespace tassel_core

#endif  // TASSEL_CORE_ESTIMATOR_IMU_INTERPOLATION_H_
