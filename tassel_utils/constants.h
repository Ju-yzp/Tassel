#ifndef TASSEL_UTILS_CONSTANTS_H_
#define TASSEL_UTILS_CONSTANTS_H_

#include <Eigen/Dense>

namespace tassel_utils {

constexpr int POSE_SIZE = 6;
constexpr int SPEED_SIZE = 3;
constexpr int BIAS_GYRO_SIZE = 3;
constexpr int BIAS_ACC_SIZE = 3;
constexpr int TOTAL_SIZE = POSE_SIZE + SPEED_SIZE + BIAS_ACC_SIZE + BIAS_GYRO_SIZE;

constexpr int POSE_IDX = 0;
constexpr int SPEED_IDX = POSE_SIZE;
constexpr int BIAS_ACC_IDX = POSE_SIZE + SPEED_SIZE;
constexpr int BIAS_GYRO_IDX = POSE_SIZE + SPEED_SIZE + BIAS_ACC_SIZE;

const int O_P = 0;
const int O_R = 3;
const int O_V = 6;
const int O_BA = 9;
const int O_BG = 12;

}  // namespace tassel_utils

#endif  // TASSEL_UTILS_CONSTANTS_H_
