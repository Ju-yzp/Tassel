#ifndef TASSEL_UTILS_MACROS_H_
#define TASSEL_UTILS_MACROS_H_

#include <source_location>

#include <spdlog/spdlog.h>

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

constexpr double ACC_N = 1.9291286492253038e-02;
constexpr double ACC_W = 8.0363303593887979e-04;
constexpr double GYR_N = 2.6401761065710893e-03;
constexpr double GYR_W = 7.0304129470456934e-05;

inline void tassel_assert_failed(
    const char* condition_str, const std::source_location& loc = std::source_location::current()) {
    spdlog::critical(fmt::format(
        "TASSEL_ASSERT failed: ({}) in function: {} ({}:{})", condition_str, loc.function_name(),
        loc.file_name(), loc.line()));
    std::abort();
}

}  // namespace tassel_utils

#define TASSEL_ASSERT(cond)                            \
    do {                                               \
        if (!(cond)) {                                 \
            tassel_utils::tassel_assert_failed(#cond); \
        }                                              \
    } while (0)

#endif  // TASSEL_UTILS_MACROS_H_
