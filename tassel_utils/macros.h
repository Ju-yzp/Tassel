#ifndef TASSEL_UTILS_MACROS_H_
#define TASSEL_UTILS_MACROS_H_

#include <spdlog/spdlog.h>
#include <source_location>

namespace tassel_utils {

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
