#ifndef TASSEL_LOOP_LOOP_TYPES_H_
#define TASSEL_LOOP_LOOP_TYPES_H_

#include <cstdint>
#include <limits>

namespace tassel_loop {

using KeyframeId = std::uint64_t;
inline constexpr KeyframeId kInvalidKeyframeId = std::numeric_limits<KeyframeId>::max();

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_LOOP_TYPES_H_
