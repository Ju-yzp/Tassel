#ifndef TASSEL_CORE_MARGINALIZATION_ABS_ORDER_MAP_H_
#define TASSEL_CORE_MARGINALIZATION_ABS_ORDER_MAP_H_

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace tassel_core {
struct AbsOrderMap {
    std::unordered_map<int64_t, std::pair<int, int>> abs_order_map;  // frame_id → (start_idx, size)
    std::size_t items = 0;                                           // 总共多少帧
    std::size_t total_size = 0;  // 所有帧的块大小之和 = 线性系统变量总数
};

}  // namespace tassel_core
#endif /* TASSEL_CORE_MARGINALIZATION_ABS_ORDER_MAP_H_ */
