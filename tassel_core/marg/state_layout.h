#ifndef TASSEL_CORE_MARG_STATE_LAYOUT_H_
#define TASSEL_CORE_MARG_STATE_LAYOUT_H_

#include <array>
#include <vector>

#include "marg/window_action.h"

namespace tassel_core {

enum class PriorLayoutKind {
    FullState,
    PoseOnlyHost,
    PoseOnly,
};

// 描述 MargLinData 紧凑列到窗口统一 15 维状态列的映射；frame_index 均指先验内索引。
class PriorStateLayout {
public:
    static constexpr int kPoseSize = 6;
    static constexpr int kSpeedBiasSize = 9;
    static constexpr int kFullStateSize = 15;

    PriorStateLayout(int state_count, int column_count);

    int stateCount() const { return state_count_; }
    int columnCount() const { return column_count_; }
    PriorLayoutKind kind() const { return kind_; }
    bool hasDelay() const { return has_delay_; }
    bool hasSpeedBias(int frame_index) const;

    int poseColumn(int frame_index) const;
    int speedBiasColumn(int frame_index) const;
    std::array<int, 3> baColumns(int frame_index) const;
    int delayColumn() const;

    // 返回 compact_column -> full_window_column；窗口列布局为 [frame(15)..., delay]。
    std::vector<int> compactToWindowColumns(int window_state_count) const;

private:
    void requireFrameIndex(int frame_index) const;

    int state_count_;
    int column_count_;
    PriorLayoutKind kind_;
    bool has_delay_;
};

std::vector<int> marginalizationColumnOrder(int window_state_count, RetainedHostAction action);

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_STATE_LAYOUT_H_
