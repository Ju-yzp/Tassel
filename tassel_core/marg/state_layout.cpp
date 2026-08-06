#include "marg/state_layout.h"

#include <stdexcept>

namespace tassel_core {

PriorStateLayout::PriorStateLayout(int state_count, int column_count)
    : state_count_(state_count), column_count_(column_count) {
    if (state_count_ <= 0 || column_count_ < 0) {
        throw std::invalid_argument("Prior state layout dimensions are invalid");
    }
    const int full_columns = state_count_ * kFullStateSize;
    const int mixed_columns = kPoseSize + (state_count_ - 1) * kFullStateSize;
    const int pose_columns = state_count_ * kPoseSize;
    if (column_count_ == full_columns || column_count_ == full_columns + 1) {
        kind_ = PriorLayoutKind::FullState;
        has_delay_ = column_count_ == full_columns + 1;
    } else if (column_count_ == mixed_columns || column_count_ == mixed_columns + 1) {
        kind_ = PriorLayoutKind::PoseOnlyHost;
        has_delay_ = column_count_ == mixed_columns + 1;
    } else if (column_count_ == pose_columns || column_count_ == pose_columns + 1) {
        kind_ = PriorLayoutKind::PoseOnly;
        has_delay_ = column_count_ == pose_columns + 1;
    } else {
        throw std::invalid_argument("Marginalization prior has an unsupported column layout");
    }
}

bool PriorStateLayout::hasSpeedBias(int frame_index) const {
    requireFrameIndex(frame_index);
    return kind_ == PriorLayoutKind::FullState ||
           (kind_ == PriorLayoutKind::PoseOnlyHost && frame_index > 0);
}

int PriorStateLayout::poseColumn(int frame_index) const {
    requireFrameIndex(frame_index);
    switch (kind_) {
        case PriorLayoutKind::FullState:
            return frame_index * kFullStateSize;
        case PriorLayoutKind::PoseOnlyHost:
            return frame_index == 0 ? 0 : kPoseSize + (frame_index - 1) * kFullStateSize;
        case PriorLayoutKind::PoseOnly:
            return frame_index * kPoseSize;
    }
    throw std::logic_error("Prior layout kind is invalid");
}

int PriorStateLayout::speedBiasColumn(int frame_index) const {
    if (!hasSpeedBias(frame_index)) {
        throw std::logic_error("Prior frame does not contain speed and bias variables");
    }
    return poseColumn(frame_index) + kPoseSize;
}

std::array<int, 3> PriorStateLayout::baColumns(int frame_index) const {
    const int first_ba_column = speedBiasColumn(frame_index) + 3;
    return {first_ba_column, first_ba_column + 1, first_ba_column + 2};
}

int PriorStateLayout::delayColumn() const {
    if (!has_delay_) {
        throw std::logic_error("Prior layout does not contain delay");
    }
    return column_count_ - 1;
}

std::vector<int> PriorStateLayout::compactToWindowColumns(int window_state_count) const {
    if (window_state_count < state_count_) {
        throw std::invalid_argument("Window does not contain every prior state");
    }
    std::vector<int> mapping(static_cast<size_t>(column_count_), -1);
    for (int frame_index = 0; frame_index < state_count_; ++frame_index) {
        const int compact_pose = poseColumn(frame_index);
        const int window_pose = frame_index * kFullStateSize;
        for (int d = 0; d < kPoseSize; ++d) {
            mapping[static_cast<size_t>(compact_pose + d)] = window_pose + d;
        }
        if (hasSpeedBias(frame_index)) {
            const int compact_speed_bias = speedBiasColumn(frame_index);
            for (int d = 0; d < kSpeedBiasSize; ++d) {
                mapping[static_cast<size_t>(compact_speed_bias + d)] = window_pose + kPoseSize + d;
            }
        }
    }
    if (has_delay_) {
        mapping[static_cast<size_t>(delayColumn())] = window_state_count * kFullStateSize;
    }
    return mapping;
}

void PriorStateLayout::requireFrameIndex(int frame_index) const {
    if (frame_index < 0 || frame_index >= state_count_) {
        throw std::out_of_range("Prior frame index is outside the layout");
    }
}

std::vector<int> marginalizationColumnOrder(int window_state_count, RetainedHostAction action) {
    if (window_state_count < 2) {
        throw std::invalid_argument("Marginalization layout requires at least two states");
    }
    constexpr int pose_size = PriorStateLayout::kPoseSize;
    constexpr int speed_bias_size = PriorStateLayout::kSpeedBiasSize;
    constexpr int state_size = PriorStateLayout::kFullStateSize;
    const int total_columns = window_state_count * state_size + 1;
    std::vector<int> order;
    order.reserve(static_cast<size_t>(total_columns));
    const auto append = [&order](int first, int count) {
        for (int i = 0; i < count; ++i) {
            order.push_back(first + i);
        }
    };

    switch (action) {
        case RetainedHostAction::InitializeRetainedSlot:
            append(0, state_size);
            append(state_size + pose_size, speed_bias_size);
            append(state_size, pose_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
        case RetainedHostAction::MarginalizeOldestFrame:
            append(state_size, state_size);
            append(0, state_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
        case RetainedHostAction::ReplaceRetainedSlot:
            append(0, state_size);
            append(state_size + pose_size, speed_bias_size);
            append(state_size, pose_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
    }
    if (static_cast<int>(order.size()) != total_columns) {
        throw std::logic_error("Marginalization column mapping is incomplete");
    }
    return order;
}

}  // namespace tassel_core
