#ifndef TASSEL_CORE_ESTIMATOR_WINDOW_MARGINALIZATION_PLAN_H_
#define TASSEL_CORE_ESTIMATOR_WINDOW_MARGINALIZATION_PLAN_H_

#include <Eigen/Core>

#include "tassel_utils/macros.h"

namespace tassel_core {

enum class HostTransition {
    kPromote,
    kKeep,
    kReplace,
};

// Describes one full-window retirement. Slot 0 is either the first full state
// (promotion) or the retained host pose with inactive speed/bias (keep/replace).
struct WindowMarginalizationPlan {
    static constexpr int kPoseSize = 6;
    static constexpr int kSpeedBiasSize = 9;
    static constexpr int kFullStateSize = 15;

    static WindowMarginalizationPlan create(bool has_retained_host, bool replace_host) {
        TASSEL_ASSERT(!replace_host || has_retained_host);
        if (!has_retained_host) return {HostTransition::kPromote, 0, 2, 0, 1};
        if (replace_host) return {HostTransition::kReplace, 1, 1, 1, 0};
        return {HostTransition::kKeep, 1, 1, 0, 1};
    }

    bool replacesHost() const { return transition == HostTransition::kReplace; }

    Eigen::MatrixXd marginalColumnsFirst(const Eigen::MatrixXd& jacobian) const {
        TASSEL_ASSERT(jacobian.cols() >= 2 * kFullStateSize + 1);
        Eigen::MatrixXd ordered(jacobian.rows(), jacobian.cols());
        int column = 0;
        if (replacesHost()) {
            ordered.middleCols(column, kFullStateSize) = jacobian.leftCols(kFullStateSize);
            column += kFullStateSize;
            ordered.middleCols(column, kSpeedBiasSize) =
                jacobian.middleCols(kFullStateSize + kPoseSize, kSpeedBiasSize);
            column += kSpeedBiasSize;
            ordered.middleCols(column, kPoseSize) = jacobian.middleCols(kFullStateSize, kPoseSize);
        } else {
            ordered.middleCols(column, kSpeedBiasSize) =
                jacobian.middleCols(kPoseSize, kSpeedBiasSize);
            column += kSpeedBiasSize;
            ordered.middleCols(column, kFullStateSize) =
                jacobian.middleCols(kFullStateSize, kFullStateSize);
            column += kFullStateSize;
            ordered.middleCols(column, kPoseSize) = jacobian.leftCols(kPoseSize);
        }
        column += kPoseSize;

        const int trailing_state_columns =
            static_cast<int>(jacobian.cols()) - 2 * kFullStateSize - 1;
        ordered.middleCols(column, trailing_state_columns) =
            jacobian.middleCols(2 * kFullStateSize, trailing_state_columns);
        column += trailing_state_columns;
        ordered.col(column) = jacobian.col(jacobian.cols() - 1);
        return ordered;
    }

    HostTransition transition;
    int imu_start_slot;
    int imu_block_count;
    int retained_pose_source_slot;
    int removed_measurement_slot;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_ESTIMATOR_WINDOW_MARGINALIZATION_PLAN_H_
