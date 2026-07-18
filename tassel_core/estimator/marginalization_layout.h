#ifndef TASSEL_CORE_ESTIMATOR_MARGINALIZATION_LAYOUT_H_
#define TASSEL_CORE_ESTIMATOR_MARGINALIZATION_LAYOUT_H_

#include <Eigen/Core>

#include "tassel_utils/macros.h"

namespace tassel_core {

enum class RetainedHostAction {
    kCreate,
    kKeep,
    kReplace,
};

// 定义单次滑窗边缘化的参数消元顺序和因子范围。
// 槽位 0 为保留宿主帧，槽位 1 为最旧的活跃状态。
class MarginalizationLayout {
public:
    static constexpr int kPoseSize = 6;
    static constexpr int kSpeedBiasSize = 9;
    static constexpr int kFullStateSize = 15;

    explicit MarginalizationLayout(RetainedHostAction host_action) : host_action_(host_action) {}

    RetainedHostAction hostAction() const { return host_action_; }
    bool replacesRetainedHost() const { return host_action_ == RetainedHostAction::kReplace; }
    bool eliminatesSlot0Motion() const { return host_action_ == RetainedHostAction::kCreate; }

    int firstImuFactorSlot() const { return host_action_ == RetainedHostAction::kCreate ? 0 : 1; }

    int imuFactorCount() const { return host_action_ == RetainedHostAction::kCreate ? 2 : 1; }

    int nextRetainedHostSourceSlot() const {
        return host_action_ == RetainedHostAction::kReplace ? 1 : 0;
    }

    int eliminatedParameterSize() const {
        if (replacesRetainedHost() || eliminatesSlot0Motion()) {
            return kFullStateSize + kSpeedBiasSize;
        }
        return kFullStateSize;
    }

    Eigen::MatrixXd reorderForElimination(const Eigen::MatrixXd& jacobian) const {
        TASSEL_ASSERT(jacobian.cols() >= 2 * kFullStateSize + 1);
        Eigen::MatrixXd reordered(jacobian.rows(), jacobian.cols());
        int destination_column = 0;
        if (replacesRetainedHost()) {
            reordered.middleCols(destination_column, kFullStateSize) =
                jacobian.leftCols(kFullStateSize);
            destination_column += kFullStateSize;
            reordered.middleCols(destination_column, kSpeedBiasSize) =
                jacobian.middleCols(kFullStateSize + kPoseSize, kSpeedBiasSize);
            destination_column += kSpeedBiasSize;
            reordered.middleCols(destination_column, kPoseSize) =
                jacobian.middleCols(kFullStateSize, kPoseSize);
        } else if (eliminatesSlot0Motion()) {
            reordered.middleCols(destination_column, kSpeedBiasSize) =
                jacobian.middleCols(kPoseSize, kSpeedBiasSize);
            destination_column += kSpeedBiasSize;
            reordered.middleCols(destination_column, kFullStateSize) =
                jacobian.middleCols(kFullStateSize, kFullStateSize);
            destination_column += kFullStateSize;
            reordered.middleCols(destination_column, kPoseSize) = jacobian.leftCols(kPoseSize);
        } else {
            reordered.middleCols(destination_column, kFullStateSize) =
                jacobian.middleCols(kFullStateSize, kFullStateSize);
            destination_column += kFullStateSize;
            reordered.middleCols(destination_column, kFullStateSize) =
                jacobian.leftCols(kFullStateSize);
        }
        destination_column +=
            replacesRetainedHost() || eliminatesSlot0Motion() ? kPoseSize : kFullStateSize;

        const int trailing_state_columns =
            static_cast<int>(jacobian.cols()) - 2 * kFullStateSize - 1;
        reordered.middleCols(destination_column, trailing_state_columns) =
            jacobian.middleCols(2 * kFullStateSize, trailing_state_columns);
        destination_column += trailing_state_columns;
        reordered.col(destination_column) = jacobian.col(jacobian.cols() - 1);
        return reordered;
    }

private:
    RetainedHostAction host_action_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_ESTIMATOR_MARGINALIZATION_LAYOUT_H_
