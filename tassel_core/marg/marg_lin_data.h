#ifndef TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_
#define TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_

#include <Eigen/Core>

#include <array>
#include <stdexcept>
#include <vector>

namespace tassel_core {

// 先验列紧凑排列为 [retained pose(6), frame1 pose+speed_bias(15), ..., delay(1)]；
// frame 0 是保留帧，不包含 speed-bias 先验列。
struct MargLinData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int PoseSize = 6;
    static constexpr int SpeedBiasSize = 9;
    static constexpr int StateSize = PoseSize + SpeedBiasSize;

    int stateCount() const { return static_cast<int>(linearization_poses.size()); }

    int poseColumn(int frame_index) const {
        requireFrame(frame_index);
        return frame_index == 0 ? 0 : PoseSize + (frame_index - 1) * StateSize;
    }

    int speedBiasColumn(int frame_index) const {
        requireFrame(frame_index);
        if (frame_index == 0) {
            throw std::logic_error("Retained prior frame has no speed-bias columns");
        }
        return poseColumn(frame_index) + PoseSize;
    }

    int delayColumn() const { return PoseSize + (stateCount() - 1) * StateSize; }

    int columnCount() const { return delayColumn() + 1; }

    void validate() const {
        if (stateCount() <= 0 ||
            static_cast<int>(linearization_speed_bias.size()) != stateCount() ||
            H.cols() != columnCount() || H.rows() != b.size()) {
            throw std::invalid_argument("Marginalization prior does not match the window state");
        }
    }

    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    std::vector<std::array<double, 6>> linearization_poses;
    std::vector<std::array<double, 9>> linearization_speed_bias;
    double linearization_delay_time = 0.0;

private:
    void requireFrame(int frame_index) const {
        if (frame_index < 0 || frame_index >= stateCount()) {
            throw std::out_of_range("Prior frame index is outside the active state");
        }
    }
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARG_LIN_DATA_H_
