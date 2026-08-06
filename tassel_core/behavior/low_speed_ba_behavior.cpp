#include "behavior/low_speed_ba_behavior.h"

#include <cmath>
#include <stdexcept>

namespace tassel_core {

LowSpeedBaBehavior::LowSpeedBaBehavior(
    double enter_speed, double exit_speed, int enter_windows, int exit_windows)
    : enter_speed_(enter_speed),
      exit_speed_(exit_speed),
      enter_windows_(enter_windows),
      exit_windows_(exit_windows) {
    if (!std::isfinite(enter_speed_) || !std::isfinite(exit_speed_) || enter_speed_ < 0.0 ||
        exit_speed_ <= enter_speed_ || enter_windows_ <= 0 || exit_windows_ <= 0) {
        throw std::invalid_argument("Invalid low-speed Ba state thresholds");
    }
}

BaUpdateMode LowSpeedBaBehavior::update(const std::vector<Eigen::Vector3d>& velocities) {
    if (velocities.empty()) {
        throw std::invalid_argument("Low-speed Ba state requires a non-empty velocity window");
    }
    bool all_below_enter = true;
    bool any_above_exit = false;
    for (const Eigen::Vector3d& velocity : velocities) {
        if (!velocity.allFinite()) {
            throw std::invalid_argument("Low-speed Ba state received a non-finite velocity");
        }
        all_below_enter = all_below_enter && velocity.norm() <= enter_speed_;
        any_above_exit = any_above_exit || velocity.norm() >= exit_speed_;
    }
    entry_candidate_ = all_below_enter;

    if (mode_ == BaUpdateMode::Normal) {
        enter_count_ = all_below_enter ? enter_count_ + 1 : 0;
        if (enter_count_ >= enter_windows_) {
            mode_ = BaUpdateMode::LowSpeedHold;
            enter_count_ = 0;
        }
        return mode_;
    }

    exit_count_ = any_above_exit ? exit_count_ + 1 : 0;
    if (exit_count_ >= exit_windows_) {
        mode_ = BaUpdateMode::Normal;
        exit_count_ = 0;
    }
    return mode_;
}

void LowSpeedBaBehavior::reset() {
    enter_count_ = 0;
    exit_count_ = 0;
    entry_candidate_ = false;
    mode_ = BaUpdateMode::Normal;
}

}  // namespace tassel_core
