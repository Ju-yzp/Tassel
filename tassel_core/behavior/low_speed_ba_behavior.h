#ifndef TASSEL_CORE_BEHAVIOR_LOW_SPEED_BA_BEHAVIOR_H_
#define TASSEL_CORE_BEHAVIOR_LOW_SPEED_BA_BEHAVIOR_H_

#include <Eigen/Core>

#include <vector>

namespace tassel_core {

enum class BaUpdateMode {
    Normal,
    LowSpeedHold,
};

class LowSpeedBaBehavior {
public:
    LowSpeedBaBehavior(double enter_speed, double exit_speed, int enter_windows, int exit_windows);

    BaUpdateMode update(const std::vector<Eigen::Vector3d>& velocities);
    bool holdsBa() const { return mode_ == BaUpdateMode::LowSpeedHold; }
    bool needsSchmidtCovariance() const { return entry_candidate_ || holdsBa(); }
    void reset();

private:
    double enter_speed_;
    double exit_speed_;
    int enter_windows_;
    int exit_windows_;
    int enter_count_ = 0;
    int exit_count_ = 0;
    bool entry_candidate_ = false;
    BaUpdateMode mode_ = BaUpdateMode::Normal;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_BEHAVIOR_LOW_SPEED_BA_BEHAVIOR_H_
