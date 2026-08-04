#include "nori/nori_sync.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tassel_hardware {

NoriSync::NoriSync(double camera_to_imu_delay) : camera_to_imu_delay_(camera_to_imu_delay) {
    if (!std::isfinite(camera_to_imu_delay_)) {
        throw std::invalid_argument("Nori camera-to-IMU delay must be finite");
    }
}

void NoriSync::push(NoriImageFrame mono, const std::vector<tassel_utils::IMUMeasurement>& imu) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
        return;
    }
    for (const auto& sample : imu) {
        if (imus_.empty() || sample.timestamp > imus_.back().timestamp) {
            imus_.push_back(sample);
        }
    }
    frames_.push_back(std::move(mono));
    cv_.notify_all();
}

bool NoriSync::waitPop(NoriSyncedFrame& output) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() {
        if (frames_.empty()) {
            return stopped_;
        }
        const double end =
            tassel_utils::frameIdToSeconds(frames_.front().frame_id) + camera_to_imu_delay_;
        return stopped_ || (!imus_.empty() && imus_.back().timestamp >= end);
    });
    if (frames_.empty()) {
        return false;
    }
    const double end =
        tassel_utils::frameIdToSeconds(frames_.front().frame_id) + camera_to_imu_delay_;
    if (imus_.empty() || imus_.back().timestamp < end) {
        return false;
    }

    output.mono = std::move(frames_.front());
    frames_.pop_front();
    output.imu.clear();
    if (has_boundary_) {
        output.imu.push_back(boundary_);
    }
    for (const auto& sample : imus_) {
        if (sample.timestamp >= end) {
            break;
        }
        if (!has_boundary_ || sample.timestamp > boundary_.timestamp) {
            output.imu.push_back(sample);
        }
    }
    boundary_ = interpolate(end);
    if (output.imu.empty() || output.imu.back().timestamp < boundary_.timestamp) {
        output.imu.push_back(boundary_);
    }
    // 相邻图像包共享边界插值样本，使两个预积分区间连续且不重复积分。
    while (!imus_.empty() && imus_.front().timestamp <= end) {
        imus_.pop_front();
    }
    has_boundary_ = true;
    lock.unlock();
    cv_.notify_all();
    return true;
}

void NoriSync::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
}

size_t NoriSync::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_.size();
}

tassel_utils::IMUMeasurement NoriSync::interpolate(double timestamp) const {
    const auto upper = std::lower_bound(
        imus_.begin(), imus_.end(), timestamp,
        [](const auto& sample, double value) { return sample.timestamp < value; });
    if (upper != imus_.end() && upper->timestamp == timestamp) {
        return *upper;
    }
    const auto* before =
        upper != imus_.begin() ? &*std::prev(upper) : (has_boundary_ ? &boundary_ : nullptr);
    const auto* after = upper != imus_.end() ? &*upper : nullptr;
    if (!before || !after || after->timestamp <= before->timestamp) {
        throw std::logic_error("Cannot interpolate Nori IMU boundary");
    }
    const double alpha = (timestamp - before->timestamp) / (after->timestamp - before->timestamp);
    return {
        before->acc + alpha * (after->acc - before->acc),
        before->gyro + alpha * (after->gyro - before->gyro), timestamp};
}

}  // namespace tassel_hardware
