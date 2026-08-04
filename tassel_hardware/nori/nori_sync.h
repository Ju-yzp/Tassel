#pragma once

#include <opencv2/core/mat.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_hardware {

struct NoriImageFrame {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    cv::Mat image;
};

struct NoriSyncedFrame {
    NoriImageFrame mono;
    std::vector<tassel_utils::IMUMeasurement> imu;
};

class NoriSync {
public:
    explicit NoriSync(double camera_to_imu_delay);

    void push(NoriImageFrame mono, const std::vector<tassel_utils::IMUMeasurement>& imu);
    bool waitPop(NoriSyncedFrame& output);
    void stop();
    size_t size() const;

private:
    tassel_utils::IMUMeasurement interpolate(double timestamp) const;

    double camera_to_imu_delay_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<NoriImageFrame> frames_;
    std::deque<tassel_utils::IMUMeasurement> imus_;
    tassel_utils::IMUMeasurement boundary_{};
    bool has_boundary_ = false;
    bool stopped_ = false;
};

}  // namespace tassel_hardware
