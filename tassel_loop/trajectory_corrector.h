#ifndef TASSEL_LOOP_TRAJECTORY_CORRECTOR_H_
#define TASSEL_LOOP_TRAJECTORY_CORRECTOR_H_

#include <sophus/se3.hpp>

#include <vector>

#include "tassel_utils/types.h"

namespace tassel_loop {

struct TimedPose {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    Sophus::SE3d pose;
};

class TrajectoryCorrector {
public:
    static std::vector<Sophus::SE3d> correct(
        const std::vector<TimedPose>& local_trajectory,
        const std::vector<TimedPose>& local_keyframes,
        const std::vector<TimedPose>& global_keyframes);
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_TRAJECTORY_CORRECTOR_H_
