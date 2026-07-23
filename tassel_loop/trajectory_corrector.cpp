#include "trajectory_corrector.h"

#include <stdexcept>
#include <string>

namespace tassel_loop {
namespace {

void validateOrder(const std::vector<TimedPose>& poses, const char* name) {
    for (size_t index = 1; index < poses.size(); ++index) {
        if (poses[index].frame_id <= poses[index - 1].frame_id) {
            throw std::invalid_argument(std::string(name) + " must be strictly ordered");
        }
    }
}

}  // namespace

std::vector<Sophus::SE3d> TrajectoryCorrector::correct(
    const std::vector<TimedPose>& local_trajectory, const std::vector<TimedPose>& local_keyframes,
    const std::vector<TimedPose>& global_keyframes) {
    if (local_keyframes.size() != global_keyframes.size()) {
        throw std::invalid_argument("Local and global keyframe counts differ");
    }
    validateOrder(local_trajectory, "Local trajectory");
    validateOrder(local_keyframes, "Local keyframes");
    validateOrder(global_keyframes, "Global keyframes");
    for (size_t index = 0; index < local_keyframes.size(); ++index) {
        if (local_keyframes[index].frame_id != global_keyframes[index].frame_id) {
            throw std::invalid_argument("Local and global keyframe ids differ");
        }
    }

    std::vector<Sophus::SE3d> corrected;
    corrected.reserve(local_trajectory.size());
    Sophus::SE3d global_T_local;
    size_t keyframe_index = 0;
    for (const TimedPose& local_pose : local_trajectory) {
        while (keyframe_index < local_keyframes.size() &&
               local_keyframes[keyframe_index].frame_id <= local_pose.frame_id) {
            global_T_local = global_keyframes[keyframe_index].pose *
                             local_keyframes[keyframe_index].pose.inverse();
            ++keyframe_index;
        }
        corrected.push_back(global_T_local * local_pose.pose);
    }
    return corrected;
}

}  // namespace tassel_loop
