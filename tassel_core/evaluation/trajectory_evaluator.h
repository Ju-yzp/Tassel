#ifndef TASSEL_CORE_EVALUATION_TRAJECTORY_EVALUATOR_H_
#define TASSEL_CORE_EVALUATION_TRAJECTORY_EVALUATOR_H_

#include <optional>
#include <sophus/se3.hpp>
#include <vector>

namespace tassel_core::evaluation {

struct TimedPose {
    double timestamp = 0.0;
    Sophus::SE3d pose;
};

struct PosePair {
    double timestamp = 0.0;
    Sophus::SE3d estimate;
    Sophus::SE3d truth;
};

struct TrajectoryError {
    Sophus::SE3d alignment;
    double position_rmse = 0.0;
    double terminal_position_error = 0.0;
    double rotation_rmse = 0.0;
};

std::optional<Sophus::SE3d> interpolatePose(
    const std::vector<TimedPose>& poses, double timestamp, double max_interval);

Sophus::SE3d alignByYawAndTranslation(const std::vector<PosePair>& poses);

TrajectoryError evaluateTrajectory(const std::vector<PosePair>& poses);

}  // namespace tassel_core::evaluation

#endif  // TASSEL_CORE_EVALUATION_TRAJECTORY_EVALUATOR_H_
