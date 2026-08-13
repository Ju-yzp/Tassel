#include "evaluation/trajectory_evaluator.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tassel_core::evaluation {

std::optional<Sophus::SE3d> interpolatePose(
    const std::vector<TimedPose>& poses, double timestamp) {
    if (poses.empty() || timestamp < poses.front().timestamp || timestamp > poses.back().timestamp) {
        return std::nullopt;
    }

    const auto upper = std::lower_bound(
        poses.begin(), poses.end(), timestamp,
        [](const TimedPose& pose, double value) { return pose.timestamp < value; });
    if (upper == poses.begin()) {
        return upper->pose;
    }
    if (upper == poses.end()) {
        return poses.back().pose;
    }

    const auto lower = std::prev(upper);
    const double duration = upper->timestamp - lower->timestamp;
    if (duration <= 0.0) {
        throw std::logic_error("Ground-truth timestamps must be strictly increasing");
    }
    const double alpha = (timestamp - lower->timestamp) / duration;
    const Eigen::Vector3d position =
        (1.0 - alpha) * lower->pose.translation() + alpha * upper->pose.translation();
    const Eigen::Quaterniond orientation =
        lower->pose.unit_quaternion().slerp(alpha, upper->pose.unit_quaternion()).normalized();
    return Sophus::SE3d(orientation, position);
}

Sophus::SE3d alignByYawAndTranslation(const std::vector<PosePair>& poses) {
    if (poses.empty()) {
        throw std::invalid_argument("Cannot align an empty trajectory");
    }

    Eigen::Vector3d estimate_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d truth_mean = Eigen::Vector3d::Zero();
    for (const auto& pose : poses) {
        estimate_mean += pose.estimate.translation();
        truth_mean += pose.truth.translation();
    }
    estimate_mean /= static_cast<double>(poses.size());
    truth_mean /= static_cast<double>(poses.size());

    double xx = 0.0;
    double xy = 0.0;
    for (const auto& pose : poses) {
        const Eigen::Vector3d estimate = pose.estimate.translation() - estimate_mean;
        const Eigen::Vector3d truth = pose.truth.translation() - truth_mean;
        xx += estimate.x() * truth.x() + estimate.y() * truth.y();
        xy += estimate.x() * truth.y() - estimate.y() * truth.x();
    }
    const double yaw = std::atan2(xy, xx);
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return Sophus::SE3d(rotation, truth_mean - rotation * estimate_mean);
}

TrajectoryError evaluateTrajectory(const std::vector<PosePair>& poses) {
    if (poses.empty()) {
        throw std::invalid_argument("Cannot evaluate an empty trajectory");
    }

    TrajectoryError result;
    result.alignment = alignByYawAndTranslation(poses);
    double squared_position_error = 0.0;
    double squared_rotation_error = 0.0;
    for (const auto& pose : poses) {
        const Sophus::SE3d aligned_estimate = result.alignment * pose.estimate;
        const Eigen::Vector3d position_error =
            aligned_estimate.translation() - pose.truth.translation();
        const Eigen::Vector3d rotation_error =
            (pose.truth.so3().inverse() * aligned_estimate.so3()).log();
        squared_position_error += position_error.squaredNorm();
        squared_rotation_error += rotation_error.squaredNorm();
        result.terminal_position_error = position_error.norm();
    }

    const double count = static_cast<double>(poses.size());
    result.position_rmse = std::sqrt(squared_position_error / count);
    result.rotation_rmse = std::sqrt(squared_rotation_error / count);
    return result;
}

}  // namespace tassel_core::evaluation
