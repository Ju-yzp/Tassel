#ifndef TASSEL_LOOP_POSE_GRAPH_H_
#define TASSEL_LOOP_POSE_GRAPH_H_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <Eigen/Core>

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "tassel_utils/types.h"

namespace tassel_loop {

struct PoseGraphStats {
    size_t node_count = 0;
    size_t odometry_factor_count = 0;
    size_t loop_factor_count = 0;
    bool optimized = false;
};

struct GraphPose {
    Eigen::Matrix3d world_R_camera = Eigen::Matrix3d::Identity();
    Eigen::Vector3d world_t_camera = Eigen::Vector3d::Zero();
};

struct PoseGraphOptimizationStats {
    double error_before = 0.0;
    double error_after = 0.0;
    double max_translation_correction = 0.0;
    double max_rotation_correction = 0.0;
    double loop_translation_variance = 0.0;
    double loop_rotation_variance = 0.0;
    double max_normalized_loop_error = 0.0;
    bool loop_rejected = false;
};

struct PoseLoopNoise {
    double translation_variance = 0.01;
    double rotation_variance = 0.0009;
};

class PoseGraph {
public:
    void addKeyframe(
        tassel_utils::FrameId frame_id, const Eigen::Matrix3d& local_R_camera,
        const Eigen::Vector3d& local_t_camera);
    bool addPoseLoop(
        tassel_utils::FrameId candidate_frame_id, tassel_utils::FrameId current_frame_id,
        const Eigen::Matrix3d& candidate_R_current, const Eigen::Vector3d& candidate_t_current,
        const PoseLoopNoise& noise = {});
    bool optimize(double max_normalized_loop_error = 0.0);
    std::optional<GraphPose> pose(tassel_utils::FrameId frame_id) const;
    std::vector<std::pair<tassel_utils::FrameId, GraphPose>> poses() const;
    bool contains(tassel_utils::FrameId frame_id) const {
        return frame_keys_.find(frame_id) != frame_keys_.end();
    }

    PoseGraphStats stats() const;
    const PoseGraphOptimizationStats& lastOptimizationStats() const {
        return last_optimization_stats_;
    }
    const gtsam::NonlinearFactorGraph& factors() const { return factors_; }
    const gtsam::Values& values() const { return values_; }

private:
    gtsam::NonlinearFactorGraph factors_;
    gtsam::Values values_;
    std::unordered_map<tassel_utils::FrameId, gtsam::Key> frame_keys_;
    std::vector<tassel_utils::FrameId> frame_order_;
    gtsam::Key last_key_ = 0;
    gtsam::Pose3 last_local_pose_;
    bool has_last_key_ = false;
    size_t odometry_factor_count_ = 0;
    size_t loop_factor_count_ = 0;
    bool optimized_ = false;
    PoseGraphOptimizationStats last_optimization_stats_;

    struct PendingLoop {
        gtsam::Key candidate_key;
        gtsam::Key current_key;
        gtsam::Pose3 measurement;
        PoseLoopNoise noise;
        bool was_optimized;
    };
    std::optional<PendingLoop> pending_loop_;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_POSE_GRAPH_H_
