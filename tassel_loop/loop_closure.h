#ifndef TASSEL_LOOP_LOOP_CLOSURE_H_
#define TASSEL_LOOP_LOOP_CLOSURE_H_

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "loop_database.h"
#include "pnp_verifier.h"
#include "pose_graph.h"
#include "trajectory_corrector.h"

namespace tassel_loop {

struct LoopClosureOptions {
    LoopOptions database;
    int max_candidates = 3;
    double min_probability = 0.03;
    double min_likelihood_ratio = 1.01;
    double fallback_min_score = 0.20;
    int pnp_min_inliers = 20;
    double pnp_min_inlier_ratio = 0.25;
    double pnp_inlier_threshold = 0.006;
    int pnp_max_iterations = 1000;
    double pnp_confidence = 0.999;
    int variance_quantile_divisor = 4;
    double max_translation_variance = 0.0;
    double optimize_max_error = 3.0;
    Sophus::SE3d imu_T_camera;
};

struct LoopPoseTransaction {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    Sophus::SE3d local_T_imu;
};

struct LoopKeyframeTransaction {
    tassel_utils::FrameId source_frame_id = tassel_utils::kInvalidFrameId;
    cv::Mat image;
    Sophus::SE3d local_T_imu;
    KeyframeId keyframe_id = kInvalidKeyframeId;
};

struct LoopLandmarkTransaction {
    KeyframeId keyframe_id = kInvalidKeyframeId;
    std::vector<LandmarkInput> landmarks;
};

enum class LoopEvent { GlobalPoseUpdated, GraphUpdated, LoopAccepted, LoopRejected };

struct LoopClosureResult {
    LoopEvent event = LoopEvent::GraphUpdated;
    tassel_utils::FrameId current_frame_id = tassel_utils::kInvalidFrameId;
    tassel_utils::FrameId candidate_frame_id = tassel_utils::kInvalidFrameId;
    std::string reason;
    cv::Mat match_image;
    std::vector<std::pair<KeyframeId, GraphPose>> graph_poses;
    std::vector<Sophus::SE3d> corrected_trajectory;
    Sophus::SE3d global_T_local;
    PnpVerification verification;
    PoseGraphOptimizationStats optimization;
    PoseGraphStats graph_stats;
};

class LoopClosure {
public:
    using Undistort = std::function<Eigen::Vector2d(const Eigen::Vector2d&)>;
    using ResultCallback = std::function<void(const LoopClosureResult&)>;

    LoopClosure(
        const std::string& vocabulary_path, LoopClosureOptions options, Undistort undistort,
        ResultCallback callback = {});
    ~LoopClosure();

    LoopClosure(const LoopClosure&) = delete;
    LoopClosure& operator=(const LoopClosure&) = delete;

    void submitPose(LoopPoseTransaction transaction);
    KeyframeId submitKeyframe(LoopKeyframeTransaction transaction);
    void submitLandmarks(LoopLandmarkTransaction transaction);
    void finish();

private:
    using Transaction =
        std::variant<LoopPoseTransaction, LoopKeyframeTransaction, LoopLandmarkTransaction>;

    KeyframeId submit(Transaction transaction, bool assign_keyframe_id = false);
    void drain();
    void process(const LoopPoseTransaction& transaction);
    void process(const LoopKeyframeTransaction& transaction);
    void process(const LoopLandmarkTransaction& transaction);
    void verify(const LoopQuery& query);
    LoopClosureResult snapshot(LoopEvent event) const;
    void publish(LoopClosureResult result) const;

    LoopClosureOptions options_;
    Undistort undistort_;
    ResultCallback callback_;
    LoopDatabase database_;
    PnpVerifier verifier_;
    PoseGraph graph_;
    std::unordered_map<KeyframeId, tassel_utils::FrameId> keyframe_sources_;
    std::unordered_map<KeyframeId, Sophus::SE3d> keyframe_local_poses_;
    std::unordered_map<KeyframeId, cv::Mat> keyframe_images_;
    std::vector<TimedPose> local_trajectory_;
    Sophus::SE3d global_T_local_;
    KeyframeId next_keyframe_id_ = 0;

    std::mutex mutex_;
    std::condition_variable finished_condition_;
    std::deque<Transaction> queue_;
    bool finishing_ = false;
    bool worker_active_ = false;
    std::exception_ptr worker_error_;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_LOOP_CLOSURE_H_
