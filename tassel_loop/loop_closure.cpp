#include "loop_closure.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tassel_loop {

LoopClosure::LoopClosure(
    const std::string& vocabulary_path, LoopClosureOptions options, Undistort undistort,
    ResultCallback callback)
    : options_(std::move(options)),
      undistort_(std::move(undistort)),
      callback_(std::move(callback)),
      database_(vocabulary_path, options_.database),
      verifier_(
          options_.pnp_min_inliers, options_.pnp_min_inlier_ratio, options_.pnp_inlier_threshold,
          options_.pnp_max_iterations, options_.pnp_confidence, options_.variance_quantile_divisor,
          options_.max_translation_variance) {
    if (!undistort_ || options_.max_candidates <= 0 || options_.min_probability < 0.0 ||
        options_.min_probability > 1.0 || options_.min_likelihood_ratio < 1.0 ||
        options_.fallback_min_score < 0.0 || options_.optimize_max_error < 0.0) {
        throw std::invalid_argument("Invalid loop closure options");
    }
}

LoopClosure::~LoopClosure() {
    try {
        finish();
    } catch (...) {
    }
}

void LoopClosure::submitPose(LoopPoseTransaction transaction) { submit(std::move(transaction)); }

void LoopClosure::submitKeyframe(LoopKeyframeTransaction transaction) {
    if (transaction.image.empty() || transaction.image.type() != CV_8UC1) {
        throw std::invalid_argument("Loop keyframe image must be grayscale");
    }
    transaction.image = transaction.image.clone();
    submit(std::move(transaction));
}

void LoopClosure::submitLandmarks(LoopLandmarkTransaction transaction) {
    submit(std::move(transaction));
}

void LoopClosure::submit(Transaction transaction) {
    bool start_worker = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_error_) {
            std::rethrow_exception(worker_error_);
        }
        if (finishing_) {
            throw std::logic_error("Loop closure is finishing");
        }
        queue_.push_back(std::move(transaction));
        if (!worker_active_) {
            worker_active_ = true;
            start_worker = true;
        }
    }
    if (start_worker) {
        try {
            std::thread(&LoopClosure::drain, this).detach();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_error_ = std::current_exception();
            queue_.clear();
            finishing_ = true;
            worker_active_ = false;
            finished_condition_.notify_all();
            throw;
        }
    }
}

void LoopClosure::finish() {
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        finishing_ = true;
        finished_condition_.wait(lock, [this]() { return !worker_active_; });
        error = worker_error_;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void LoopClosure::drain() {
    try {
        std::deque<Transaction> batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch.swap(queue_);
        }
        for (const Transaction& transaction : batch) {
            std::visit([this](const auto& value) { process(value); }, transaction);
        }

        bool schedule_next = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) {
                worker_active_ = false;
                finished_condition_.notify_all();
            } else {
                schedule_next = true;
            }
        }
        if (schedule_next) {
            std::thread(&LoopClosure::drain, this).detach();
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_error_ = std::current_exception();
        queue_.clear();
        finishing_ = true;
        worker_active_ = false;
        finished_condition_.notify_all();
    }
}

void LoopClosure::process(const LoopPoseTransaction& transaction) {
    if (transaction.frame_id == tassel_utils::kInvalidFrameId) {
        throw std::invalid_argument("Loop pose frame id is invalid");
    }
    local_poses_[transaction.frame_id] = transaction.local_T_imu;
    local_trajectory_.push_back({transaction.frame_id, transaction.local_T_imu});
    LoopClosureResult result = snapshot(LoopEvent::kGlobalPoseUpdated);
    result.current_frame_id = transaction.frame_id;
    result.corrected_trajectory.push_back(global_T_local_ * transaction.local_T_imu);
    publish(std::move(result));
}

void LoopClosure::process(const LoopKeyframeTransaction& transaction) {
    if (transaction.frame_id == tassel_utils::kInvalidFrameId ||
        graph_.contains(transaction.frame_id)) {
        throw std::invalid_argument("Loop keyframe is invalid or duplicated");
    }
    local_poses_[transaction.frame_id] = transaction.local_T_imu;
    keyframe_images_[transaction.frame_id] = transaction.image;
    const Sophus::SE3d local_T_camera = transaction.local_T_imu * options_.imu_T_camera;
    graph_.addKeyframe(
        transaction.frame_id, local_T_camera.rotationMatrix(), local_T_camera.translation());
    LoopClosureResult result = snapshot(LoopEvent::kGraphUpdated);
    result.current_frame_id = transaction.frame_id;
    publish(std::move(result));
}

void LoopClosure::process(const LoopLandmarkTransaction& transaction) {
    const auto image = keyframe_images_.find(transaction.frame_id);
    if (image == keyframe_images_.end() || transaction.landmarks.empty()) {
        LoopClosureResult result = snapshot(LoopEvent::kLoopRejected);
        result.current_frame_id = transaction.frame_id;
        result.reason = image == keyframe_images_.end() ? "missing_keyframe" : "no_landmarks";
        if (image != keyframe_images_.end()) {
            keyframe_images_.erase(image);
        }
        publish(std::move(result));
        return;
    }
    const LoopQuery query = database_.addKeyframe(transaction.frame_id, image->second);
    database_.attachLandmarks(transaction.frame_id, transaction.landmarks);
    keyframe_images_.erase(image);
    verify(query);
}

void LoopClosure::verify(const LoopQuery& query) {
    const bool posterior_accepted = query.loop_probability >= options_.min_probability;
    const size_t count = std::min(
        query.verification_candidates.size(), static_cast<size_t>(options_.max_candidates));
    PnpMatches best_matches;
    PnpVerification best;
    size_t best_rank = 0;
    for (size_t index = 0; index < count; ++index) {
        const auto& candidate = query.verification_candidates[index];
        if ((!posterior_accepted && candidate.raw_score < options_.fallback_min_score) ||
            candidate.likelihood < options_.min_likelihood_ratio) {
            continue;
        }
        PnpMatches matches = database_.matchCandidateLandmarks(query.frame_id, candidate.frame_id);
        std::vector<Eigen::Vector2d> normalized;
        normalized.reserve(matches.current_points.size());
        for (const cv::Point2f& point : matches.current_points) {
            normalized.push_back(undistort_(Eigen::Vector2d(point.x, point.y)));
        }
        const PnpVerification verification = verifier_.verify(matches.host_points, normalized);
        if (verification.accepted &&
            (best_rank == 0 || verification.inlier_count > best.inlier_count ||
             (verification.inlier_count == best.inlier_count &&
              verification.mean_reprojection_error < best.mean_reprojection_error))) {
            best_matches = std::move(matches);
            best = verification;
            best_rank = index + 1;
        }
    }
    if (best_rank == 0) {
        return;
    }

    const Sophus::SE3d measurement(best.host_R_current, best.host_t_current);
    const bool added = graph_.addPoseLoop(
        best_matches.candidate_frame_id, best_matches.current_frame_id,
        measurement.rotationMatrix(), measurement.translation(),
        PoseLoopNoise{best.translation_variance, best.rotation_variance});
    if (!added) {
        return;
    }
    const bool accepted = graph_.optimize(options_.optimize_max_error);
    LoopClosureResult result =
        snapshot(accepted ? LoopEvent::kLoopAccepted : LoopEvent::kLoopRejected);
    result.current_frame_id = best_matches.current_frame_id;
    result.candidate_frame_id = best_matches.candidate_frame_id;
    result.verification = best;
    result.optimization = graph_.lastOptimizationStats();
    if (!accepted) {
        result.reason = "graph_inconsistent";
        publish(std::move(result));
        return;
    }
    result.match_image =
        database_.drawCandidate(best_matches.current_frame_id, best_matches.candidate_frame_id);
    const auto graph_pose = graph_.pose(best_matches.current_frame_id);
    const auto local_pose = local_poses_.find(best_matches.current_frame_id);
    if (graph_pose && local_pose != local_poses_.end()) {
        const Sophus::SE3d global_T_camera(graph_pose->world_R_camera, graph_pose->world_t_camera);
        result.global_T_local =
            global_T_camera * (local_pose->second * options_.imu_T_camera).inverse();
        global_T_local_ = result.global_T_local;
    }
    publish(std::move(result));
}

LoopClosureResult LoopClosure::snapshot(LoopEvent event) const {
    LoopClosureResult result;
    result.event = event;
    result.graph_stats = graph_.stats();
    if (event != LoopEvent::kGlobalPoseUpdated) {
        result.graph_poses = graph_.poses();
    }
    if (event == LoopEvent::kLoopAccepted) {
        std::vector<TimedPose> local_keyframes;
        std::vector<TimedPose> global_keyframes;
        for (const auto& [frame_id, graph_pose] : result.graph_poses) {
            const auto local_pose = local_poses_.find(frame_id);
            if (local_pose != local_poses_.end()) {
                local_keyframes.push_back({frame_id, local_pose->second * options_.imu_T_camera});
                global_keyframes.push_back(
                    {frame_id, Sophus::SE3d(graph_pose.world_R_camera, graph_pose.world_t_camera)});
            }
        }
        result.corrected_trajectory =
            TrajectoryCorrector::correct(local_trajectory_, local_keyframes, global_keyframes);
    }
    return result;
}

void LoopClosure::publish(LoopClosureResult result) const {
    if (callback_) {
        callback_(result);
    }
}

}  // namespace tassel_loop
