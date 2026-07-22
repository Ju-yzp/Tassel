#ifndef TASSEL_LOOP_LOOP_DATABASE_H_
#define TASSEL_LOOP_LOOP_DATABASE_H_

#include <DBoW3/DBoW3.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "loop_hypothesis_tracker.h"
#include "tassel_utils/types.h"

namespace tassel_loop {

struct LoopOptions {
    int fast_threshold = 20;
    int max_keypoints = 800;
    int recent_exclusion = 30;
    int top_k = 3;
    int likelihood_pool_size = 20;
    double min_score = 0.05;
    double brief_ratio = 0.8;
    double brief_max_distance = 80.0;
};

struct LoopCandidate {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    double score = 0.0;
};

struct LoopVerificationCandidate {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    double raw_score = 0.0;
    double likelihood = 1.0;
    double posterior = 0.0;
};

struct LoopQuery {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    int keypoint_count = 0;
    double loop_probability = 0.0;
    std::vector<LoopCandidate> candidates;
    std::vector<LoopHypothesis> hypotheses;
    std::vector<LoopVerificationCandidate> verification_candidates;
};

struct LandmarkInput {
    int feature_id = -1;
    cv::Point2f pixel;
    Eigen::Vector3d host_uv = Eigen::Vector3d::Zero();
    double host_depth = 0.0;
};

struct PnpMatches {
    tassel_utils::FrameId current_frame_id = tassel_utils::kInvalidFrameId;
    tassel_utils::FrameId candidate_frame_id = tassel_utils::kInvalidFrameId;
    std::vector<cv::Point2f> current_points;
    std::vector<Eigen::Vector3d> host_points;
};

class LoopDatabase {
public:
    LoopDatabase(const std::string& vocabulary_path, LoopOptions options = {});

    LoopQuery addKeyframe(tassel_utils::FrameId frame_id, const cv::Mat& gray_image);
    size_t attachLandmarks(
        tassel_utils::FrameId frame_id, const std::vector<LandmarkInput>& landmarks);
    PnpMatches matchCandidateLandmarks(
        tassel_utils::FrameId current_frame_id, tassel_utils::FrameId candidate_frame_id) const;
    cv::Mat drawCandidate(
        tassel_utils::FrameId current_frame_id, tassel_utils::FrameId candidate_frame_id) const;
    bool contains(tassel_utils::FrameId frame_id) const {
        return records_.find(frame_id) != records_.end();
    }
    size_t size() const { return records_.size(); }

private:
    struct Record {
        cv::Mat image;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cv::Mat landmark_descriptors;
        std::vector<Eigen::Vector3d> landmark_host_points;
    };

    Record extract(const cv::Mat& gray_image) const;

    LoopOptions options_;
    cv::Ptr<cv::FastFeatureDetector> detector_;
    cv::Ptr<cv::Feature2D> descriptor_;
    DBoW3::Database database_;
    LoopHypothesisTracker hypothesis_tracker_;
    std::vector<tassel_utils::FrameId> entry_frames_;
    std::unordered_map<tassel_utils::FrameId, Record> records_;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_LOOP_DATABASE_H_
