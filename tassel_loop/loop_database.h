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
#include "loop_types.h"
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
    KeyframeId frame_id = kInvalidKeyframeId;
    double score = 0.0;
};

struct LoopVerificationCandidate {
    KeyframeId frame_id = kInvalidKeyframeId;
    double raw_score = 0.0;
    double likelihood = 1.0;
    double posterior = 0.0;
};

struct LoopQuery {
    KeyframeId frame_id = kInvalidKeyframeId;
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
    KeyframeId current_frame_id = kInvalidKeyframeId;
    KeyframeId candidate_frame_id = kInvalidKeyframeId;
    std::vector<cv::Point2f> current_points;
    std::vector<Eigen::Vector3d> host_points;
};

class LoopDatabase {
public:
    LoopDatabase(const std::string& vocabulary_path, LoopOptions options = {});

    LoopQuery addKeyframe(KeyframeId frame_id, const cv::Mat& gray_image);
    size_t attachLandmarks(KeyframeId frame_id, const std::vector<LandmarkInput>& landmarks);
    PnpMatches matchCandidateLandmarks(
        KeyframeId current_frame_id, KeyframeId candidate_frame_id) const;
    cv::Mat drawCandidate(KeyframeId current_frame_id, KeyframeId candidate_frame_id) const;
    bool contains(KeyframeId frame_id) const { return records_.find(frame_id) != records_.end(); }
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
    std::vector<KeyframeId> entry_frames_;
    std::unordered_map<KeyframeId, Record> records_;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_LOOP_DATABASE_H_
