#include "loop_database.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tassel_loop {

LoopDatabase::LoopDatabase(const std::string& vocabulary_path, LoopOptions options)
    : options_(options),
      detector_(cv::FastFeatureDetector::create(options.fast_threshold, true)),
      descriptor_(cv::xfeatures2d::BriefDescriptorExtractor::create()) {
    if (options_.fast_threshold <= 0 || options_.max_keypoints <= 0 ||
        options_.recent_exclusion < 0 || options_.top_k <= 0 ||
        options_.likelihood_pool_size < options_.top_k || options_.min_score < 0.0 ||
        options_.brief_ratio <= 0.0 || options_.brief_ratio >= 1.0 ||
        options_.brief_max_distance <= 0.0) {
        throw std::invalid_argument("Invalid loop database options");
    }
    DBoW3::Vocabulary vocabulary(vocabulary_path);
    if (vocabulary.empty()) {
        throw std::runtime_error("BRIEF vocabulary is empty: " + vocabulary_path);
    }
    database_.setVocabulary(vocabulary, false);
}

LoopDatabase::Record LoopDatabase::extract(const cv::Mat& gray_image) const {
    if (gray_image.empty() || gray_image.type() != CV_8UC1) {
        throw std::invalid_argument("LoopDatabase expects a non-empty CV_8UC1 image");
    }

    Record record;
    record.image = gray_image.clone();
    detector_->detect(record.image, record.keypoints);
    if (record.keypoints.size() > static_cast<size_t>(options_.max_keypoints)) {
        std::sort(
            record.keypoints.begin(), record.keypoints.end(),
            [](const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) {
                return lhs.response > rhs.response;
            });
        record.keypoints.resize(options_.max_keypoints);
    }
    descriptor_->compute(record.image, record.keypoints, record.descriptors);
    return record;
}

LoopQuery LoopDatabase::addKeyframe(tassel_utils::FrameId frame_id, const cv::Mat& gray_image) {
    if (records_.find(frame_id) != records_.end()) {
        throw std::invalid_argument("Duplicate loop keyframe id");
    }

    Record record = extract(gray_image);
    LoopQuery query;
    query.frame_id = frame_id;
    query.keypoint_count = static_cast<int>(record.keypoints.size());

    const int eligible_count = static_cast<int>(entry_frames_.size()) - options_.recent_exclusion;
    std::vector<std::pair<tassel_utils::FrameId, double>> visual_scores;
    if (!record.descriptors.empty() && eligible_count > 0) {
        DBoW3::QueryResults results;
        // DBoW3 1cc587b 实际将 max_id 作为开区间上界，
        // 与公开头文件中记录的闭区间语义不同。
        database_.query(
            record.descriptors, results, std::min(options_.likelihood_pool_size, eligible_count),
            eligible_count);
        for (const auto& result : results) {
            if (result.Id >= entry_frames_.size()) {
                continue;
            }
            visual_scores.emplace_back(entry_frames_[result.Id], result.Score);
            if (result.Score >= options_.min_score &&
                query.candidates.size() < static_cast<size_t>(options_.top_k)) {
                query.candidates.push_back({entry_frames_[result.Id], result.Score});
            }
        }
    }

    const HypothesisUpdate hypothesis_update = hypothesis_tracker_.update(visual_scores);
    query.loop_probability = hypothesis_update.loop_probability;
    query.hypotheses = hypothesis_update.hypotheses;

    for (const LoopHypothesis& hypothesis : query.hypotheses) {
        if (hypothesis.raw_score >= options_.min_score) {
            query.verification_candidates.push_back(
                {hypothesis.frame_id, hypothesis.raw_score, hypothesis.likelihood,
                 hypothesis.posterior});
        }
    }
    if (!record.descriptors.empty()) {
        const DBoW3::EntryId entry_id = database_.add(record.descriptors);
        if (entry_id != entry_frames_.size()) {
            throw std::runtime_error("DBoW3 entry sequence is inconsistent");
        }
        entry_frames_.push_back(frame_id);
        hypothesis_tracker_.addPlace(frame_id);
    }
    records_.emplace(frame_id, std::move(record));
    return query;
}

size_t LoopDatabase::attachLandmarks(
    tassel_utils::FrameId frame_id, const std::vector<LandmarkInput>& landmarks) {
    const auto record_it = records_.find(frame_id);
    if (record_it == records_.end()) {
        throw std::invalid_argument("Cannot attach landmarks to an unknown keyframe");
    }
    Record& record = record_it->second;
    if (!record.landmark_descriptors.empty() || !record.landmark_host_points.empty()) {
        throw std::invalid_argument("Landmarks are already attached to this keyframe");
    }

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(landmarks.size());
    for (size_t index = 0; index < landmarks.size(); ++index) {
        const LandmarkInput& landmark = landmarks[index];
        if (!std::isfinite(landmark.pixel.x) || !std::isfinite(landmark.pixel.y) ||
            !landmark.host_uv.allFinite() || !std::isfinite(landmark.host_depth) ||
            landmark.host_depth <= 0.0) {
            continue;
        }
        cv::KeyPoint keypoint(landmark.pixel, 1.0f);
        keypoint.class_id = static_cast<int>(index);
        keypoints.push_back(keypoint);
    }

    descriptor_->compute(record.image, keypoints, record.landmark_descriptors);
    record.landmark_host_points.reserve(keypoints.size());
    for (const cv::KeyPoint& keypoint : keypoints) {
        if (keypoint.class_id < 0 || keypoint.class_id >= static_cast<int>(landmarks.size())) {
            throw std::runtime_error("BRIEF did not preserve landmark keypoint identity");
        }
        const LandmarkInput& landmark = landmarks[keypoint.class_id];
        record.landmark_host_points.push_back(landmark.host_uv * landmark.host_depth);
    }
    if (record.landmark_descriptors.rows != static_cast<int>(record.landmark_host_points.size())) {
        throw std::runtime_error("Landmark descriptor alignment is inconsistent");
    }
    return record.landmark_host_points.size();
}

cv::Mat LoopDatabase::drawCandidate(
    tassel_utils::FrameId current_frame_id, tassel_utils::FrameId candidate_frame_id) const {
    const auto current_it = records_.find(current_frame_id);
    const auto candidate_it = records_.find(candidate_frame_id);
    if (current_it == records_.end() || candidate_it == records_.end()) {
        return {};
    }

    std::vector<cv::DMatch> matches;
    cv::BFMatcher(cv::NORM_HAMMING, true)
        .match(current_it->second.descriptors, candidate_it->second.descriptors, matches);
    std::sort(matches.begin(), matches.end());
    constexpr size_t kMaxDisplayMatches = 80;
    if (matches.size() > kMaxDisplayMatches) {
        matches.resize(kMaxDisplayMatches);
    }

    cv::Mat image;
    cv::drawMatches(
        current_it->second.image, current_it->second.keypoints, candidate_it->second.image,
        candidate_it->second.keypoints, matches, image);
    return image;
}

PnpMatches LoopDatabase::matchCandidateLandmarks(
    tassel_utils::FrameId current_frame_id, tassel_utils::FrameId candidate_frame_id) const {
    PnpMatches output;
    output.current_frame_id = current_frame_id;
    output.candidate_frame_id = candidate_frame_id;
    const auto current_it = records_.find(current_frame_id);
    const auto candidate_it = records_.find(candidate_frame_id);
    if (candidate_it == records_.end() || candidate_it->second.landmark_descriptors.empty()) {
        return output;
    }
    if (current_it == records_.end()) {
        return output;
    }

    std::vector<std::vector<cv::DMatch>> nearest_matches;
    cv::BFMatcher(cv::NORM_HAMMING, false)
        .knnMatch(
            current_it->second.descriptors, candidate_it->second.landmark_descriptors,
            nearest_matches, 2);
    std::vector<cv::DMatch> matches;
    matches.reserve(nearest_matches.size());
    for (const auto& nearest : nearest_matches) {
        if (nearest.size() == 2 && nearest[0].distance <= options_.brief_max_distance &&
            nearest[0].distance < options_.brief_ratio * nearest[1].distance) {
            matches.push_back(nearest[0]);
        }
    }
    // 限制 PnP 计算量时，仅保留描述子距离最小的匹配。
    std::sort(matches.begin(), matches.end());
    constexpr size_t kMaxPnpMatches = 300;
    if (matches.size() > kMaxPnpMatches) {
        matches.resize(kMaxPnpMatches);
    }
    output.current_points.reserve(matches.size());
    output.host_points.reserve(matches.size());
    for (const cv::DMatch& match : matches) {
        output.current_points.push_back(current_it->second.keypoints[match.queryIdx].pt);
        output.host_points.push_back(candidate_it->second.landmark_host_points[match.trainIdx]);
    }
    return output;
}

}  // namespace tassel_loop
