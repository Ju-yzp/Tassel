#include "loop_hypothesis_tracker.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tassel_loop {

LoopHypothesisTracker::LoopHypothesisTracker(std::vector<double> prediction)
    : prediction_(std::move(prediction)) {
    if (prediction_.size() < 2 || prediction_.front() < 0.0 || prediction_.front() >= 1.0 ||
        std::any_of(
            prediction_.begin() + 1, prediction_.end(), [](double value) { return value < 0.0; })) {
        throw std::invalid_argument("Invalid loop hypothesis prediction kernel");
    }
}

void LoopHypothesisTracker::addPlace(tassel_utils::FrameId frame_id) {
    if (place_indices_.contains(frame_id)) {
        throw std::invalid_argument("Duplicate loop hypothesis place id");
    }
    place_indices_[frame_id] = places_.size();
    places_.push_back(frame_id);
    posterior_.push_back(0.0);
}

std::vector<double> LoopHypothesisTracker::adjustedLikelihoods(
    const std::vector<std::pair<tassel_utils::FrameId, double>>& visual_scores,
    double& virtual_likelihood) const {
    std::vector<double> likelihoods(places_.size(), 1.0);
    std::vector<double> positive_scores;
    positive_scores.reserve(visual_scores.size());
    for (const auto& [frame_id, score] : visual_scores) {
        if (!place_indices_.contains(frame_id) || !std::isfinite(score) || score < 0.0) {
            throw std::invalid_argument("Invalid visual likelihood input");
        }
        if (score > 0.0) {
            positive_scores.push_back(score);
        }
    }
    if (positive_scores.empty()) {
        virtual_likelihood = 2.0;
        return likelihoods;
    }

    const double mean = std::accumulate(positive_scores.begin(), positive_scores.end(), 0.0) /
                        positive_scores.size();
    double variance = 0.0;
    for (double score : positive_scores) {
        variance += (score - mean) * (score - mean);
    }
    const double standard_deviation = std::sqrt(variance / positive_scores.size());
    constexpr double kEpsilon = 1e-4;
    for (const auto& [frame_id, score] : visual_scores) {
        if (score > mean + standard_deviation && mean > 0.0) {
            likelihoods[place_indices_.at(frame_id)] =
                (score - (standard_deviation - kEpsilon)) / mean;
        }
    }
    virtual_likelihood = standard_deviation > kEpsilon ? mean / standard_deviation + 1.0 : 2.0;
    return likelihoods;
}

std::vector<double> LoopHypothesisTracker::predictPrior() const {
    const size_t place_count = places_.size();
    std::vector<double> prior(place_count + 1, 0.0);
    if (place_count == 0) {
        prior.front() = 1.0;
        return prior;
    }

    constexpr double kVirtualStayProbability = 0.9;
    prior.front() += virtual_posterior_ * kVirtualStayProbability;
    const double virtual_to_places =
        virtual_posterior_ * (1.0 - kVirtualStayProbability) / place_count;
    for (size_t target = 0; target < place_count; ++target) {
        prior[target + 1] += virtual_to_places;
    }

    const double place_to_virtual = prediction_.front();
    for (size_t source = 0; source < place_count; ++source) {
        const double source_posterior = posterior_[source];
        prior.front() += source_posterior * place_to_virtual;
        double weight_sum = 0.0;
        for (size_t target = 0; target < place_count; ++target) {
            const size_t distance = source > target ? source - target : target - source;
            if (distance + 1 < prediction_.size()) {
                weight_sum += prediction_[distance + 1];
            }
        }
        if (weight_sum <= 0.0) {
            continue;
        }
        for (size_t target = 0; target < place_count; ++target) {
            const size_t distance = source > target ? source - target : target - source;
            if (distance + 1 < prediction_.size()) {
                prior[target + 1] += source_posterior * (1.0 - place_to_virtual) *
                                     prediction_[distance + 1] / weight_sum;
            }
        }
    }
    return prior;
}

HypothesisUpdate LoopHypothesisTracker::update(
    const std::vector<std::pair<tassel_utils::FrameId, double>>& visual_scores) {
    double virtual_likelihood = 1.0;
    const std::vector<double> likelihoods = adjustedLikelihoods(visual_scores, virtual_likelihood);
    const std::vector<double> prior = predictPrior();
    virtual_posterior_ = virtual_likelihood * prior.front();
    double normalization = virtual_posterior_;
    for (size_t index = 0; index < places_.size(); ++index) {
        posterior_[index] = likelihoods[index] * prior[index + 1];
        normalization += posterior_[index];
    }
    if (normalization <= 0.0 || !std::isfinite(normalization)) {
        throw std::runtime_error("Loop hypothesis posterior cannot be normalized");
    }
    virtual_posterior_ /= normalization;
    for (double& posterior : posterior_) {
        posterior /= normalization;
    }

    std::unordered_map<tassel_utils::FrameId, double> raw_scores;
    for (const auto& [frame_id, score] : visual_scores) {
        raw_scores[frame_id] = score;
    }
    HypothesisUpdate output;
    output.virtual_posterior = virtual_posterior_;
    output.loop_probability = 1.0 - virtual_posterior_;
    output.hypotheses.reserve(places_.size());
    for (size_t index = 0; index < places_.size(); ++index) {
        output.hypotheses.push_back(
            {places_[index], raw_scores[places_[index]], likelihoods[index], posterior_[index]});
    }
    std::sort(
        output.hypotheses.begin(), output.hypotheses.end(),
        [](const LoopHypothesis& lhs, const LoopHypothesis& rhs) {
            return lhs.posterior > rhs.posterior;
        });
    return output;
}

void LoopHypothesisTracker::reset() {
    places_.clear();
    place_indices_.clear();
    posterior_.clear();
    virtual_posterior_ = 1.0;
}

}  // namespace tassel_loop
