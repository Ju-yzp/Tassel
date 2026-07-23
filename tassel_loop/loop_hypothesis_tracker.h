#ifndef TASSEL_LOOP_LOOP_HYPOTHESIS_TRACKER_H_
#define TASSEL_LOOP_LOOP_HYPOTHESIS_TRACKER_H_

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loop_types.h"

namespace tassel_loop {

struct LoopHypothesis {
    KeyframeId frame_id = kInvalidKeyframeId;
    double raw_score = 0.0;
    double likelihood = 1.0;
    double posterior = 0.0;
};

struct HypothesisUpdate {
    double virtual_posterior = 1.0;
    double loop_probability = 0.0;
    std::vector<LoopHypothesis> hypotheses;
};

class LoopHypothesisTracker {
public:
    explicit LoopHypothesisTracker(
        std::vector<double> prediction = {0.1, 0.36, 0.30, 0.16, 0.062, 0.0151, 0.00255, 0.000324});

    void addPlace(KeyframeId frame_id);
    HypothesisUpdate update(const std::vector<std::pair<KeyframeId, double>>& visual_scores);
    void reset();

private:
    std::vector<double> adjustedLikelihoods(
        const std::vector<std::pair<KeyframeId, double>>& visual_scores,
        double& virtual_likelihood) const;
    std::vector<double> predictPrior() const;

    std::vector<double> prediction_;
    std::vector<KeyframeId> places_;
    std::unordered_map<KeyframeId, size_t> place_indices_;
    std::vector<double> posterior_;
    double virtual_posterior_ = 1.0;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_LOOP_HYPOTHESIS_TRACKER_H_
