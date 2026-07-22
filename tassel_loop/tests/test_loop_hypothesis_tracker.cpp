#include "loop_hypothesis_tracker.h"

#include <gtest/gtest.h>

TEST(LoopHypothesisTrackerTest, RejectsInvalidKernelAndDuplicatePlaces) {
    EXPECT_THROW(tassel_loop::LoopHypothesisTracker({0.1}), std::invalid_argument);
    tassel_loop::LoopHypothesisTracker tracker;
    tracker.addPlace(100);
    EXPECT_THROW(tracker.addPlace(100), std::invalid_argument);
}

TEST(LoopHypothesisTrackerTest, ConvertsDistinctVisualScoreToLikelihoodPeak) {
    tassel_loop::LoopHypothesisTracker tracker;
    tracker.addPlace(100);
    tracker.addPlace(200);
    tracker.addPlace(300);
    const auto update = tracker.update({{100, 0.1}, {200, 0.8}, {300, 0.2}});

    ASSERT_EQ(update.hypotheses.size(), 3u);
    EXPECT_EQ(update.hypotheses.front().frame_id, 200);
    EXPECT_GT(update.hypotheses.front().likelihood, 1.0);
    EXPECT_GT(update.loop_probability, 0.0);
    EXPECT_LT(update.loop_probability, 1.0);
}

TEST(LoopHypothesisTrackerTest, TemporalPredictionSupportsNeighboringPlace) {
    tassel_loop::LoopHypothesisTracker tracker;
    for (int id = 100; id <= 500; id += 100) {
        tracker.addPlace(id);
    }
    const auto first = tracker.update({{100, 0.1}, {200, 0.2}, {300, 0.9}, {400, 0.1}});
    ASSERT_EQ(first.hypotheses.front().frame_id, 300);

    const auto second = tracker.update({{200, 0.1}, {300, 0.2}, {400, 0.9}, {500, 0.1}});
    EXPECT_EQ(second.hypotheses.front().frame_id, 400);
    EXPECT_GT(second.hypotheses.front().posterior, first.hypotheses.front().posterior * 0.5);
}

TEST(LoopHypothesisTrackerTest, ResetReturnsToVirtualPlace) {
    tassel_loop::LoopHypothesisTracker tracker;
    tracker.addPlace(100);
    tracker.update({{100, 0.9}});
    tracker.reset();
    const auto update = tracker.update({});
    EXPECT_TRUE(update.hypotheses.empty());
    EXPECT_DOUBLE_EQ(update.virtual_posterior, 1.0);
}
