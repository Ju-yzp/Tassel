#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "frond_end/feature_tracker.h"

namespace tassel_core {

class FeatureTrackerTestAccess {
public:
    static void setCurrentFeatures(
        FeatureTracker& tracker, std::vector<cv::Point2f> points, std::vector<size_t> ids,
        std::vector<int> ages) {
        tracker.ctc_.cur_pts = std::move(points);
        tracker.ctc_.cur_ids = std::move(ids);
        tracker.ctc_.tracked_times = std::move(ages);
    }

    static void setMask(FeatureTracker& tracker) { tracker.setMask(); }

    static const std::vector<cv::Point2f>& points(const FeatureTracker& tracker) {
        return tracker.ctc_.cur_pts;
    }

    static const std::vector<size_t>& ids(const FeatureTracker& tracker) {
        return tracker.ctc_.cur_ids;
    }

    static const std::vector<int>& ages(const FeatureTracker& tracker) {
        return tracker.ctc_.tracked_times;
    }
};

namespace {

FeatureTracker makeTracker() {
    constexpr int kWidth = 200;
    constexpr int kHeight = 160;
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 100.0, 0.0, 100.0, 0.0, 100.0, 80.0, 0.0, 0.0, 1.0);
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    FeatureTracker tracker;
    tracker.setCamera(
        CameraFactory::create("radtan", camera_matrix, distortion, kWidth, kHeight), 40, 50, 0, 0,
        15.0, 20);
    return tracker;
}

TEST(FeatureTrackerTest, KeepsLongestTrackWhenCurrentFeaturesCluster) {
    FeatureTracker tracker = makeTracker();
    FeatureTrackerTestAccess::setCurrentFeatures(
        tracker, {{100.0F, 80.0F}, {106.0F, 80.0F}, {150.0F, 80.0F}}, {10, 20, 30}, {2, 8, 3});

    FeatureTrackerTestAccess::setMask(tracker);

    EXPECT_EQ(FeatureTrackerTestAccess::points(tracker).size(), 2);
    EXPECT_EQ(FeatureTrackerTestAccess::ids(tracker), (std::vector<size_t>{20, 30}));
    EXPECT_EQ(FeatureTrackerTestAccess::ages(tracker), (std::vector<int>{8, 3}));
}

TEST(FeatureTrackerTest, RejectsMismatchedCurrentFeatureState) {
    FeatureTracker tracker = makeTracker();
    FeatureTrackerTestAccess::setCurrentFeatures(tracker, {{100.0F, 80.0F}}, {10}, {});

    EXPECT_THROW(FeatureTrackerTestAccess::setMask(tracker), std::logic_error);
}

}  // namespace
}  // namespace tassel_core
