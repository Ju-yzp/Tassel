#include "loop_closure.h"

#include <DBoW3/DBoW3.h>
#include <gtest/gtest.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace {

cv::Mat makeTexture(int seed) {
    cv::Mat image(240, 320, CV_8UC1);
    cv::RNG(seed).fill(image, cv::RNG::UNIFORM, 0, 256);
    return image;
}

std::string createVocabulary() {
    std::vector<cv::Mat> training;
    auto detector = cv::FastFeatureDetector::create(10, true);
    auto descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    for (int index = 0; index < 8; ++index) {
        const cv::Mat image = makeTexture(index + 1);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detect(image, keypoints);
        descriptor->compute(image, keypoints, descriptors);
        training.push_back(descriptors);
    }
    DBoW3::Vocabulary vocabulary(4, 3, DBoW3::TF, DBoW3::L1_NORM);
    vocabulary.create(training);
    const std::filesystem::path path =
        std::filesystem::path(testing::TempDir()) / "tassel_loop_closure_vocabulary.yml.gz";
    vocabulary.save(path.string());
    return path.string();
}

TEST(LoopClosureTest, ProcessesEstimatorTransactionsInOrderBeforeFinish) {
    tassel_loop::LoopClosureOptions options;
    options.database.fast_threshold = 10;
    options.database.max_keypoints = 300;
    options.database.recent_exclusion = 1;
    options.database.top_k = 1;
    options.database.likelihood_pool_size = 3;
    options.database.min_score = 0.0;
    std::atomic_int graph_updates{0};
    std::atomic_int global_pose_updates{0};
    tassel_loop::LoopClosure closure(
        createVocabulary(), options, [](const Eigen::Vector2d& point) { return point; },
        [&graph_updates, &global_pose_updates](const tassel_loop::LoopClosureResult& result) {
            if (result.event == tassel_loop::LoopEvent::kGraphUpdated) {
                ++graph_updates;
            } else if (result.event == tassel_loop::LoopEvent::kGlobalPoseUpdated) {
                ++global_pose_updates;
                EXPECT_EQ(result.corrected_trajectory.size(), 1u);
            }
        });

    const tassel_utils::FrameId frame_id = 100;
    closure.submitPose({frame_id, Sophus::SE3d()});
    closure.submitKeyframe({frame_id, makeTexture(9), Sophus::SE3d()});
    std::vector<tassel_loop::LandmarkInput> landmarks;
    int feature_id = 0;
    for (int y = 40; y <= 200; y += 40) {
        for (int x = 40; x <= 280; x += 40) {
            landmarks.push_back(
                {feature_id++, cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                 Eigen::Vector3d(0.1 * x, 0.1 * y, 1.0), 2.0});
        }
    }
    closure.submitLandmarks({frame_id, std::move(landmarks)});
    closure.finish();

    EXPECT_EQ(graph_updates.load(), 1);
    EXPECT_EQ(global_pose_updates.load(), 1);
    EXPECT_THROW(closure.submitPose({200, Sophus::SE3d()}), std::logic_error);
}

}  // namespace
