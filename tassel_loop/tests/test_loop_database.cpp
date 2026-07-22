#include "loop_database.h"

#include <DBoW3/DBoW3.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

cv::Mat makeTexture(int seed) {
    cv::Mat image(240, 320, CV_8UC1);
    cv::RNG rng(seed);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    return image;
}

std::string createVocabulary() {
    std::vector<cv::Mat> training;
    auto detector = cv::FastFeatureDetector::create(10, true);
    auto descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    for (int image_index = 0; image_index < 12; ++image_index) {
        const cv::Mat image = makeTexture(image_index + 1);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detect(image, keypoints);
        descriptor->compute(image, keypoints, descriptors);
        training.push_back(descriptors);
    }

    DBoW3::Vocabulary vocabulary(4, 3, DBoW3::TF, DBoW3::L1_NORM);
    vocabulary.create(training);
    const std::filesystem::path path =
        std::filesystem::path(testing::TempDir()) / "tassel_test_brief_vocabulary.yml.gz";
    vocabulary.save(path.string());
    return path.string();
}

class LoopDatabaseTest : public testing::Test {
protected:
    void SetUp() override {
        tassel_loop::LoopOptions options;
        options.fast_threshold = 10;
        options.max_keypoints = 300;
        options.recent_exclusion = 1;
        options.top_k = 1;
        options.likelihood_pool_size = 3;
        options.min_score = 0.0;
        database_ = std::make_unique<tassel_loop::LoopDatabase>(createVocabulary(), options);
    }

    std::unique_ptr<tassel_loop::LoopDatabase> database_;
};

TEST(LoopDatabaseOptionsTest, RejectsInvalidOptions) {
    tassel_loop::LoopOptions options;
    options.top_k = 0;
    EXPECT_THROW(tassel_loop::LoopDatabase(createVocabulary(), options), std::invalid_argument);
}

TEST_F(LoopDatabaseTest, RejectsEmptyAndNonGrayImages) {
    EXPECT_THROW(database_->addKeyframe(1, {}), std::invalid_argument);
    EXPECT_THROW(database_->addKeyframe(1, cv::Mat(10, 10, CV_8UC3)), std::invalid_argument);
}

TEST_F(LoopDatabaseTest, RejectsDuplicateFrameIds) {
    EXPECT_FALSE(database_->contains(10));
    database_->addKeyframe(10, makeTexture(1));
    EXPECT_TRUE(database_->contains(10));
    EXPECT_THROW(database_->addKeyframe(10, makeTexture(1)), std::invalid_argument);
}

TEST_F(LoopDatabaseTest, ExcludesRecentFrameAndMapsEntryToFrameId) {
    const cv::Mat image = makeTexture(7);
    const auto first = database_->addKeyframe(100, image);
    const auto second = database_->addKeyframe(200, image);
    const auto third = database_->addKeyframe(300, image);

    EXPECT_GT(first.keypoint_count, 0);
    EXPECT_TRUE(first.candidates.empty());
    EXPECT_TRUE(second.candidates.empty());
    ASSERT_EQ(third.candidates.size(), 1U);
    EXPECT_EQ(third.candidates.front().frame_id, 100);
    EXPECT_GT(third.candidates.front().score, 0.0);
    EXPECT_EQ(database_->size(), 3U);
    EXPECT_GT(third.loop_probability, 0.0);
    EXPECT_FALSE(third.hypotheses.empty());
    ASSERT_FALSE(third.verification_candidates.empty());
    EXPECT_EQ(third.verification_candidates.front().frame_id, third.hypotheses.front().frame_id);
    EXPECT_GT(third.verification_candidates.front().posterior, 0.0);
}

TEST_F(LoopDatabaseTest, DrawsExplicitCandidateMatches) {
    const cv::Mat image = makeTexture(9);
    database_->addKeyframe(100, image);
    database_->addKeyframe(200, image);
    database_->addKeyframe(300, image);

    const cv::Mat matches = database_->drawCandidate(300, 100);
    EXPECT_FALSE(matches.empty());
    EXPECT_EQ(matches.type(), CV_8UC3);
    EXPECT_EQ(matches.rows, image.rows);
    EXPECT_EQ(matches.cols, image.cols * 2);
}

TEST_F(LoopDatabaseTest, AttachesDelayedLandmarksAndPreservesDescriptorAlignment) {
    const cv::Mat image = makeTexture(11);
    database_->addKeyframe(100, image);
    database_->addKeyframe(200, image);

    std::vector<tassel_loop::LandmarkInput> landmarks;
    int landmark_index = 0;
    for (int y = 40; y <= 200; y += 40) {
        for (int x = 40; x <= 280; x += 40) {
            landmarks.push_back(
                {landmark_index, cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                 Eigen::Vector3d(landmark_index, x, y), 2.0});
            ++landmark_index;
        }
    }
    landmarks.push_back({landmark_index, cv::Point2f(1.0f, 1.0f), Eigen::Vector3d::Ones(), 0.0});
    const size_t stored_count = database_->attachLandmarks(100, landmarks);
    EXPECT_GT(stored_count, 0u);
    EXPECT_LT(stored_count, landmarks.size());

    database_->addKeyframe(300, image);
    const tassel_loop::PnpMatches matches = database_->matchCandidateLandmarks(300, 100);
    EXPECT_EQ(matches.candidate_frame_id, 100);
    ASSERT_FALSE(matches.current_points.empty());
    ASSERT_EQ(matches.current_points.size(), matches.host_points.size());
    for (const Eigen::Vector3d& host_point : matches.host_points) {
        const int source_index = static_cast<int>(host_point.x() / 2.0);
        ASSERT_GE(source_index, 0);
        ASSERT_LT(source_index, static_cast<int>(landmarks.size()));
        EXPECT_TRUE(host_point.isApprox(
            landmarks[source_index].host_uv * landmarks[source_index].host_depth));
    }
}

TEST_F(LoopDatabaseTest, RejectsLandmarksForUnknownOrAlreadyAttachedKeyframe) {
    const std::vector<tassel_loop::LandmarkInput> landmarks = {
        {1, cv::Point2f(80.0f, 80.0f), Eigen::Vector3d::Ones(), 2.0}};
    EXPECT_THROW(database_->attachLandmarks(100, landmarks), std::invalid_argument);
    database_->addKeyframe(100, makeTexture(3));
    EXPECT_EQ(database_->attachLandmarks(100, landmarks), 1u);
    EXPECT_THROW(database_->attachLandmarks(100, landmarks), std::invalid_argument);
}

}  // namespace
