#include <DBoW3/DBoW3.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: train_brief_vocabulary OUTPUT MAX_IMAGES IMAGE_DIR...\n";
        return 1;
    }

    const fs::path output_path = argv[1];
    const size_t max_images = std::stoul(argv[2]);
    if (max_images == 0) {
        std::cerr << "MAX_IMAGES must be positive\n";
        return 1;
    }

    std::vector<fs::path> image_paths;
    for (int arg_index = 3; arg_index < argc; ++arg_index) {
        for (const auto& entry : fs::recursive_directory_iterator(argv[arg_index])) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                image_paths.push_back(entry.path());
            }
        }
    }
    std::sort(image_paths.begin(), image_paths.end());
    if (image_paths.empty()) {
        std::cerr << "No training images found\n";
        return 1;
    }

    auto detector = cv::FastFeatureDetector::create(20, true);
    auto descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    std::vector<cv::Mat> training_descriptors;
    const size_t sample_count = std::min(max_images, image_paths.size());
    for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        const size_t image_index = sample_index * image_paths.size() / sample_count;
        const cv::Mat image = cv::imread(image_paths[image_index].string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            continue;
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detect(image, keypoints);
        if (keypoints.size() > 800) {
            std::sort(
                keypoints.begin(), keypoints.end(),
                [](const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) {
                    return lhs.response > rhs.response;
                });
            keypoints.resize(800);
        }
        descriptor->compute(image, keypoints, descriptors);
        if (!descriptors.empty()) {
            training_descriptors.push_back(descriptors);
        }
    }
    if (training_descriptors.empty()) {
        std::cerr << "No BRIEF descriptors extracted\n";
        return 1;
    }

    std::cout << "Training BRIEF vocabulary from " << training_descriptors.size() << " images\n";
    DBoW3::Vocabulary vocabulary(10, 4, DBoW3::TF_IDF, DBoW3::L1_NORM);
    vocabulary.create(training_descriptors);
    vocabulary.save(output_path.string(), true);
    std::cout << "Saved " << vocabulary.size() << " words to " << output_path << "\n";
    return 0;
}
