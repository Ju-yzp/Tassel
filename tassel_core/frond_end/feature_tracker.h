#ifndef TASSEL_CORE_FEATURE_TRACKER_H_
#define TASSEL_CORE_FEATURE_TRACKER_H_

// OpenCV
#include <cstddef>
#include <limits>
#include <opencv2/core.hpp>

// 标准库
#include <unordered_map>

// Tassel
#include "cam/camera_factory.h"
#include "feature.h"

namespace tassel_core {
class FeatureTrackerTestAccess;

class FeatureTracker {
public:
    FeatureTracker(
        bool flow_back = false, double max_square_move_dist = 0.5, bool enable_statistics = false,
        int track_age_color_scale = 5, double min_gradient = 50.0);

    void setCamera(
        Camera camera, int per_grid_rows = 4, int per_grid_cols = 4, double mask_radius = 15.0,
        int min_feature_num = 150);

    void setValidMask(const cv::Mat& mask, int margin = 0);

    std::unordered_map<int, FeaturePerFrame> monoTracking(
        const cv::Mat& img, const std::unordered_map<int, cv::Point2f>& predicted_pixels = {});

    void reset();

    void drawTrackingResult(cv::Mat& img);

private:
    friend class FeatureTrackerTestAccess;

    struct CameraTrackingContext {
        // 特征信息
        std::vector<cv::Point2f> prev_pts;
        std::vector<cv::Point2f> cur_pts;
        std::vector<size_t> prev_ids;
        std::vector<size_t> cur_ids;
        cv::Mat prev_img;
        cv::Mat mask;
        cv::Mat valid_mask;
        double mask_radius;
        size_t feature_count;
        int min_feature_num;

        // 网格管理
        int per_grid_rows, per_grid_cols;
        int grid_rows, grid_cols;
        std::vector<bool> grid_mask;

        // 相机模型
        Camera camera;

        // 梯度缓存，每帧覆盖
        cv::Mat grad;

        // 跟踪历史
        std::vector<int> tracked_times;
    };

    struct TimingRange {
        double min_ms = std::numeric_limits<double>::infinity();
        double max_ms = 0.0;
        double sum_ms = 0.0;

        void add(double ms);
        double avg(size_t count) const;
        void reset();
    };

    struct TimingStats {
        size_t count = 0;
        TimingRange total;
        TimingRange match;
        TimingRange mask;
        TimingRange extract;
        TimingRange pack;
        TimingRange lk_forward;
        TimingRange lk_backward;
        TimingRange match_filter;
        TimingRange gradient;
        TimingRange tensor;
        TimingRange response;
        TimingRange grid_search;
        size_t tracked_sum = 0;
        size_t new_sum = 0;

        void add(
            double total_ms, double match_ms, double mask_ms, double extract_ms, double pack_ms,
            size_t tracked_count, size_t new_count);
        void reset();
    };

    struct MatchTiming {
        double forward_ms = 0.0;
        double backward_ms = 0.0;
        double filter_ms = 0.0;
    };

    struct ExtractTiming {
        double gradient_ms = 0.0;
        double tensor_ms = 0.0;
        double response_ms = 0.0;
        double grid_search_ms = 0.0;
    };

    inline bool isOutOfImage(cv::Point2f pt, int rows, int cols) {
        return pt.x < 0 || pt.x > cols - 1 || pt.y < 0 || pt.y > rows - 1;
    }

    inline double computeSquareDist(cv::Point2f& p1, cv::Point2f& p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return dx * dx + dy * dy;
    }

    void extractNewFeatures(
        const cv::Mat& img, std::vector<cv::Point2f>& pts, ExtractTiming& timing);

    void monoMatching(
        const cv::Mat& prev_img, const cv::Mat& cur_img, std::vector<cv::Point2f>& prev_pts,
        std::vector<cv::Point2f>& cur_pts, std::vector<size_t>& prev_ids,
        std::vector<size_t>& cur_ids, const std::unordered_map<int, cv::Point2f>& predicted_pixels,
        MatchTiming& timing);

    void setMask();

    CameraTrackingContext ctc_;
    TimingStats timing_stats_;

    bool flow_back_;

    double max_square_move_dist_;

    bool enable_statistics_;
    int track_age_color_scale_;
    double min_gradient_thres_;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FEATURE_TRACKER_H_
