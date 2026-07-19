#ifndef TASSEL_CORE_FEATURE_TRACKER_H_
#define TASSEL_CORE_FEATURE_TRACKER_H_

// OpenCV
#include <cstddef>
#include <opencv2/core.hpp>

// 标准库
#include <unordered_map>

// Tassel
#include "cam/camera_factory.h"
#include "feature.h"

namespace tassel_core {
class FeatureTracker {
public:
    FeatureTracker(
        bool flow_back = false, double max_square_move_dist = 0.5, bool enable_statistics = false,
        int tracked_times_thres = 5, double min_gradient = 50.0);

    void addCamera(
        Camera camera, int per_grid_rows = 4, int per_grid_cols = 4, int grid_edge_rows = 2,
        int grid_edge_cols = 2, double mask_radius = 15.0, int min_feature_num = 100);

    std::unordered_map<int, FeaturePerFrame> monoTracking(size_t camera_id, const cv::Mat& img);

    std::unordered_map<int, FeaturePerFrame> stereoTracking(
        size_t left_camera_id, const cv::Mat& left_img, size_t right_camera_id,
        const cv::Mat& right_img);

    void reset();

    void drawTrackingResult(size_t camera_id, cv::Mat& img);

private:
    struct CameraTrackingContext {
        // 特征信息
        std::vector<cv::Point2f> prev_pts;
        std::vector<cv::Point2f> cur_pts;
        std::vector<size_t> prev_ids;
        std::vector<size_t> cur_ids;
        cv::Mat prev_img;
        cv::Mat mask;
        double mask_radius;
        size_t feature_count;
        int min_feature_num;

        // 网格管理
        int per_grid_rows, per_grid_cols;
        int grid_rows, grid_cols;
        int grid_edge_rows, grid_edge_cols;
        std::vector<bool> grid_mask;

        // 相机模型
        Camera camera;

        // 梯度缓存，每帧覆盖
        cv::Mat grad;

        // 跟踪历史
        std::vector<int> tracked_times;
    };

    inline bool isOutOfImage(cv::Point2f pt, int rows, int cols) {
        return pt.x < 0 || pt.x > cols - 1 || pt.y < 0 || pt.y > rows - 1;
    }

    inline double computeSquareDist(cv::Point2f& p1, cv::Point2f& p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return dx * dx + dy * dy;
    }

    void extractNewFeatures(size_t camera_id, const cv::Mat& img, std::vector<cv::Point2f>& pts);

    void monoMatching(
        size_t camera_id, const cv::Mat& prev_img, const cv::Mat& cur_img,
        std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& cur_pts,
        std::vector<size_t>& prev_ids, std::vector<size_t>& cur_ids);

    void setMask(size_t camera_id);

    std::unordered_map<size_t, CameraTrackingContext> ctc_map_;

    bool flow_back_;

    double max_square_move_dist_;

    bool enable_statistics_;
    int tracked_times_thres_;
    double min_gradient_thres_;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_FEATURE_TRACKER_H_
