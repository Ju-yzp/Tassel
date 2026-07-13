// tassel
#include "feature_tracker.h"

// logger
#include <spdlog/spdlog.h>

// opencv
#include <opencv2/core/hal/interface.h>
#include <cmath>
#include <cstddef>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <unordered_map>
#include <utility>

namespace tassel_core {

FeatureTracker::FeatureTracker(
    bool flow_back, double max_square_move_dist, bool enable_statistics, int tracked_times_thres,
    double min_gradient)
    : flow_back_(flow_back),
      max_square_move_dist_(max_square_move_dist),
      enable_statistics_(enable_statistics),
      tracked_times_thres_(tracked_times_thres),
      min_gradient_thres_(min_gradient) {}

void FeatureTracker::addCamera(
    Camera camera, int per_grid_rows, int per_grid_cols, int grid_edge_rows, int grid_edge_cols,
    double mask_radius, int min_feature_num) {
    size_t camera_id = ctc_map_.size();
    CameraTrackingContext ctc;
    ctc.camera = std::move(camera);
    if (ctc.camera == nullptr) {
        throw std::invalid_argument(
            "Camera pointer cannot be null for camera " + std::to_string(camera_id));
    }
    ctc.per_grid_rows = per_grid_rows;
    ctc.per_grid_cols = per_grid_cols;
    ctc.grid_edge_rows = grid_edge_rows;
    ctc.grid_edge_cols = grid_edge_cols;
    ctc.grid_rows =
        (ctc.camera->get_height() - 2 * grid_edge_rows + per_grid_rows - 1) / per_grid_rows;
    ctc.grid_cols =
        (ctc.camera->get_width() - 2 * grid_edge_cols + per_grid_cols - 1) / per_grid_cols;
    ctc.mask_radius = mask_radius;
    if (ctc.grid_rows <= 0 || ctc.grid_cols <= 0) {
        throw std::invalid_argument(
            "Invalid grid parameters for camera " + std::to_string(camera_id));
    }
    if (mask_radius <= 0) {
        throw std::invalid_argument("Invalid mask radius for camera " + std::to_string(camera_id));
    }
    if (per_grid_rows <= 0 || per_grid_cols <= 0) {
        throw std::invalid_argument(
            "Invalid per-grid parameters for camera " + std::to_string(camera_id));
    }

    ctc.grid_mask.resize(ctc.grid_rows * ctc.grid_cols, false);
    ctc.feature_count = 0;
    ctc.min_feature_num = min_feature_num;
    int height = ctc.camera->get_height();
    int width = ctc.camera->get_width();
    ctc.grad = cv::Mat::zeros(height, width, CV_32F);
    ctc_map_[camera_id] = std::move(ctc);
}

std::unordered_map<int, FeaturePerFrame> FeatureTracker::monoTracking(
    size_t camera_id, const cv::Mat& img) {
    if (!ctc_map_.contains(camera_id)) {
        spdlog::error("FeatureTracker::monoTracking: unknown camera_id {}", camera_id);
        return std::unordered_map<int, FeaturePerFrame>();
    }
    CameraTrackingContext& ctc = ctc_map_[camera_id];
    cv::Mat& prev_img = ctc.prev_img;
    std::vector<cv::Point2f>& prev_pts = ctc.prev_pts;
    std::vector<cv::Point2f>& cur_pts = ctc.cur_pts;
    std::vector<size_t>& prev_ids = ctc.prev_ids;
    std::vector<size_t>& cur_ids = ctc.cur_ids;

    cur_ids = prev_ids;
    cur_pts = prev_pts;

    if (!prev_pts.empty()) {
        monoMatching(camera_id, prev_img, img, prev_pts, cur_pts, prev_ids, cur_ids);
    }

    if (img.empty() || img.type() != CV_8UC1 || img.rows != ctc.camera->get_height() ||
        img.cols != ctc.camera->get_width()) {
        spdlog::error("FeatureTracker::monoTracking: invalid image for camera_id {}", camera_id);
        return std::unordered_map<int, FeaturePerFrame>();
    }

    if (static_cast<int>(cur_pts.size()) < ctc.min_feature_num) {
        setMask(camera_id);
        std::vector<cv::Point2f> new_pts;
        extractNewFeatures(camera_id, img, new_pts);
        // spdlog::info("camera {}: tracked={} new={}", camera_id, cur_pts.size(), new_pts.size());
        for (size_t i = 0; i < new_pts.size(); ++i) {
            cur_pts.emplace_back(new_pts[i]);
            cur_ids.emplace_back(ctc.feature_count++);
            if (enable_statistics_) {
                ctc.tracked_times.emplace_back(0);
            }
        }
    }

    std::unordered_map<int, FeaturePerFrame> feature_frame;
    auto* camera = ctc.camera.get();
    for (size_t i = 0; i < cur_ids.size(); ++i) {
        Eigen::Vector2d pt(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector2d uv = camera->undistort(pt);
        FeaturePerFrame fpf;
        fpf.setLeft(uv, cur_pts[i]);
        feature_frame[cur_ids[i]] = fpf;
    }
    prev_img = img;
    std::swap(prev_pts, cur_pts);
    std::swap(prev_ids, cur_ids);
    cur_ids.clear();
    cur_pts.clear();
    return feature_frame;
}

std::unordered_map<int, FeaturePerFrame> FeatureTracker::stereoTracking(
    size_t left_camera_id, const cv::Mat& left_img, size_t right_camera_id,
    const cv::Mat& right_img) {
    auto feature_frame = monoTracking(left_camera_id, left_img);
    CameraTrackingContext& r_ctc = ctc_map_[right_camera_id];
    CameraTrackingContext& l_ctc = ctc_map_[left_camera_id];
    r_ctc.prev_img = l_ctc.prev_img;
    r_ctc.prev_pts = l_ctc.prev_pts;
    r_ctc.cur_pts = l_ctc.prev_pts;
    r_ctc.prev_ids = l_ctc.prev_ids;
    r_ctc.cur_ids = l_ctc.prev_ids;
    if (enable_statistics_) {
        r_ctc.tracked_times = l_ctc.tracked_times;
    }
    monoMatching(
        right_camera_id, r_ctc.prev_img, right_img, r_ctc.prev_pts, r_ctc.cur_pts, r_ctc.prev_ids,
        r_ctc.cur_ids);
    std::swap(r_ctc.prev_pts, r_ctc.cur_pts);
    std::swap(r_ctc.prev_ids, r_ctc.cur_ids);
    r_ctc.prev_img = right_img;
    auto* r_camera = r_ctc.camera.get();
    for (size_t i = 0; i < r_ctc.prev_ids.size(); ++i) {
        auto it = feature_frame.find(r_ctc.prev_ids[i]);
        if (it != feature_frame.end()) {
            Eigen::Vector2d pt(r_ctc.prev_pts[i].x, r_ctc.prev_pts[i].y);
            Eigen::Vector2d uv = r_camera->undistort(pt);
            FeaturePerFrame& fpf = feature_frame[r_ctc.prev_ids[i]];
            fpf.setRight(uv, r_ctc.prev_pts[i]);
        } else {
            spdlog::error("stereoTracking: feature not found");
        }
    }
    return feature_frame;
}

void FeatureTracker::reset() {
    for (auto& item : ctc_map_) {
        auto& ctc = item.second;
        ctc.prev_pts.clear();
        ctc.cur_pts.clear();
        ctc.prev_ids.clear();
        ctc.cur_ids.clear();
        ctc.prev_img = cv::Mat();
        ctc.mask = cv::Mat();
        ctc.grid_mask.assign(ctc.grid_rows * ctc.grid_cols, false);
        ctc.feature_count = 0;
        ctc.tracked_times.clear();
    }
}

void FeatureTracker::drawTrackingResult(size_t camera_id, cv::Mat& img) {
    if (img.type() == CV_8UC1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    if (!ctc_map_.contains(camera_id)) {
        spdlog::error("FeatureTracker::drawTrackingResult: unknown camera_id {}", camera_id);
        return;
    }

    CameraTrackingContext& ctc = ctc_map_[camera_id];
    std::vector<cv::Point2f>& prev_pts = ctc.prev_pts;
    std::vector<int>& tracked_times = ctc.tracked_times;

    if (enable_statistics_) {
        for (size_t i = 0; i < prev_pts.size(); ++i) {
            float ratio = std::min(tracked_times[i], tracked_times_thres_) /
                          static_cast<float>(tracked_times_thres_);
            const cv::Scalar color(255 * (1.0 - ratio), 0, 255 * ratio);
            cv::circle(img, prev_pts[i], 4, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::circle(img, prev_pts[i], 3, color, -1, cv::LINE_AA);
        }
    } else {
        for (const auto& pt : prev_pts) {
            cv::circle(img, pt, 4, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::circle(img, pt, 3, cv::Scalar(0, 220, 0), -1, cv::LINE_AA);
        }
    }
}

void FeatureTracker::monoMatching(
    size_t camera_id, const cv::Mat& prev_img, const cv::Mat& cur_img,
    std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& cur_pts,
    std::vector<size_t>& prev_ids, std::vector<size_t>& cur_ids) {
    if (!ctc_map_.contains(camera_id)) {
        spdlog::error("FeatureTracker::monoMatching: unknown camera_id {}", camera_id);
        return;
    }
    if (prev_pts.empty() || prev_ids.empty()) {
        spdlog::warn(
            "FeatureTracker::monoMatching camera {}: prev_pts({}) or prev_ids({}) is empty",
            camera_id, prev_pts.size(), prev_ids.size());
        return;
    }

    std::vector<uchar> p2c_status;
    std::vector<float> p2c_err;
    size_t num = prev_pts.size();
    cv::calcOpticalFlowPyrLK(
        prev_img, cur_img, prev_pts, cur_pts, p2c_status, p2c_err, cv::Size(21, 21), 3);

    if (flow_back_) {
        std::vector<cv::Point2f> copy_pts = prev_pts;
        std::vector<uchar> c2p_status;
        std::vector<float> c2p_err;
        cv::calcOpticalFlowPyrLK(
            cur_img, prev_img, cur_pts, copy_pts, c2p_status, c2p_err, cv::Size(21, 21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        for (size_t i = 0; i < num; ++i) {
            p2c_status[i] = p2c_status[i]
                                ? (c2p_status[i] && (computeSquareDist(prev_pts[i], copy_pts[i]) <
                                                     max_square_move_dist_))
                                : 0;
        }
    }

    int rows = ctc_map_[camera_id].camera->get_height();
    int cols = ctc_map_[camera_id].camera->get_width();
    size_t valid_count = 0;
    std::vector<int>& tracked_times = ctc_map_[camera_id].tracked_times;
    for (size_t index = 0; index < num; ++index) {
        if (p2c_status[index] && !isOutOfImage(cur_pts[index], rows, cols)) {
            cur_pts[valid_count] = cur_pts[index];
            cur_ids[valid_count] = cur_ids[index];
            if (enable_statistics_) {
                tracked_times[valid_count] = tracked_times[index] + 1;
            }
            ++valid_count;
        }
    }

    cur_pts.resize(valid_count);
    cur_ids.resize(valid_count);
    if (enable_statistics_) {
        tracked_times.resize(valid_count);
    }
}

void FeatureTracker::setMask(size_t camera_id) {
    if (!ctc_map_.contains(camera_id)) {
        spdlog::error("FeatureTracker::setMask: unknown camera_id {}", camera_id);
        return;
    }
    CameraTrackingContext& ctc = ctc_map_[camera_id];
    cv::Mat& mask = ctc.mask;
    int rows = ctc.camera->get_height();
    int cols = ctc.camera->get_width();
    mask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    ctc.grid_mask.assign(ctc.grid_rows * ctc.grid_cols, false);
    const std::vector<cv::Point2f>& cur_pts = ctc.cur_pts;
    const double mask_radius = ctc.mask_radius;
    const int grid_edge_rows = ctc.grid_edge_rows;
    const int grid_edge_cols = ctc.grid_edge_cols;
    for (auto& pt : cur_pts) {
        cv::circle(mask, pt, mask_radius, cv::Scalar(0), -1);
        int y = pt.y - grid_edge_rows;
        int x = pt.x - grid_edge_cols;
        if (y < rows - grid_edge_rows && x < cols - grid_edge_cols) {
            int id = x / ctc.per_grid_cols + (y / ctc.per_grid_rows) * ctc.grid_cols;
            if (id < static_cast<int>(ctc.grid_mask.size()) && id >= 0) {
                ctc.grid_mask[id] = true;
            }
        }
    }
}

void FeatureTracker::extractNewFeatures(
    size_t camera_id, const cv::Mat& img, std::vector<cv::Point2f>& new_pts) {
    CameraTrackingContext& ctc = ctc_map_[camera_id];
    cv::Mat grad_x, grad_y;
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 3);

    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(grad_x, grad_x, Ix2);
    cv::multiply(grad_y, grad_y, Iy2);
    cv::multiply(grad_x, grad_y, Ixy);

    int block_size = 3;
    cv::boxFilter(Ix2, Ix2, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(Iy2, Iy2, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(Ixy, Ixy, CV_32F, cv::Size(block_size, block_size));

    cv::Mat diff_sq, ixy_sq;
    cv::pow(Ix2 - Iy2, 2, diff_sq);
    cv::pow(Ixy * 2, 2, ixy_sq);
    cv::Mat term;
    cv::sqrt(diff_sq + ixy_sq, term);
    ctc.grad = (Ix2 + Iy2 - term) * 0.5f;

    const int rows = ctc.camera->get_height();
    const int cols = ctc.camera->get_width();
    const int grid_rows = ctc.grid_rows;
    const int grid_cols = ctc.grid_cols;
    const int edge_y = ctc.grid_edge_rows;
    const int edge_x = ctc.grid_edge_cols;
    const int cell_h = ctc.per_grid_rows;
    const int cell_w = ctc.per_grid_cols;
    const std::vector<bool>& grid_mask = ctc.grid_mask;

    const size_t ncells = grid_rows * grid_cols;
    std::vector<float> best_scores(ncells, min_gradient_thres_);
    std::vector<cv::Point2f> best_pts(ncells, cv::Point2f(-1, -1));

    const int y0 = edge_y, y1 = rows - edge_y;
    const int x0 = edge_x, x1 = cols - edge_x;

    for (int y = y0; y < y1; ++y) {
        const uchar* mask_row = ctc.mask.ptr<uchar>(y);
        const float* grad_row = ctc.grad.ptr<float>(y);
        for (int x = x0; x < x1; ++x) {
            if (mask_row[x] == 0) continue;
            int cell_r = (y - y0) / cell_h;
            int cell_c = (x - x0) / cell_w;
            int idx = cell_r * grid_cols + cell_c;
            if (grid_mask[idx]) continue;
            float s = grad_row[x];
            if (s > best_scores[idx]) {
                best_scores[idx] = s;
                best_pts[idx] = cv::Point2f(x, y);
            }
        }
    }
    for (size_t i = 0; i < ncells; ++i) {
        if (best_pts[i].x != -1) new_pts.emplace_back(best_pts[i]);
    }
}

}  // namespace tassel_core
