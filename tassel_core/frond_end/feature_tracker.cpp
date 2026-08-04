// Tassel
#include "feature_tracker.h"

// 日志
#include <spdlog/spdlog.h>

// OpenCV
#include <opencv2/core/hal/interface.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <unordered_map>
#include <utility>

namespace tassel_core {

namespace {
using SteadyClock = std::chrono::steady_clock;
constexpr size_t kTimingReportPeriod = 30;

double elapsedMs(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}
}  // namespace

void FeatureTracker::TimingRange::add(double ms) {
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
    sum_ms += ms;
}

double FeatureTracker::TimingRange::avg(size_t count) const {
    return count > 0 ? sum_ms / static_cast<double>(count) : 0.0;
}

void FeatureTracker::TimingRange::reset() {
    min_ms = std::numeric_limits<double>::infinity();
    max_ms = 0.0;
    sum_ms = 0.0;
}

void FeatureTracker::TimingStats::add(
    double total_ms, double match_ms, double mask_ms, double extract_ms, double pack_ms,
    size_t tracked_count, size_t new_count) {
    ++count;
    total.add(total_ms);
    match.add(match_ms);
    mask.add(mask_ms);
    extract.add(extract_ms);
    pack.add(pack_ms);
    tracked_sum += tracked_count;
    new_sum += new_count;
}

void FeatureTracker::TimingStats::reset() {
    count = 0;
    total.reset();
    match.reset();
    mask.reset();
    extract.reset();
    pack.reset();
    lk_forward.reset();
    lk_backward.reset();
    match_filter.reset();
    gradient.reset();
    tensor.reset();
    response.reset();
    grid_search.reset();
    tracked_sum = 0;
    new_sum = 0;
}

FeatureTracker::FeatureTracker(
    bool flow_back, double max_square_move_dist, bool enable_statistics, int track_age_color_scale,
    double min_gradient)
    : flow_back_(flow_back),
      max_square_move_dist_(max_square_move_dist),
      enable_statistics_(enable_statistics),
      track_age_color_scale_(track_age_color_scale),
      min_gradient_thres_(min_gradient) {
    if (max_square_move_dist_ < 0.0 || track_age_color_scale_ <= 0 || min_gradient_thres_ < 0.0) {
        throw std::invalid_argument("Invalid FeatureTracker configuration");
    }
}

void FeatureTracker::setCamera(
    Camera camera, int per_grid_rows, int per_grid_cols, double mask_radius, int min_feature_num) {
    CameraTrackingContext ctc;
    ctc.camera = std::move(camera);
    if (ctc.camera == nullptr) {
        throw std::invalid_argument("FeatureTracker camera cannot be null");
    }
    if (per_grid_rows <= 0 || per_grid_cols <= 0) {
        throw std::invalid_argument("Invalid FeatureTracker per-grid parameters");
    }
    if (mask_radius <= 0 || min_feature_num < 0) {
        throw std::invalid_argument("Invalid FeatureTracker extraction parameters");
    }
    ctc.per_grid_rows = per_grid_rows;
    ctc.per_grid_cols = per_grid_cols;
    ctc.grid_rows = (ctc.camera->get_height() + per_grid_rows - 1) / per_grid_rows;
    ctc.grid_cols = (ctc.camera->get_width() + per_grid_cols - 1) / per_grid_cols;
    ctc.mask_radius = mask_radius;
    if (ctc.grid_rows <= 0 || ctc.grid_cols <= 0) {
        throw std::invalid_argument("Invalid FeatureTracker grid dimensions");
    }
    ctc.grid_mask.resize(ctc.grid_rows * ctc.grid_cols, false);
    ctc.feature_count = 0;
    ctc.min_feature_num = min_feature_num;
    int height = ctc.camera->get_height();
    int width = ctc.camera->get_width();
    ctc.grad = cv::Mat::zeros(height, width, CV_32F);
    ctc.valid_mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
    ctc_ = std::move(ctc);
}

void FeatureTracker::setValidMask(const cv::Mat& mask, int margin) {
    if (!ctc_.camera) {
        throw std::logic_error("FeatureTracker camera has not been configured");
    }
    if (mask.empty() || mask.type() != CV_8UC1 || mask.rows != ctc_.camera->get_height() ||
        mask.cols != ctc_.camera->get_width()) {
        throw std::invalid_argument("FeatureTracker valid mask does not match the camera image");
    }
    if (margin < 0) {
        throw std::invalid_argument("FeatureTracker valid mask margin must not be negative");
    }
    if (margin == 0) {
        ctc_.valid_mask = mask.clone();
        return;
    }
    // 特征中心必须离无效区至少 margin 像素，避免 LK patch 和梯度窗口跨越硬掩码边界。
    const int kernel_size = 2 * margin + 1;
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::erode(mask, ctc_.valid_mask, kernel);
}

std::unordered_map<int, FeaturePerFrame> FeatureTracker::monoTracking(
    const cv::Mat& img, const std::unordered_map<int, cv::Point2f>& predicted_pixels) {
    const auto total_start = SteadyClock::now();
    if (!ctc_.camera) {
        throw std::logic_error("FeatureTracker camera has not been configured");
    }
    CameraTrackingContext& ctc = ctc_;
    if (img.empty() || img.type() != CV_8UC1 || img.rows != ctc.camera->get_height() ||
        img.cols != ctc.camera->get_width()) {
        spdlog::error("FeatureTracker::monoTracking: invalid image");
        return std::unordered_map<int, FeaturePerFrame>();
    }
    cv::Mat& prev_img = ctc.prev_img;
    std::vector<cv::Point2f>& prev_pts = ctc.prev_pts;
    std::vector<cv::Point2f>& cur_pts = ctc.cur_pts;
    std::vector<size_t>& prev_ids = ctc.prev_ids;
    std::vector<size_t>& cur_ids = ctc.cur_ids;

    cur_ids = prev_ids;
    cur_pts = prev_pts;

    double match_ms = 0.0;
    MatchTiming match_timing;
    if (!prev_pts.empty()) {
        const auto match_start = SteadyClock::now();
        monoMatching(
            prev_img, img, prev_pts, cur_pts, prev_ids, cur_ids, predicted_pixels, match_timing);
        match_ms = elapsedMs(match_start);
    }

    auto stage_start = SteadyClock::now();
    setMask();
    const double mask_ms = elapsedMs(stage_start);
    size_t new_count = 0;
    double extract_ms = 0.0;
    ExtractTiming extract_timing;
    if (prev_pts.empty() || static_cast<int>(cur_pts.size()) < ctc.min_feature_num) {
        std::vector<cv::Point2f> new_pts;
        stage_start = SteadyClock::now();
        extractNewFeatures(img, new_pts, extract_timing);
        extract_ms = elapsedMs(stage_start);
        new_count = new_pts.size();
        for (size_t i = 0; i < new_pts.size(); ++i) {
            cur_pts.emplace_back(new_pts[i]);
            cur_ids.emplace_back(ctc.feature_count++);
            ctc.tracked_times.emplace_back(1);
        }
    }

    stage_start = SteadyClock::now();
    std::unordered_map<int, FeaturePerFrame> feature_frame;
    auto* camera = ctc.camera.get();
    for (size_t i = 0; i < cur_ids.size(); ++i) {
        Eigen::Vector2d pt(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector2d uv = camera->undistort(pt);
        FeaturePerFrame fpf;
        fpf.setObservation(uv, cur_pts[i]);
        feature_frame[cur_ids[i]] = fpf;
    }
    const double pack_ms = elapsedMs(stage_start);
    const size_t tracked_count = cur_pts.size();
    prev_img = img;
    std::swap(prev_pts, cur_pts);
    std::swap(prev_ids, cur_ids);
    cur_ids.clear();
    cur_pts.clear();
    timing_stats_.add(
        elapsedMs(total_start), match_ms, mask_ms, extract_ms, pack_ms, tracked_count, new_count);
    timing_stats_.lk_forward.add(match_timing.forward_ms);
    timing_stats_.lk_backward.add(match_timing.backward_ms);
    timing_stats_.match_filter.add(match_timing.filter_ms);
    timing_stats_.gradient.add(extract_timing.gradient_ms);
    timing_stats_.tensor.add(extract_timing.tensor_ms);
    timing_stats_.response.add(extract_timing.response_ms);
    timing_stats_.grid_search.add(extract_timing.grid_search_ms);
    if (timing_stats_.count >= kTimingReportPeriod) {
        const auto& stats = timing_stats_;
        spdlog::info(
            "FeatureTracker timing range frames={} total={:.3f}/{:.3f}/{:.3f} "
            "match={:.3f}/{:.3f}/{:.3f} mask={:.3f}/{:.3f}/{:.3f} "
            "extract={:.3f}/{:.3f}/{:.3f} pack={:.3f}/{:.3f}/{:.3f} "
            "tracked_avg={:.1f} new_avg={:.1f}",
            stats.count, stats.total.min_ms, stats.total.avg(stats.count), stats.total.max_ms,
            stats.match.min_ms, stats.match.avg(stats.count), stats.match.max_ms, stats.mask.min_ms,
            stats.mask.avg(stats.count), stats.mask.max_ms, stats.extract.min_ms,
            stats.extract.avg(stats.count), stats.extract.max_ms, stats.pack.min_ms,
            stats.pack.avg(stats.count), stats.pack.max_ms,
            static_cast<double>(stats.tracked_sum) / static_cast<double>(stats.count),
            static_cast<double>(stats.new_sum) / static_cast<double>(stats.count));
        spdlog::info(
            "OpenCV timing range frames={} lk_fwd={:.3f}/{:.3f}/{:.3f} "
            "lk_back={:.3f}/{:.3f}/{:.3f} filter={:.3f}/{:.3f}/{:.3f} "
            "gradient={:.3f}/{:.3f}/{:.3f} tensor={:.3f}/{:.3f}/{:.3f} "
            "response={:.3f}/{:.3f}/{:.3f} grid={:.3f}/{:.3f}/{:.3f}",
            stats.count, stats.lk_forward.min_ms, stats.lk_forward.avg(stats.count),
            stats.lk_forward.max_ms, stats.lk_backward.min_ms, stats.lk_backward.avg(stats.count),
            stats.lk_backward.max_ms, stats.match_filter.min_ms,
            stats.match_filter.avg(stats.count), stats.match_filter.max_ms, stats.gradient.min_ms,
            stats.gradient.avg(stats.count), stats.gradient.max_ms, stats.tensor.min_ms,
            stats.tensor.avg(stats.count), stats.tensor.max_ms, stats.response.min_ms,
            stats.response.avg(stats.count), stats.response.max_ms, stats.grid_search.min_ms,
            stats.grid_search.avg(stats.count), stats.grid_search.max_ms);
        timing_stats_.reset();
    }
    return feature_frame;
}

void FeatureTracker::reset() {
    ctc_.prev_pts.clear();
    ctc_.cur_pts.clear();
    ctc_.prev_ids.clear();
    ctc_.cur_ids.clear();
    ctc_.prev_img = cv::Mat();
    ctc_.mask = cv::Mat();
    ctc_.grid_mask.assign(ctc_.grid_rows * ctc_.grid_cols, false);
    ctc_.feature_count = 0;
    ctc_.tracked_times.clear();
    timing_stats_.reset();
}

void FeatureTracker::drawTrackingResult(cv::Mat& img) {
    if (img.type() == CV_8UC1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    CameraTrackingContext& ctc = ctc_;
    std::vector<cv::Point2f>& prev_pts = ctc.prev_pts;
    std::vector<int>& tracked_times = ctc.tracked_times;

    if (enable_statistics_) {
        for (size_t i = 0; i < prev_pts.size(); ++i) {
            float ratio = std::min(tracked_times[i], track_age_color_scale_) /
                          static_cast<float>(track_age_color_scale_);
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
    const cv::Mat& prev_img, const cv::Mat& cur_img, std::vector<cv::Point2f>& prev_pts,
    std::vector<cv::Point2f>& cur_pts, std::vector<size_t>& prev_ids, std::vector<size_t>& cur_ids,
    const std::unordered_map<int, cv::Point2f>& predicted_pixels, MatchTiming& timing) {
    if (prev_pts.empty() || prev_ids.empty()) {
        spdlog::info(
            "FeatureTracker::monoMatching: prev_pts({}) or prev_ids({}) is empty", prev_pts.size(),
            prev_ids.size());
        return;
    }

    std::vector<uchar> p2c_status;
    std::vector<float> p2c_err;
    size_t num = prev_pts.size();
    bool has_initial_flow = false;
    const int rows = ctc_.camera->get_height();
    const int cols = ctc_.camera->get_width();
    for (size_t i = 0; i < num; ++i) {
        const auto prediction = predicted_pixels.find(static_cast<int>(prev_ids[i]));
        if (prediction == predicted_pixels.end() || isOutOfImage(prediction->second, rows, cols)) {
            continue;
        }
        // 预测必须按上一帧仍存活的 feature ID 对齐，不能依赖容器或滑窗索引顺序。
        cur_pts[i] = prediction->second;
        has_initial_flow = true;
    }
    auto stage_start = SteadyClock::now();
    cv::calcOpticalFlowPyrLK(
        prev_img, cur_img, prev_pts, cur_pts, p2c_status, p2c_err, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        has_initial_flow ? cv::OPTFLOW_USE_INITIAL_FLOW : 0);
    timing.forward_ms = elapsedMs(stage_start);

    if (flow_back_) {
        std::vector<cv::Point2f> copy_pts = prev_pts;
        std::vector<uchar> c2p_status;
        std::vector<float> c2p_err;
        stage_start = SteadyClock::now();
        cv::calcOpticalFlowPyrLK(
            cur_img, prev_img, cur_pts, copy_pts, c2p_status, c2p_err, cv::Size(21, 21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);
        timing.backward_ms = elapsedMs(stage_start);

        for (size_t i = 0; i < num; ++i) {
            p2c_status[i] = p2c_status[i]
                                ? (c2p_status[i] && (computeSquareDist(prev_pts[i], copy_pts[i]) <
                                                     max_square_move_dist_))
                                : 0;
        }
    }

    stage_start = SteadyClock::now();
    size_t valid_count = 0;
    std::vector<int>& tracked_times = ctc_.tracked_times;
    for (size_t index = 0; index < num; ++index) {
        const int x = cvRound(cur_pts[index].x);
        const int y = cvRound(cur_pts[index].y);
        if (p2c_status[index] && !isOutOfImage(cur_pts[index], rows, cols) && x >= 0 && x < cols &&
            y >= 0 && y < rows && ctc_.valid_mask.at<uchar>(y, x) != 0) {
            cur_pts[valid_count] = cur_pts[index];
            cur_ids[valid_count] = cur_ids[index];
            tracked_times[valid_count] = tracked_times[index] + 1;
            ++valid_count;
        }
    }

    cur_pts.resize(valid_count);
    cur_ids.resize(valid_count);
    tracked_times.resize(valid_count);
    timing.filter_ms = elapsedMs(stage_start);
}

void FeatureTracker::setMask() {
    CameraTrackingContext& ctc = ctc_;
    cv::Mat& mask = ctc.mask;
    int rows = ctc.camera->get_height();
    int cols = ctc.camera->get_width();
    mask = ctc.valid_mask.clone();
    ctc.grid_mask.assign(ctc.grid_rows * ctc.grid_cols, false);
    if (ctc.cur_pts.size() != ctc.cur_ids.size() ||
        ctc.cur_pts.size() != ctc.tracked_times.size()) {
        throw std::logic_error("Current feature points, IDs, and track ages are inconsistent");
    }

    std::vector<size_t> order(ctc.cur_pts.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return ctc.tracked_times[lhs] > ctc.tracked_times[rhs];
    });

    std::vector<cv::Point2f> distributed_pts;
    std::vector<size_t> distributed_ids;
    std::vector<int> distributed_ages;
    distributed_pts.reserve(ctc.cur_pts.size());
    distributed_ids.reserve(ctc.cur_ids.size());
    distributed_ages.reserve(ctc.tracked_times.size());
    const double mask_radius = ctc.mask_radius;
    for (size_t index : order) {
        const cv::Point2f& pt = ctc.cur_pts[index];
        const int pixel_x = cvRound(pt.x);
        const int pixel_y = cvRound(pt.y);
        if (pixel_x < 0 || pixel_x >= cols || pixel_y < 0 || pixel_y >= rows) {
            throw std::logic_error("Current feature lies outside the camera image");
        }
        if (mask.at<uchar>(pixel_y, pixel_x) == 0) {
            continue;
        }
        distributed_pts.push_back(pt);
        distributed_ids.push_back(ctc.cur_ids[index]);
        distributed_ages.push_back(ctc.tracked_times[index]);
        cv::circle(mask, pt, mask_radius, cv::Scalar(0), -1);
        int y = cvRound(pt.y);
        int x = cvRound(pt.x);
        if (y >= 0 && x >= 0 && y < rows && x < cols) {
            int id = x / ctc.per_grid_cols + (y / ctc.per_grid_rows) * ctc.grid_cols;
            if (id < static_cast<int>(ctc.grid_mask.size()) && id >= 0) {
                ctc.grid_mask[id] = true;
            }
        }
    }
    ctc.cur_pts = std::move(distributed_pts);
    ctc.cur_ids = std::move(distributed_ids);
    ctc.tracked_times = std::move(distributed_ages);
}

void FeatureTracker::extractNewFeatures(
    const cv::Mat& img, std::vector<cv::Point2f>& new_pts, ExtractTiming& timing) {
    CameraTrackingContext& ctc = ctc_;
    const int rows = ctc.camera->get_height();
    const int cols = ctc.camera->get_width();
    const int grid_rows = ctc.grid_rows;
    const int grid_cols = ctc.grid_cols;
    const int cell_h = ctc.per_grid_rows;
    const int cell_w = ctc.per_grid_cols;
    const std::vector<bool>& grid_mask = ctc.grid_mask;

    const size_t ncells = grid_rows * grid_cols;
    if (grid_mask.size() != ncells) {
        throw std::logic_error("FeatureTracker grid mask size does not match grid dimensions");
    }
    ctc.grad.setTo(0);
    std::vector<float> best_scores(ncells, min_gradient_thres_);
    std::vector<cv::Point2f> best_pts(ncells, cv::Point2f(-1, -1));

    const int y0 = 0, y1 = rows;
    const int x0 = 0, x1 = cols;
    constexpr int kBlockSize = 3;
    constexpr int kRoiPadding = 1;

    auto stage_start = SteadyClock::now();
    const auto grid_start = SteadyClock::now();
    for (int cell_r = 0; cell_r < grid_rows; ++cell_r) {
        const int cell_y0 = y0 + cell_r * cell_h;
        const int cell_y1 = std::min(cell_y0 + cell_h, y1);
        if (cell_y0 >= cell_y1) {
            continue;
        }
        for (int cell_c = 0; cell_c < grid_cols; ++cell_c) {
            const int idx = cell_r * grid_cols + cell_c;
            if (grid_mask[idx]) {
                continue;
            }
            const int cell_x0 = x0 + cell_c * cell_w;
            const int cell_x1 = std::min(cell_x0 + cell_w, x1);
            if (cell_x0 >= cell_x1) {
                continue;
            }
            const cv::Rect roi(cell_x0, cell_y0, cell_x1 - cell_x0, cell_y1 - cell_y0);
            const int padded_x0 = std::max(0, roi.x - kRoiPadding);
            const int padded_y0 = std::max(0, roi.y - kRoiPadding);
            const int padded_x1 = std::min(cols, roi.x + roi.width + kRoiPadding);
            const int padded_y1 = std::min(rows, roi.y + roi.height + kRoiPadding);
            const cv::Rect padded_roi(
                padded_x0, padded_y0, padded_x1 - padded_x0, padded_y1 - padded_y0);

            stage_start = SteadyClock::now();
            cv::Mat grad_x, grad_y;
            cv::Sobel(img(padded_roi), grad_x, CV_32F, 1, 0, 3);
            cv::Sobel(img(padded_roi), grad_y, CV_32F, 0, 1, 3);
            timing.gradient_ms += elapsedMs(stage_start);

            stage_start = SteadyClock::now();
            cv::Mat ix2, iy2, ixy;
            cv::multiply(grad_x, grad_x, ix2);
            cv::multiply(grad_y, grad_y, iy2);
            cv::multiply(grad_x, grad_y, ixy);
            cv::boxFilter(ix2, ix2, CV_32F, cv::Size(kBlockSize, kBlockSize));
            cv::boxFilter(iy2, iy2, CV_32F, cv::Size(kBlockSize, kBlockSize));
            cv::boxFilter(ixy, ixy, CV_32F, cv::Size(kBlockSize, kBlockSize));
            timing.tensor_ms += elapsedMs(stage_start);

            stage_start = SteadyClock::now();
            cv::Mat diff_sq, ixy_sq;
            cv::pow(ix2 - iy2, 2, diff_sq);
            cv::pow(ixy * 2, 2, ixy_sq);
            cv::Mat term;
            cv::sqrt(diff_sq + ixy_sq, term);
            cv::Mat response = (ix2 + iy2 - term) * 0.5f;
            timing.response_ms += elapsedMs(stage_start);

            // 网格占用已经表达需要补点的 ROI，响应图只在这些 ROI 内有效。
            const cv::Rect inner_roi(
                roi.x - padded_roi.x, roi.y - padded_roi.y, roi.width, roi.height);
            const cv::Mat response_inner = response(inner_roi);
            response_inner.copyTo(ctc.grad(roi));
            double max_score = 0.0;
            cv::Point max_point;
            cv::minMaxLoc(response_inner, nullptr, &max_score, nullptr, &max_point, ctc.mask(roi));
            if (max_score > min_gradient_thres_) {
                best_scores[idx] = static_cast<float>(max_score);
                best_pts[idx] = cv::Point2f(
                    static_cast<float>(cell_x0 + max_point.x),
                    static_cast<float>(cell_y0 + max_point.y));
            }
        }
    }
    std::vector<size_t> candidate_indices;
    candidate_indices.reserve(ncells);
    for (size_t i = 0; i < ncells; ++i) {
        if (best_pts[i].x >= 0) {
            candidate_indices.push_back(i);
        }
    }
    std::sort(candidate_indices.begin(), candidate_indices.end(), [&](size_t lhs, size_t rhs) {
        return best_scores[lhs] > best_scores[rhs];
    });

    for (size_t index : candidate_indices) {
        const cv::Point2f& candidate = best_pts[index];
        const int x = cvRound(candidate.x);
        const int y = cvRound(candidate.y);
        if (ctc.mask.at<uchar>(y, x) == 0) {
            continue;
        }
        new_pts.push_back(candidate);
        cv::circle(ctc.mask, candidate, ctc.mask_radius, cv::Scalar(0), -1);
    }
    timing.grid_search_ms = elapsedMs(grid_start);
}

}  // namespace tassel_core
