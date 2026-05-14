#include <depthai/depthai.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cam/camera_rad_tan.h"
#include "frond_end/feature_tracker.h"
#include "tassel_utils/timer.h"

int main() {
    // ── cam0 (left) intrinsics ─────────────────────────────────────────────
    cv::Mat K0 =
        (cv::Mat_<double>(3, 3) << 455.510864, 0.000000, 328.529851, 0.000000, 455.426715,
         225.596721, 0.0, 0.0, 1.0);
    cv::Mat D0 = (cv::Mat_<double>(1, 5) << 0.010831, -0.007841, 0.000166, 0.000512, 0.000000);

    // ── cam1 (right) intrinsics ────────────────────────────────────────────
    cv::Mat K1 =
        (cv::Mat_<double>(3, 3) << 462.007162, 0.000000, 313.439176, 0.000000, 461.452922,
         229.403156, 0.0, 0.0, 1.0);
    cv::Mat D1 = (cv::Mat_<double>(1, 5) << 0.028404, -0.022619, -0.000793, -0.002258, 0.000000);

    // ── Tracker settings ───────────────────────────────────────────────────
    const int rows = 480;
    const int cols = 640;
    const int per_grid_rows = 40;
    const int per_grid_cols = 60;
    const int edge_x = 15;
    const int edge_y = 15;
    const double mask_radius = 15.0;
    const double max_square_move_dist = 0.8;
    const bool flow_back = true;
    const double min_gradient = 20.0;
    const bool enable_statistics = false;

    // ── OAK pipeline — left + right mono cameras ───────────────────────────
    dai::Pipeline pipeline;

    auto mono_left = pipeline.create<dai::node::MonoCamera>();
    mono_left->setCamera("left");
    mono_left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    auto xout_left = pipeline.create<dai::node::XLinkOut>();
    xout_left->setStreamName("left");
    mono_left->out.link(xout_left->input);

    auto mono_right = pipeline.create<dai::node::MonoCamera>();
    mono_right->setCamera("right");
    mono_right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    auto xout_right = pipeline.create<dai::node::XLinkOut>();
    xout_right->setStreamName("right");
    mono_right->out.link(xout_right->input);

    dai::Device device(pipeline);
    auto queue_left = device.getOutputQueue("left", 8, false);
    auto queue_right = device.getOutputQueue("right", 8, false);

    // ── Feature tracker with left (id=0) and right (id=1) cameras ──────────
    tassel_core::FeatureTracker tracker(
        flow_back, max_square_move_dist, enable_statistics, 5, min_gradient);

    auto cam0 = std::make_unique<tassel_core::CameraRadTan>(K0, D0, cols, rows);
    tracker.addCamera(std::move(cam0), per_grid_rows, per_grid_cols, edge_y, edge_x, mask_radius);

    auto cam1 = std::make_unique<tassel_core::CameraRadTan>(K1, D1, cols, rows);
    tracker.addCamera(std::move(cam1), per_grid_rows, per_grid_cols, edge_y, edge_x, mask_radius);

    cv::namedWindow("stereo", cv::WINDOW_NORMAL);

    while (true) {
        auto left_frame = queue_left->get<dai::ImgFrame>();
        auto right_frame = queue_right->get<dai::ImgFrame>();
        if (!left_frame || !right_frame) continue;

        auto left_data = left_frame->getData();
        cv::Mat left_img(
            left_frame->getHeight(), left_frame->getWidth(), CV_8UC1, left_data.data());
        auto right_data = right_frame->getData();
        cv::Mat right_img(
            right_frame->getHeight(), right_frame->getWidth(), CV_8UC1, right_data.data());
        left_img = left_img.clone();
        right_img = right_img.clone();

        {
            tassel_utils::Timer t("stereoTracking");
            tracker.stereoTracking(0, left_img, 1, right_img);
        }

        cv::Mat disp_left, disp_right;
        cv::cvtColor(left_img, disp_left, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(0, disp_left);
        cv::cvtColor(right_img, disp_right, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(1, disp_right);

        cv::Mat stereo;
        cv::hconcat(disp_left, disp_right, stereo);
        cv::imshow("stereo", stereo);
        if (cv::waitKey(1) == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
