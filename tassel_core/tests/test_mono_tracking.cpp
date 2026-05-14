#include <depthai/depthai.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cam/camera_rad_tan.h"
#include "frond_end/feature_tracker.h"
#include "tassel_utils/timer.h"

int main() {
    // Camera intrinsics — cam0 (left)
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 455.510864, 0.000000, 328.529851, 0.000000, 455.426715,
         225.596721, 0.0, 0.0, 1.0);
    cv::Mat D = (cv::Mat_<double>(1, 5) << 0.010831, -0.007841, 0.000166, 0.000512, 0.000000);

    // Tracker settings
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

    // OAK pipeline — built-in auto exposure
    dai::Pipeline pipeline;
    auto mono = pipeline.create<dai::node::MonoCamera>();
    auto xout = pipeline.create<dai::node::XLinkOut>();
    xout->setStreamName("mono");
    mono->out.link(xout->input);
    mono->setCamera("left");
    mono->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

    dai::Device device(pipeline);
    auto queue = device.getOutputQueue("mono", 8, false);

    // Feature tracker
    auto camera = std::make_unique<tassel_core::CameraRadTan>(K, D, cols, rows);
    tassel_core::FeatureTracker tracker(
        flow_back, max_square_move_dist, enable_statistics, 5, min_gradient);
    tracker.addCamera(std::move(camera), per_grid_rows, per_grid_cols, edge_y, edge_x, mask_radius);

    cv::namedWindow("tracking", cv::WINDOW_NORMAL);

    while (true) {
        auto frame = queue->get<dai::ImgFrame>();
        if (!frame) continue;

        auto data = frame->getData();
        cv::Mat img(frame->getHeight(), frame->getWidth(), CV_8UC1, data.data());
        img = img.clone();

        {
            tassel_utils::Timer t("monoTracking");
            tracker.monoTracking(0, img);
        }

        cv::Mat disp;
        cv::cvtColor(img, disp, cv::COLOR_GRAY2BGR);
        tracker.drawTrackingResult(0, disp);

        cv::imshow("tracking", disp);
        if (cv::waitKey(1) == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
