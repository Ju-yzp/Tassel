// =============================================================================
// test_viewer.cpp
//
// 目的：
//   手动检查 Viewer 的里程计、轨迹、图像、点云和误差发布接口。
//
// 测试设计：
//   构造一个椭圆轨道运动模型, 持续发布相机位姿、路径、合成图像、点云和误差曲线,
//   用 ROS2/RViz 等外部工具观察 topic 是否正确更新。
//
// 通过条件：
//   这是依赖 ROS2 可视化环境的冒烟测试；各 topic 能持续发布且可视化结果随时间
//   平滑变化。
// =============================================================================

#include "viewer/viewer.h"

#include <algorithm>

class EllipticalOrbit {
public:
    EllipticalOrbit(double a, double b, double omega, const Eigen::Vector3d& normal)
        : a_(a), b_(b), omega_(omega), normal_(normal.normalized()) {
        Eigen::Vector3d arbitrary(0.0, 0.0, 1.0);
        if (std::abs(normal_.dot(arbitrary)) > 0.99) {
            arbitrary = Eigen::Vector3d(1.0, 0.0, 0.0);
        }
        u_ = normal_.cross(arbitrary).normalized();
        v_ = normal_.cross(u_).normalized();
    }

    Eigen::Vector3d position(double t) const {
        double theta = omega_ * t;
        return a_ * std::cos(theta) * u_ + b_ * std::sin(theta) * v_;
    }

    Eigen::Vector3d velocity(double t) const {
        double theta = omega_ * t;
        return -a_ * omega_ * std::sin(theta) * u_ + b_ * omega_ * std::cos(theta) * v_;
    }

    Eigen::Quaterniond orientation(double t) const {
        Eigen::Vector3d forward = -position(t).normalized();
        Eigen::Vector3d right = forward.cross(normal_).normalized();
        Eigen::Vector3d up = right.cross(forward).normalized();

        Eigen::Matrix3d rot;
        rot.col(0) = right;
        rot.col(1) = forward;
        rot.col(2) = up;
        return Eigen::Quaterniond(rot);
    }

    Eigen::Vector3d angularVelocity() const { return omega_ * normal_; }

private:
    double a_, b_, omega_;
    Eigen::Vector3d normal_, u_, v_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto viewer = std::make_shared<tassel_tools::Viewer>("world");

    viewer->createOdometryPublisher("camera", "odom/camera");
    viewer->createPathPublisher("path/camera");
    viewer->createImagePublisher("image/camera");
    viewer->createPointCloudPublisher("landmarks");
    viewer->createErrorPublisher("error");

    const double a = 5.0;
    const double b = 3.0;
    const double omega = 0.4;
    const double dt = 0.05;
    const double sphere_r = 0.8;
    const double fx = 500.0;
    const double img_w = 640.0;
    const double img_h = 480.0;

    EllipticalOrbit orbit(a, b, omega, Eigen::Vector3d(0.0, 1.0, 0.0));

    const cv::Scalar sphere_color(51, 179, 255);  // BGR

    cv::Mat canvas(img_h, img_w, CV_8UC3);
    rclcpp::Time start = viewer->now();
    rclcpp::Rate rate(1.0 / dt);

    while (rclcpp::ok()) {
        double t = (viewer->now() - start).seconds();

        auto pos = orbit.position(t);
        auto vel = orbit.velocity(t);
        auto ori = orbit.orientation(t);
        auto ang_vel = orbit.angularVelocity();

        canvas.setTo(cv::Scalar(30, 30, 30));
        double dist = pos.norm();
        int proj_r = static_cast<int>(fx * sphere_r / dist);
        proj_r = std::max(proj_r, 3);
        cv::Point center(static_cast<int>(img_w / 2), static_cast<int>(img_h / 2));

        cv::circle(canvas, center, proj_r, sphere_color, cv::FILLED);

        cv::putText(
            canvas, "t = " + std::to_string(t).substr(0, 5) + " s", cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
        cv::putText(
            canvas,
            "d = " + std::to_string(dist).substr(0, 4) + " m  r = " + std::to_string(proj_r) +
                " px",
            cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 2);

        viewer->publishOdometry("odom/camera", pos, ori, vel, ang_vel);
        viewer->publishPath("path/camera", pos, ori);
        viewer->publishImage("image/camera", "camera", canvas);
        viewer->publishPointCloud("landmarks", {pos});
        viewer->publishError("error", static_cast<float>(dist));

        rclcpp::spin_some(viewer);
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
