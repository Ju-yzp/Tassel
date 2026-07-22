#ifndef TASSEL_TOOLS_VIEWER_H_
#define TASSEL_TOOLS_VIEWER_H_

// rclcpp
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

// Eigen
#include <Eigen/Dense>

// 标准库
#include <string>
#include <unordered_map>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

namespace tassel_tools {
class Viewer : public rclcpp::Node {
public:
    explicit Viewer(const std::string& frame_id = "world");

    void createCompressedImagePublisher(
        const std::string& topic_name, const rclcpp::QoS qos = rclcpp::QoS(10));

    void publishCompressedImage(
        const std::string& topic, const std::string& frame_id, const cv::Mat& image,
        const std::string& format = "jpeg", double timestamp = -1.0);

    void createOdometryPublisher(
        const std::string& child_frame_id, const std::string& topic_name,
        const rclcpp::QoS& qos = rclcpp::QoS(10));

    void publishOdometry(
        const std::string& topic, const Eigen::Vector3d& position,
        const Eigen::Quaterniond& orientation,
        const Eigen::Vector3d& linear_velocity = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d& angular_velocity = Eigen::Vector3d::Zero(), double timestamp = -1.0);

    void createPathPublisher(
        const std::string& topic_name, const rclcpp::QoS& qos = rclcpp::QoS(10),
        size_t max_poses = 300);

    void publishPath(
        const std::string& topic, const Eigen::Vector3d& position,
        const Eigen::Quaterniond& orientation, double timestamp = -1.0);

    void publishPathSnapshot(
        const std::string& topic, const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Quaterniond>& orientations, double timestamp = -1.0);

    void publishVisualFactorWindow(
        const std::string& topic, const std::vector<int>& counts, double timestamp = -1.0);

private:
    // 压缩图像
    std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr>
        compressed_image_publishers_;

    // 里程计
    std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr>
        odometry_publishers_;
    std::unordered_map<std::string, nav_msgs::msg::Odometry> odometry_;

    // 轨迹
    std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr>
        path_publishers_;
    std::unordered_map<std::string, nav_msgs::msg::Path> paths_;
    std::unordered_map<std::string, size_t> path_max_poses_;

    std::string frame_id_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Time messageStamp(double timestamp) const;
};
}  // namespace tassel_tools
#endif  // TASSEL_TOOLS_VIEWER_H_
