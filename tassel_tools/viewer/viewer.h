#ifndef TASSEL_TOOLS_VIEWER_H_
#define TASSEL_TOOLS_VIEWER_H_

// rclcpp
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

// eigen
#include <Eigen/Dense>

// cpp
#include <string>
#include <unordered_map>
#include <vector>

// opencv
#include <opencv2/core.hpp>

namespace tassel_tools {
class Viewer : public rclcpp::Node {
public:
    explicit Viewer(const std::string& frame_id = "world");

    void createImagePublisher(
        const std::string& topic_name, const rclcpp::QoS qos = rclcpp::QoS(10));

    void publishImage(
        const std::string& topic_name, const std::string& frame_id, const cv::Mat& image);

    void createCompressedImagePublisher(
        const std::string& topic_name, const rclcpp::QoS qos = rclcpp::QoS(10));

    void publishCompressedImage(
        const std::string& topic, const std::string& frame_id, const cv::Mat& image,
        const std::string& format = "jpeg");

    void createOdometryPublisher(
        const std::string& child_frame_id, const std::string& topic_name,
        const rclcpp::QoS& qos = rclcpp::QoS(10));

    void publishOdometry(
        const std::string& topic, const Eigen::Vector3d& position,
        const Eigen::Quaterniond& orientation,
        const Eigen::Vector3d& linear_velocity = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d& angular_velocity = Eigen::Vector3d::Zero());

    void createPathPublisher(
        const std::string& topic_name, const rclcpp::QoS& qos = rclcpp::QoS(10));

    void publishPath(
        const std::string& topic, const Eigen::Vector3d& position,
        const Eigen::Quaterniond& orientation);

    void createPointCloudPublisher(
        const std::string& topic_name, const rclcpp::QoS& qos = rclcpp::QoS(10));

    void publishPointCloud(const std::string& topic, const std::vector<Eigen::Vector3d>& points);

    void createErrorPublisher(
        const std::string& topic_name, const rclcpp::QoS& qos = rclcpp::QoS(10));

    void publishError(const std::string& topic, float error);

private:
    // image
    std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>
        image_publishers_;

    // compressed image
    std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr>
        compressed_image_publishers_;

    // odometry
    std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr>
        odometry_publishers_;
    std::unordered_map<std::string, nav_msgs::msg::Odometry> odometry_;

    // path
    std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr>
        path_publishers_;
    std::unordered_map<std::string, nav_msgs::msg::Path> paths_;

    // pointcloud
    std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr>
        pointcloud_publishers_;

    // error
    std::unordered_map<std::string, rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr>
        error_publishers_;

    std::string frame_id_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
}  // namespace tassel_tools
#endif  // TASSEL_TOOLS_VIEWER_H_
