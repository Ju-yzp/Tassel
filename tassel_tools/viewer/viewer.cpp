// Copyright (c) 2026 Wu JunPing
// Licensed under the MIT License.
// Design references: Open-VINS, Basalt, and VINS-Mono.

#include "viewer/viewer.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tassel_tools {
Viewer::Viewer(const std::string& frame_id) : Node("viewer"), frame_id_(frame_id) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
}

rclcpp::Time Viewer::messageStamp(double timestamp) const {
    if (timestamp < 0.0) {
        return this->now();
    }
    return rclcpp::Time(static_cast<int64_t>(timestamp * 1e9), RCL_ROS_TIME);
}

void Viewer::createCompressedImagePublisher(const std::string& topic_name, const rclcpp::QoS qos) {
    if (compressed_image_publishers_.find(topic_name) != compressed_image_publishers_.end()) {
        RCLCPP_WARN(
            this->get_logger(), "CompressedImage topic %s already exists, skipping creation.",
            topic_name.c_str());
        return;
    }
    auto publisher =
        this->template create_publisher<sensor_msgs::msg::CompressedImage>(topic_name, qos);
    compressed_image_publishers_[topic_name] = publisher;
}

void Viewer::publishCompressedImage(
    const std::string& topic, const std::string& frame_id, const cv::Mat& image,
    const std::string& format, double timestamp) {
    if (compressed_image_publishers_.find(topic) == compressed_image_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "CompressedImage topic %s not found!", topic.c_str());
        return;
    }

    std::vector<uchar> buf;
    if (format == "jpeg" || format == "jpg") {
        cv::imencode(".jpg", image, buf);
    } else if (format == "png") {
        cv::imencode(".png", image, buf);
    } else if (format == "webp") {
        cv::imencode(".webp", image, buf);
    } else {
        RCLCPP_WARN(
            this->get_logger(), "Unsupported compressed format '%s', falling back to jpeg.",
            format.c_str());
        cv::imencode(".jpg", image, buf);
    }

    sensor_msgs::msg::CompressedImage msg;
    msg.header.stamp = messageStamp(timestamp);
    msg.header.frame_id = frame_id;
    msg.format = format;
    msg.data = std::move(buf);

    compressed_image_publishers_[topic]->publish(msg);
}

void Viewer::createOdometryPublisher(
    const std::string& child_frame_id, const std::string& topic_name, const rclcpp::QoS& qos) {
    if (odometry_publishers_.find(topic_name) != odometry_publishers_.end()) {
        RCLCPP_WARN(
            this->get_logger(), "Odometry topic %s already exists, skipping creation.",
            topic_name.c_str());
        return;
    }
    auto publisher = this->template create_publisher<nav_msgs::msg::Odometry>(topic_name, qos);
    odometry_publishers_[topic_name] = publisher;
    nav_msgs::msg::Odometry odom;
    odom.header.frame_id = frame_id_;
    odom.child_frame_id = child_frame_id;
    odometry_[topic_name] = odom;
}

void Viewer::publishOdometry(
    const std::string& topic, const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation, const Eigen::Vector3d& linear_velocity,
    const Eigen::Vector3d& angular_velocity, double timestamp) {
    if (odometry_publishers_.find(topic) == odometry_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Odometry topic %s not found!", topic.c_str());
        return;
    }
    auto& odom = odometry_[topic];
    odom.header.stamp = messageStamp(timestamp);
    odom.pose.pose.position.x = position.x();
    odom.pose.pose.position.y = position.y();
    odom.pose.pose.position.z = position.z();
    odom.pose.pose.orientation.x = orientation.x();
    odom.pose.pose.orientation.y = orientation.y();
    odom.pose.pose.orientation.z = orientation.z();
    odom.pose.pose.orientation.w = orientation.w();
    odom.twist.twist.linear.x = linear_velocity.x();
    odom.twist.twist.linear.y = linear_velocity.y();
    odom.twist.twist.linear.z = linear_velocity.z();
    odom.twist.twist.angular.x = angular_velocity.x();
    odom.twist.twist.angular.y = angular_velocity.y();
    odom.twist.twist.angular.z = angular_velocity.z();

    odometry_publishers_[topic]->publish(odom);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = odom.header.stamp;
    tf.header.frame_id = frame_id_;
    tf.child_frame_id = odom.child_frame_id;
    tf.transform.translation.x = position.x();
    tf.transform.translation.y = position.y();
    tf.transform.translation.z = position.z();
    tf.transform.rotation.x = orientation.x();
    tf.transform.rotation.y = orientation.y();
    tf.transform.rotation.z = orientation.z();
    tf.transform.rotation.w = orientation.w();
    tf_broadcaster_->sendTransform(tf);
}

void Viewer::createPathPublisher(
    const std::string& topic_name, const rclcpp::QoS& qos, size_t max_poses) {
    if (path_publishers_.find(topic_name) != path_publishers_.end()) {
        RCLCPP_WARN(
            this->get_logger(), "Path topic %s already exists, skipping creation.",
            topic_name.c_str());
        return;
    }
    auto publisher = this->template create_publisher<nav_msgs::msg::Path>(topic_name, qos);
    path_publishers_[topic_name] = publisher;
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_id_;
    paths_[topic_name] = path;
    path_max_poses_[topic_name] = max_poses;
}

void Viewer::publishPath(
    const std::string& topic, const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation, double timestamp) {
    if (path_publishers_.find(topic) == path_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Path topic %s not found!", topic.c_str());
        return;
    }
    auto& path = paths_[topic];
    path.header.stamp = messageStamp(timestamp);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = path.header;
    pose_stamped.pose.position.x = position.x();
    pose_stamped.pose.position.y = position.y();
    pose_stamped.pose.position.z = position.z();
    pose_stamped.pose.orientation.x = orientation.x();
    pose_stamped.pose.orientation.y = orientation.y();
    pose_stamped.pose.orientation.z = orientation.z();
    pose_stamped.pose.orientation.w = orientation.w();

    path.poses.push_back(pose_stamped);
    const size_t max_poses = path_max_poses_[topic];
    if (max_poses > 0 && path.poses.size() > max_poses) {
        path.poses.erase(path.poses.begin(), path.poses.begin() + (path.poses.size() - max_poses));
    }
    path_publishers_[topic]->publish(path);
}

void Viewer::publishPathSnapshot(
    const std::string& topic, const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Quaterniond>& orientations, double timestamp) {
    if (path_publishers_.find(topic) == path_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Path topic %s not found!", topic.c_str());
        return;
    }
    if (positions.size() != orientations.size()) {
        throw std::invalid_argument("Path snapshot position and orientation counts differ");
    }

    auto& path = paths_[topic];
    path.header.stamp = messageStamp(timestamp);
    path.poses.clear();
    path.poses.reserve(positions.size());
    for (size_t index = 0; index < positions.size(); ++index) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = path.header;
        pose_stamped.pose.position.x = positions[index].x();
        pose_stamped.pose.position.y = positions[index].y();
        pose_stamped.pose.position.z = positions[index].z();
        pose_stamped.pose.orientation.x = orientations[index].x();
        pose_stamped.pose.orientation.y = orientations[index].y();
        pose_stamped.pose.orientation.z = orientations[index].z();
        pose_stamped.pose.orientation.w = orientations[index].w();
        path.poses.push_back(std::move(pose_stamped));
    }
    path_publishers_[topic]->publish(path);
}

void Viewer::publishVisualFactorWindow(
    const std::string& topic, const std::vector<int>& counts, double timestamp) {
    constexpr int kSegmentWidth = 64;
    constexpr int kImageHeight = 72;
    constexpr double kGreenFactorCount = 200.0;
    if (counts.empty()) {
        return;
    }

    cv::Mat image(kImageHeight, kSegmentWidth * static_cast<int>(counts.size()), CV_8UC3);
    for (size_t index = 0; index < counts.size(); ++index) {
        const int count = std::max(0, counts[index]);
        cv::Scalar color(0, 0, 0);
        if (count > 0) {
            const double ratio = std::clamp(count / kGreenFactorCount, 0.0, 1.0);
            const int red = ratio < 0.5 ? 255 : static_cast<int>(510.0 * (1.0 - ratio));
            const int green = ratio < 0.5 ? static_cast<int>(510.0 * ratio) : 255;
            color = cv::Scalar(0, green, red);
        }

        const int x0 = static_cast<int>(index) * kSegmentWidth;
        cv::rectangle(image, cv::Rect(x0, 0, kSegmentWidth, kImageHeight), color, cv::FILLED);
        cv::line(
            image, cv::Point(x0, 0), cv::Point(x0, kImageHeight - 1), cv::Scalar(96, 96, 96), 1);
        const cv::Scalar text_color =
            count == 0 ? cv::Scalar(255, 255, 255) : cv::Scalar(20, 20, 20);
        cv::putText(
            image, std::to_string(index), cv::Point(x0 + 6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.48,
            text_color, 1, cv::LINE_AA);
        cv::putText(
            image, std::to_string(count), cv::Point(x0 + 6, 53), cv::FONT_HERSHEY_SIMPLEX, 0.58,
            text_color, 1, cv::LINE_AA);
    }
    publishCompressedImage(topic, frame_id_, image, "png", timestamp);
}

}  // namespace tassel_tools
