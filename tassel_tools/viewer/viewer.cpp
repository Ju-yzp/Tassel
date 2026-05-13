#include "viewer/viewer.h"

namespace tassel_tools {
Viewer::Viewer(const std::string& frame_id) : Node("viewer"), frame_id_(frame_id) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
}

// ── Image ──────────────────────────────────────────────────────────────────

void Viewer::createImagePublisher(const std::string& topic_name, const rclcpp::QoS qos) {
    if (image_publishers_.find(topic_name) != image_publishers_.end()) {
        RCLCPP_WARN(
            this->get_logger(), "Image topic %s already exists, skipping creation.",
            topic_name.c_str());
        return;
    }
    auto publisher = this->template create_publisher<sensor_msgs::msg::Image>(topic_name, qos);
    image_publishers_[topic_name] = publisher;
}

void Viewer::publishImage(
    const std::string& topic, const std::string& frame_id, const cv::Mat& image) {
    if (image_publishers_.find(topic) == image_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Image topic %s not found!", topic.c_str());
        return;
    }

    std::string encoding;
    switch (image.type()) {
        case CV_8UC1:
            encoding = sensor_msgs::image_encodings::MONO8;
            break;
        case CV_8UC3:
            encoding = sensor_msgs::image_encodings::BGR8;
            break;
        case CV_8UC4:
            encoding = sensor_msgs::image_encodings::BGRA8;
            break;
        case CV_16UC1:
            encoding = sensor_msgs::image_encodings::MONO16;
            break;
        case CV_32FC1:
            encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            break;
        default:
            RCLCPP_WARN(
                this->get_logger(),
                "Unsupported image type (depth=%d, channels=%d) for topic %s, skipping publish.",
                image.depth(), image.channels(), topic.c_str());
            return;
    }

    cv_bridge::CvImage cv_bridge_msg;
    cv_bridge_msg.header.stamp = this->now();
    cv_bridge_msg.header.frame_id = frame_id;
    cv_bridge_msg.encoding = encoding;
    cv_bridge_msg.image = image;

    image_publishers_[topic]->publish(*cv_bridge_msg.toImageMsg());
}

// ── CompressedImage ──────────────────────────────────────────────────────────

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
    const std::string& format) {
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
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id;
    msg.format = format;
    msg.data = std::move(buf);

    compressed_image_publishers_[topic]->publish(msg);
}

// ── Odometry + TF ──────────────────────────────────────────────────────────

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
    const Eigen::Vector3d& angular_velocity) {
    if (odometry_publishers_.find(topic) == odometry_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Odometry topic %s not found!", topic.c_str());
        return;
    }
    auto& odom = odometry_[topic];
    odom.header.stamp = this->now();
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

// ── PointCloud ──────────────────────────────────────────────────────────────

void Viewer::createPointCloudPublisher(const std::string& topic_name, const rclcpp::QoS& qos) {
    if (pointcloud_publishers_.find(topic_name) != pointcloud_publishers_.end()) {
        RCLCPP_WARN(
            this->get_logger(), "PointCloud topic %s already exists, skipping creation.",
            topic_name.c_str());
        return;
    }
    auto publisher =
        this->template create_publisher<sensor_msgs::msg::PointCloud2>(topic_name, qos);
    pointcloud_publishers_[topic_name] = publisher;
}

void Viewer::publishPointCloud(
    const std::string& topic, const std::vector<Eigen::Vector3d>& points) {
    if (pointcloud_publishers_.find(topic) == pointcloud_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "PointCloud topic %s not found!", topic.c_str());
        return;
    }
    if (points.empty()) return;

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = this->now();
    cloud.header.frame_id = frame_id_;
    cloud.height = 1;
    cloud.width = points.size();
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    cloud.fields.resize(3);
    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;
    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;
    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;

    cloud.point_step = 12;
    cloud.row_step = cloud.width * cloud.point_step;
    cloud.data.resize(cloud.row_step);

    auto* data = reinterpret_cast<float*>(cloud.data.data());
    for (size_t i = 0; i < points.size(); ++i) {
        data[i * 3 + 0] = static_cast<float>(points[i].x());
        data[i * 3 + 1] = static_cast<float>(points[i].y());
        data[i * 3 + 2] = static_cast<float>(points[i].z());
    }

    pointcloud_publishers_[topic]->publish(cloud);
}

// ── Error ──────────────────────────────────────────────────────────────────

void Viewer::createErrorPublisher(const std::string& topic_name, const rclcpp::QoS& qos) {
    if (error_publishers_.find(topic_name) != error_publishers_.end()) {
        RCLCPP_WARN(
            this->get_logger(), "Error topic %s already exists, skipping creation.",
            topic_name.c_str());
        return;
    }
    auto publisher = this->template create_publisher<std_msgs::msg::Float32>(topic_name, qos);
    error_publishers_[topic_name] = publisher;
}

void Viewer::publishError(const std::string& topic, float error) {
    if (error_publishers_.find(topic) == error_publishers_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Error topic %s not found!", topic.c_str());
        return;
    }

    std_msgs::msg::Float32 msg;
    msg.data = error;
    error_publishers_[topic]->publish(msg);
}

}  // namespace tassel_tools
