#include <ceres/ceres.h>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "nori/nori_decoder.h"
#include "nori/nori_device.h"
#include "tassel_utils/types.h"

namespace {

constexpr int kCaptureWidth = 4000;
constexpr int kCaptureHeight = 1200;
constexpr double kGravity = 9.80665;

struct PoseMean {
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_std = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_std = Eigen::Vector3d::Zero();
    size_t count = 0;
    double start_time = 0.0;
    double end_time = 0.0;
};

struct AccNormResidual {
    explicit AccNormResidual(const Eigen::Vector3d& acc_mean) : acc_mean_(acc_mean) {}

    template <typename T>
    bool operator()(const T* const bias, const T* const lower, T* residual) const {
        const T x = T(acc_mean_.x()) - bias[0];
        const T y = T(acc_mean_.y()) - bias[1];
        const T z = T(acc_mean_.z()) - bias[2];

        // L 为下三角校正矩阵；全 3x3 会引入不可观的旋转自由度。
        const T cx = lower[0] * x;
        const T cy = lower[1] * x + lower[2] * y;
        const T cz = lower[3] * x + lower[4] * y + lower[5] * z;
        residual[0] = ceres::sqrt(cx * cx + cy * cy + cz * cz) - T(kGravity);
        return true;
    }

    Eigen::Vector3d acc_mean_;
};

Eigen::Vector3d meanVector(
    const std::vector<tassel_utils::IMUMeasurement>& measurements,
    const Eigen::Vector3d tassel_utils::IMUMeasurement::*field) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& measurement : measurements) {
        sum += measurement.*field;
    }
    return sum / static_cast<double>(measurements.size());
}

Eigen::Vector3d stdVector(
    const std::vector<tassel_utils::IMUMeasurement>& measurements,
    const Eigen::Vector3d tassel_utils::IMUMeasurement::*field, const Eigen::Vector3d& mean) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& measurement : measurements) {
        const Eigen::Vector3d diff = measurement.*field - mean;
        sum += diff.cwiseProduct(diff);
    }
    return (sum / static_cast<double>(measurements.size())).cwiseSqrt();
}

std::vector<tassel_utils::IMUMeasurement> readFrameImu(
    tassel_hardware::NoriDevice& device, tassel_hardware::NoriDecoder& decoder,
    int bytes_per_line) {
    tassel_hardware::NoriCapture capture;
    while (!device.tryRead(capture)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const size_t required_bytes = static_cast<size_t>(bytes_per_line) * kCaptureHeight;
    if (capture.bytes.size() < required_bytes) {
        throw std::runtime_error("Incomplete Nori YUYV frame");
    }
    const cv::Mat yuyv(
        kCaptureHeight, kCaptureWidth, CV_8UC2, capture.bytes.data(), bytes_per_line);
    cv::Mat gray;
    cv::extractChannel(yuyv, gray, 0);

    tassel_hardware::NoriFrameTiming timing;
    std::vector<tassel_utils::IMUMeasurement> measurements;
    if (!decoder.decode(gray, timing, measurements)) {
        measurements.clear();
    }
    return measurements;
}

PoseMean collectStaticPose(
    tassel_hardware::NoriDevice& device, tassel_hardware::NoriDecoder& decoder, int bytes_per_line,
    double duration_sec) {
    std::vector<tassel_utils::IMUMeasurement> collected;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
           duration_sec) {
        std::vector<tassel_utils::IMUMeasurement> frame_imu =
            readFrameImu(device, decoder, bytes_per_line);
        collected.insert(collected.end(), frame_imu.begin(), frame_imu.end());
    }

    if (collected.size() < 50) {
        throw std::runtime_error("Too few IMU samples collected for one static pose");
    }

    PoseMean pose;
    pose.acc = meanVector(collected, &tassel_utils::IMUMeasurement::acc);
    pose.gyro = meanVector(collected, &tassel_utils::IMUMeasurement::gyro);
    pose.acc_std = stdVector(collected, &tassel_utils::IMUMeasurement::acc, pose.acc);
    pose.gyro_std = stdVector(collected, &tassel_utils::IMUMeasurement::gyro, pose.gyro);
    pose.count = collected.size();
    pose.start_time = collected.front().timestamp;
    pose.end_time = collected.back().timestamp;
    return pose;
}

void writeYaml(
    const std::string& path, const std::vector<PoseMean>& poses, const double* acc_bias,
    const double* lower, const Eigen::Vector3d& gyro_bias, double final_cost) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output yaml: " + path);
    }
    out << std::setprecision(16);
    out << "imu_model: multi_static_lower_triangular\n";
    out << "gravity: " << kGravity << "\n";
    out << "accelerometer:\n";
    out << "  bias: [" << acc_bias[0] << ", " << acc_bias[1] << ", " << acc_bias[2] << "]\n";
    out << "  correction_matrix:\n";
    out << "    - [" << lower[0] << ", 0.0, 0.0]\n";
    out << "    - [" << lower[1] << ", " << lower[2] << ", 0.0]\n";
    out << "    - [" << lower[3] << ", " << lower[4] << ", " << lower[5] << "]\n";
    out << "gyroscope:\n";
    out << "  bias: [" << gyro_bias.x() << ", " << gyro_bias.y() << ", " << gyro_bias.z() << "]\n";
    out << "final_cost: " << final_cost << "\n";
    out << "poses:\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        const PoseMean& pose = poses[i];
        out << "  - index: " << i << "\n";
        out << "    count: " << pose.count << "\n";
        out << "    acc_mean: [" << pose.acc.x() << ", " << pose.acc.y() << ", " << pose.acc.z()
            << "]\n";
        out << "    gyro_mean: [" << pose.gyro.x() << ", " << pose.gyro.y() << ", " << pose.gyro.z()
            << "]\n";
        out << "    acc_std: [" << pose.acc_std.x() << ", " << pose.acc_std.y() << ", "
            << pose.acc_std.z() << "]\n";
        out << "    gyro_std: [" << pose.gyro_std.x() << ", " << pose.gyro_std.y() << ", "
            << pose.gyro_std.z() << "]\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string device_path = argc > 1 ? argv[1] : "/dev/video4";
    const int pose_count = argc > 2 ? std::stoi(argv[2]) : 6;
    const double duration_sec = argc > 3 ? std::stod(argv[3]) : 3.0;
    const std::string output_path = argc > 4 ? argv[4] : "/tmp/nori_imu_multistatic_calib.yaml";

    if (pose_count < 6) {
        throw std::invalid_argument("pose_count must be at least 6");
    }
    if (!std::isfinite(duration_sec) || duration_sec <= 0.0) {
        throw std::invalid_argument("duration_sec must be positive");
    }

    tassel_hardware::NoriDeviceConfig config;
    config.width = kCaptureWidth;
    config.height = kCaptureHeight;
    config.fps = 30;
    tassel_hardware::NoriDevice device(device_path, config);
    tassel_hardware::NoriDecoder decoder;
    const int bytes_per_line = device.bytesPerLine();

    std::vector<PoseMean> poses;
    poses.reserve(static_cast<size_t>(pose_count));
    std::cout << "Nori IMU multi-static calibration\n";
    std::cout << "device=" << device_path << " poses=" << pose_count << " duration=" << duration_sec
              << "s output=" << output_path << '\n';
    for (int i = 0; i < pose_count; ++i) {
        std::cout << "\nSet static pose " << (i + 1) << "/" << pose_count << ", then press Enter.";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        PoseMean pose = collectStaticPose(device, decoder, bytes_per_line, duration_sec);
        std::cout << "samples=" << pose.count << " acc=" << pose.acc.transpose()
                  << " |acc|=" << pose.acc.norm() << " gyro=" << pose.gyro.transpose()
                  << " gyro_std=" << pose.gyro_std.transpose() << '\n';
        poses.push_back(pose);
    }

    double acc_bias[3] = {0.0, 0.0, 0.0};
    double lower[6] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    ceres::Problem problem;
    for (const PoseMean& pose : poses) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AccNormResidual, 1, 3, 6>(
                new AccNormResidual(pose.acc)),
            nullptr, acc_bias, lower);
    }
    problem.SetParameterLowerBound(lower, 0, 0.5);
    problem.SetParameterLowerBound(lower, 2, 0.5);
    problem.SetParameterLowerBound(lower, 5, 0.5);
    problem.SetParameterUpperBound(lower, 0, 1.5);
    problem.SetParameterUpperBound(lower, 2, 1.5);
    problem.SetParameterUpperBound(lower, 5, 1.5);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    for (const PoseMean& pose : poses) {
        gyro_bias += static_cast<double>(pose.count) * pose.gyro;
    }
    const size_t total_count = std::accumulate(
        poses.begin(), poses.end(), size_t{0},
        [](size_t sum, const PoseMean& pose) { return sum + pose.count; });
    gyro_bias /= static_cast<double>(total_count);

    writeYaml(output_path, poses, acc_bias, lower, gyro_bias, summary.final_cost);
    std::cout << "\n" << summary.BriefReport() << '\n';
    std::cout << "acc_bias=" << Eigen::Map<Eigen::Vector3d>(acc_bias).transpose() << '\n';
    std::cout << "gyro_bias=" << gyro_bias.transpose() << '\n';
    std::cout << "written: " << output_path << '\n';
    return 0;
}
