#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <ceres/gradient_checker.h>
#include <sophus/so3.hpp>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "imu_test_utils.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

class ImuFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        constexpr double imu_dt = 0.0025;
        constexpr int steps_per_frame = 67;
        constexpr int frame_count = 6;
        const Eigen::Vector3d acceleration(0.3, -0.1, 0.05);
        const Eigen::Vector3d angular_velocity(0.1, 0.3, 0.4);
        const Eigen::Vector3d accel_bias(0.08, -0.04, 0.03);
        const Eigen::Vector3d gyro_bias(0.012, -0.006, 0.004);
        const test::ImuTimeline timeline = test::generateConstantMotionTimeline(
            frame_count * steps_per_frame * imu_dt, imu_dt, acceleration, angular_velocity,
            accel_bias, gyro_bias);

        Eigen::Matrix<double, 18, 18> noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = 0.0193 * 0.0193 * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = 0.00264 * 0.00264 * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = noise.block<3, 3>(0, 0);
        noise.block<3, 3>(9, 9) = noise.block<3, 3>(3, 3);
        noise.block<3, 3>(12, 12) = 1.48e-7 * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = 1.48e-7 * Eigen::Matrix3d::Identity();

        poses_.resize(frame_count);
        speed_biases_.resize(frame_count);
        preintegrators_.resize(frame_count - 1);
        for (int frame = 0; frame < frame_count; ++frame) {
            const auto& sample = timeline.at_index(frame * steps_per_frame);
            const Eigen::Vector3d rotation_vector = Sophus::SO3d(sample.rot_w_i).log();
            for (int d = 0; d < 3; ++d) {
                poses_[frame][d] = sample.pos_w_i[d];
                poses_[frame][d + 3] = rotation_vector[d];
                speed_biases_[frame][d] = sample.vel_w[d];
                speed_biases_[frame][d + 3] = accel_bias[d];
                speed_biases_[frame][d + 6] = gyro_bias[d];
            }
            if (frame == frame_count - 1) {
                continue;
            }

            auto preintegrator = std::make_shared<MidPointIntegrator>(accel_bias, gyro_bias, noise);
            const int first_sample = frame * steps_per_frame;
            const int last_sample = (frame + 1) * steps_per_frame;
            for (int sample_index = first_sample; sample_index <= last_sample; ++sample_index) {
                const auto& imu = timeline.at_index(sample_index);
                tassel_utils::IMUMeasurement measurement;
                measurement.timestamp = imu.ts;
                measurement.acc = imu.acc;
                measurement.gyro = imu.gyro;
                ASSERT_TRUE(preintegrator->propagate(measurement));
            }
            preintegrators_[frame] = std::move(preintegrator);
        }
    }

    std::vector<std::array<double, 6>> poses_;
    std::vector<std::array<double, 9>> speed_biases_;
    std::vector<std::shared_ptr<MidPointIntegrator>> preintegrators_;
};

TEST_F(ImuFactorTest, AnalyticJacobiansMatchNumericDifferentiation) {
    SE3RightManifold manifold;
    const std::vector<const ceres::Manifold*> manifolds = {&manifold, nullptr, &manifold, nullptr};
    ceres::NumericDiffOptions options;
    options.relative_step_size = 1e-6;

    for (size_t frame = 0; frame < preintegrators_.size(); ++frame) {
        IMUFactor<MidPointIntegrator> factor(preintegrators_[frame]);
        ceres::GradientChecker checker(&factor, &manifolds, options);
        const double* parameters[] = {
            poses_[frame].data(), speed_biases_[frame].data(), poses_[frame + 1].data(),
            speed_biases_[frame + 1].data()};
        ceres::GradientChecker::ProbeResults results;
        const bool relative_check_ok = checker.Probe(parameters, 5e-4, &results);
        double maximum_absolute_error = 0.0;
        for (size_t block = 0; block < results.local_jacobians.size(); ++block) {
            maximum_absolute_error = std::max(
                maximum_absolute_error,
                (results.local_jacobians[block] - results.local_numeric_jacobians[block])
                    .cwiseAbs()
                    .maxCoeff());
        }
        EXPECT_TRUE(relative_check_ok || maximum_absolute_error < 1e-7)
            << "frame=" << frame << " max_relative_error=" << results.maximum_relative_error
            << " max_absolute_error=" << maximum_absolute_error << '\n'
            << results.error_log;
    }
}

}  // namespace
}  // namespace tassel_core
