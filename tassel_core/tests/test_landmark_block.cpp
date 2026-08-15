#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include <sophus/so3.hpp>

#include "cam/camera_rad_tan.h"
#include "factor/reprojection_factor.h"
#include "factor/visual_frame_cache.h"
#include "frond_end/feature.h"
#include "marg/landmark_block.h"
#include "state/state.h"

namespace tassel_core {
namespace {

constexpr double QrTolerance = 1e-12;
using EntryMap = std::map<std::pair<int, int>, double>;

double mappedValue(const EntryMap& entries, int row, int col) {
    const auto entry = entries.find({row, col});
    return entry == entries.end() ? 0.0 : entry->second;
}

void expectMatrixNear(
    const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, double tolerance) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    const double scale = std::max({1.0, actual.norm(), expected.norm()});
    EXPECT_LE((actual - expected).norm(), tolerance * scale);
}

TEST(LandmarkBlockTest, LinearizeFillsSpecifiedRowsAndColumns) {
    constexpr int NumFrames = 3;
    constexpr int StateSize = 6;
    State state(NumFrames);
    CameraRadTan camera(Eigen::Matrix3d::Identity(), Eigen::VectorXd::Zero(4), 640, 480);
    state.camera = &camera;
    state.visual_sqrt_info << 2.0, 0.0, 0.0, 3.0;
    state.param_time_delay = 0.015;
    state.frames[0].param_pose = {0.1, -0.2, 0.3, 0.02, -0.01, 0.03};
    state.frames[1].param_pose = {0.4, 0.1, -0.1, -0.03, 0.04, 0.01};
    state.frames[2].param_pose = {-0.2, 0.3, 0.2, 0.01, 0.02, -0.04};
    state.frames[0].imu_gyro = {0.1, -0.2, 0.3};
    state.frames[1].imu_gyro = {-0.2, 0.1, 0.05};
    state.frames[2].imu_gyro = {0.3, 0.2, -0.1};
    state.frames[0].imu_acc = {0.2, -0.1, 9.7};
    state.frames[1].imu_acc = {-0.1, 0.3, 9.8};
    state.frames[2].imu_acc = {0.4, 0.1, 9.6};

    Feature feature(0, NumFrames);
    feature.estimated_depth = 4.0;
    FeaturePerFrame host;
    host.setObservation({0.2, -0.1}, {0.2F, -0.1F});
    host.sync_delay = 0.01;
    FeaturePerFrame target1;
    target1.setObservation({0.16, -0.08}, {0.16F, -0.08F});
    target1.sync_delay = 0.012;
    FeaturePerFrame target2;
    target2.setObservation({0.23, -0.04}, {0.23F, -0.04F});
    target2.sync_delay = 0.018;
    feature.observations = {host, target1, target2};
    const Eigen::Matrix3d ric = Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.02, 0.03)).matrix();
    const Eigen::Vector3d tic(0.04, -0.01, 0.02);
    VisualFrameCache cache(state, ric);
    cache.PrepareForEvaluation(true, true);

    LandmarkBlock block(StateSize, nullptr);
    block.allocate(NumFrames, NumFrames - 1, StateSize);
    block.linearize(feature, -1, state, ric, tic);

    EntryMap expected;
    const double inverse_depth = 1.0 / feature.estimated_depth;
    for (int observation = 1; observation < NumFrames; ++observation) {
        const int target_frame = feature.observationFrameIndex(observation);
        const auto& target = feature.observations[observation];
        ReprojectionFactor factor(
            host.uv, Eigen::Vector2d(target.pt.x, target.pt.y), ric, tic, state.frames[0].imu_gyro,
            state.frames[target_frame].imu_gyro, state.frames[0].imu_acc,
            state.frames[target_frame].imu_acc, state.frames[0].param_speed_bias.data(),
            state.frames[target_frame].param_speed_bias.data(),
            state.frames[0].param_speed_bias.data() + 6,
            state.frames[target_frame].param_speed_bias.data() + 6,
            state.frames[0].param_speed_bias.data() + 3,
            state.frames[target_frame].param_speed_bias.data() + 3, state.visual_sqrt_info,
            state.camera, host.sync_delay, target.sync_delay, &state, 0, target_frame);
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> host_jacobian;
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> target_jacobian;
        Eigen::Vector2d delay_jacobian;
        Eigen::Vector2d landmark_jacobian;
        Eigen::Vector2d residual;
        std::vector<double*> jacobians = {
            host_jacobian.data(), target_jacobian.data(), delay_jacobian.data(),
            landmark_jacobian.data()};
        std::vector<const double*> parameters = {
            state.frames[0].param_pose.data(), state.frames[target_frame].param_pose.data(),
            &state.param_time_delay, &inverse_depth};
        ASSERT_TRUE(factor.Evaluate(parameters.data(), residual.data(), jacobians.data()));
        host_jacobian.rightCols<3>() *= Sophus::SO3d::leftJacobianInverse(
            -Eigen::Map<const Eigen::Vector3d>(state.frames[0].param_pose.data() + 3));
        target_jacobian.rightCols<3>() *= Sophus::SO3d::leftJacobianInverse(
            -Eigen::Map<const Eigen::Vector3d>(state.frames[target_frame].param_pose.data() + 3));

        const int row = (observation - 1) * 2;
        for (int local_row = 0; local_row < 2; ++local_row) {
            for (int local_col = 0; local_col < StateSize; ++local_col) {
                expected[{row + local_row, local_col}] = host_jacobian(local_row, local_col);
                expected[{row + local_row, target_frame * StateSize + local_col}] =
                    target_jacobian(local_row, local_col);
            }
            expected[{row + local_row, block.get_delay_index()}] = delay_jacobian(local_row);
            expected[{row + local_row, block.get_landmark_index()}] = landmark_jacobian(local_row);
            expected[{row + local_row, block.get_residual_index()}] = residual(local_row);
        }
    }

    const auto& storage = block.get_storage();
    EXPECT_EQ(storage.rows(), (NumFrames - 1) * 2);
    EXPECT_EQ(block.get_padding_index(), NumFrames * StateSize);
    EXPECT_EQ(block.get_delay_index() % 4, 0);
    EXPECT_EQ(block.get_landmark_index(), block.get_delay_index() + 1);
    EXPECT_EQ(block.get_residual_index(), block.get_landmark_index() + 1);
    for (int row = 0; row < storage.rows(); ++row) {
        for (int col = 0; col < storage.cols(); ++col) {
            EXPECT_NEAR(storage(row, col), mappedValue(expected, row, col), QrTolerance)
                << "row=" << row << " col=" << col;
        }
    }
}

TEST(LandmarkBlockTest, MarginalizeLandmarkMatchesHouseholderQr) {
    LandmarkBlock block(6, nullptr);
    block.allocate(3, 3, 6);
    auto& storage = block.get_mutable_storage();
    for (int row = 0; row < storage.rows(); ++row) {
        for (int col = 0; col < storage.cols(); ++col) {
            storage(row, col) = 0.25 * (row + 1) - 0.1 * (col + 2) + 0.01 * row * col;
        }
    }
    storage.col(block.get_landmark_index()) << 2.0, -1.0, 3.0, 4.0, -2.0, 1.0;
    const Eigen::MatrixXd original = storage;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(original.col(block.get_landmark_index()));
    const Eigen::MatrixXd expected = qr.householderQ().adjoint() * original;

    block.marginalizeLandmark();

    expectMatrixNear(storage, expected, QrTolerance);
    EXPECT_EQ(block.get_kept_rows(), storage.rows() - 1);
    EXPECT_LE(storage.col(block.get_landmark_index()).tail(storage.rows() - 1).norm(), QrTolerance);
}

TEST(LandmarkBlockTest, ZeroLandmarkJacobianKeepsEveryConstraint) {
    LandmarkBlock block(6, nullptr);
    block.allocate(2, 1, 6);
    auto& storage = block.get_mutable_storage();
    storage.setRandom();
    storage.col(block.get_landmark_index()).setZero();
    const Eigen::MatrixXd original = storage;

    block.marginalizeLandmark();

    EXPECT_EQ(block.get_kept_rows(), block.get_num_rows());
    EXPECT_TRUE(storage.isApprox(original));
}

}  // namespace
}  // namespace tassel_core
