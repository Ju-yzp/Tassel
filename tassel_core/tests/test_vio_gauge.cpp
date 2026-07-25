#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "state/vio_gauge.h"

namespace tassel_core {
namespace {

TEST(VioGaugeTest, RestoresPositionAndYawWithoutChangingObservableTilt) {
    State state(4);
    state.latest_frame_index = 3;
    const int first_frame_index = 1;

    std::vector<FrameState> reference_frames = state.frames;
    for (int i = first_frame_index; i <= state.latest_frame_index; ++i) {
        reference_frames[i].P = Eigen::Vector3d(0.4 * i, -0.2 * i, 0.1 * i);
        reference_frames[i].R = (Eigen::AngleAxisd(0.15 * i, Eigen::Vector3d::UnitZ()) *
                                 Eigen::AngleAxisd(-0.03 * i, Eigen::Vector3d::UnitY()) *
                                 Eigen::AngleAxisd(0.02 * i, Eigen::Vector3d::UnitX()))
                                    .toRotationMatrix();
        reference_frames[i].V = Eigen::Vector3d(0.3 * i, 0.1 * i, -0.05 * i);
        reference_frames[i].Ba = Eigen::Vector3d(0.01, 0.02, 0.03);
        reference_frames[i].Bg = Eigen::Vector3d(-0.01, 0.005, 0.002);
    }

    const Eigen::Matrix3d drift_rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d drift_translation(2.0, -3.0, 0.8);
    for (int i = first_frame_index; i <= state.latest_frame_index; ++i) {
        state.frames[i] = reference_frames[i];
        state.frames[i].P = drift_rotation * reference_frames[i].P + drift_translation;
        state.frames[i].R = drift_rotation * reference_frames[i].R;
        state.frames[i].V = drift_rotation * reference_frames[i].V;
    }

    restoreVioGauge(
        state, first_frame_index, reference_frames[first_frame_index].R,
        reference_frames[first_frame_index].P);

    for (int i = first_frame_index; i <= state.latest_frame_index; ++i) {
        EXPECT_TRUE(state.frames[i].P.isApprox(reference_frames[i].P, 1e-12));
        EXPECT_TRUE(state.frames[i].R.isApprox(reference_frames[i].R, 1e-12));
        EXPECT_TRUE(state.frames[i].V.isApprox(reference_frames[i].V, 1e-12));
        EXPECT_TRUE(state.frames[i].Ba.isApprox(reference_frames[i].Ba, 1e-12));
        EXPECT_TRUE(state.frames[i].Bg.isApprox(reference_frames[i].Bg, 1e-12));
    }
}

TEST(VioGaugeTest, RejectsReferenceOutsideActiveWindow) {
    State state(3);
    state.latest_frame_index = 2;
    EXPECT_THROW(
        restoreVioGauge(state, 3, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()),
        std::out_of_range);
}

TEST(VioGaugeTest, KeepsObservableTiltUpdate) {
    State state(2);
    state.latest_frame_index = 1;
    const Eigen::Matrix3d reference_rotation =
        Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d optimized_rotation = (Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()) *
                                                Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()))
                                                   .toRotationMatrix();
    state.frames[1].R = optimized_rotation;

    restoreVioGauge(state, 1, reference_rotation, Eigen::Vector3d::Zero());

    // 航向被恢复，而倾斜相对重力方向的夹角保持优化结果。
    EXPECT_NEAR(std::atan2(state.frames[1].R(1, 0), state.frames[1].R(0, 0)), 0.2, 1e-12);
    EXPECT_NEAR(state.frames[1].R(2, 2), optimized_rotation(2, 2), 1e-12);
    EXPECT_FALSE(state.frames[1].R.isApprox(reference_rotation, 1e-6));
}

}  // namespace
}  // namespace tassel_core
