#include "vio_gauge.h"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace tassel_core {

namespace {

double yaw(const Eigen::Matrix3d& rotation) { return std::atan2(rotation(1, 0), rotation(0, 0)); }

}  // namespace

void restoreVioGauge(
    State& state, int first_frame_index, const Eigen::Matrix3d& reference_rotation,
    const Eigen::Vector3d& reference_position) {
    if (first_frame_index < 0 || first_frame_index > state.latest_frame_index) {
        throw std::out_of_range("VIO gauge reference frame is outside the active window");
    }

    const FrameState& optimized_reference = state.frames[first_frame_index];
    const double yaw_correction = yaw(reference_rotation) - yaw(optimized_reference.R);
    const Eigen::Matrix3d rotation_correction =
        Eigen::AngleAxisd(yaw_correction, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d optimized_reference_position = optimized_reference.P;

    for (int frame_index = first_frame_index; frame_index <= state.latest_frame_index;
         ++frame_index) {
        FrameState& frame = state.frames[frame_index];
        frame.P =
            rotation_correction * (frame.P - optimized_reference_position) + reference_position;
        frame.R = rotation_correction * frame.R;
        frame.V = rotation_correction * frame.V;
    }
}

}  // namespace tassel_core
