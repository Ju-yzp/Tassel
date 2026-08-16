#include "initial/dynamic_initializer.h"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "frond_end/feature_manager.h"
#include "initial/initial_alignment.h"
#include "initial/initial_sfm.h"
#include "state/state.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

DynamicInitializer::DynamicInitializer(
    const tassel_tools::Parameters& params, std::shared_ptr<State> state,
    std::shared_ptr<FeatureManager> feature_manager, PreintegratorStorage& preintegrators,
    const Eigen::Matrix<double, 18, 18>& noise)
    : params_(params),
      state_(std::move(state)),
      feature_manager_(std::move(feature_manager)),
      preintegrators_(preintegrators),
      noise_(noise) {
    if (!state_ || !feature_manager_) {
        throw std::invalid_argument("Dynamic initializer requires state and feature manager");
    }
}

bool DynamicInitializer::initialize() {
    constexpr int kFirstActiveFrameIndex = 1;
    const int last_frame_index = state_->latest_active_frame_index;
    if (last_frame_index < kFirstActiveFrameIndex || last_frame_index >= state_->max_frame_count ||
        preintegrators_.size() < static_cast<size_t>(last_frame_index)) {
        throw std::logic_error("Dynamic initialization window has invalid storage bounds");
    }
    const int frame_count = last_frame_index - kFirstActiveFrameIndex + 1;
    if (frame_count < 2) {
        throw std::logic_error("Dynamic initialization requires at least two active frames");
    }

    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> positions;
    InitialSFM sfm(
        params_.sfm_min_points, params_.sfm_min_inliers, params_.sfm_epipolar_threshold,
        params_.sfm_pnp_threshold, params_.sfm_ba_iterations);
    if (!sfm.construct(
            *state_, *feature_manager_, params_.ric, rotations, positions,
            kFirstActiveFrameIndex)) {
        spdlog::warn("VIO initialization failed: SFM stage rejected all candidates");
        return false;
    }

    std::vector<Eigen::Vector3d> velocities(frame_count, Eigen::Vector3d::Zero());
    // 初始化各阶段必须具备事务语义；失败时不能留下半写入的 bias 或预积分结果。
    PreintegratorStorage working_preintegrators = preintegrators_;
    std::vector<Eigen::Matrix3d> rotation_bias_jacobians;
    std::vector<Eigen::Matrix3d> delta_rotations;
    Eigen::Vector3d bias_linearization = Eigen::Vector3d::Zero();
    bool has_bias_linearization = false;
    for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
        if (!has_bias_linearization) {
            bias_linearization = working_preintegrators[i].bg_linearized;
            has_bias_linearization = true;
        } else if (!working_preintegrators[i].bg_linearized.isApprox(bias_linearization, 1e-12)) {
            throw std::logic_error(
                "Initialization preintegrators use inconsistent gyro bias linearization");
        }
        rotation_bias_jacobians.push_back(working_preintegrators[i].get_dq_dbg());
        delta_rotations.push_back(working_preintegrators[i].final_delta_q);
    }

    const Eigen::Vector3d correction =
        solveGyroBiasCorrection(rotations, rotation_bias_jacobians, delta_rotations, params_.ric);
    if (!correction.allFinite()) {
        spdlog::warn("VIO initialization failed: gyro bias solve rejected the window");
        return false;
    }
    const Eigen::Vector3d gyro_bias = bias_linearization + correction;

    for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
        working_preintegrators[i].repropagate(state_->frames[i].accel_bias, gyro_bias, noise_);
    }

    std::vector<Eigen::Vector3d> delta_positions;
    std::vector<Eigen::Vector3d> delta_velocities;
    std::vector<double> dts;
    for (int i = kFirstActiveFrameIndex; i < last_frame_index; ++i) {
        delta_positions.push_back(working_preintegrators[i].final_delta_p);
        delta_velocities.push_back(working_preintegrators[i].final_delta_v);
        dts.push_back(working_preintegrators[i].sum_dt);
    }

    Eigen::Vector3d gravity;
    double scale = 0.0;
    if (!linearAlignment(
            rotations, positions, velocities, delta_velocities, delta_positions, dts, gravity,
            scale, params_.ric, params_.tic, params_.init_gravity_tolerance, params_.g_norm)) {
        spdlog::warn("VIO initialization failed: linear alignment rejected the window");
        return false;
    }
    if (!refineGravitySpeeds(
            velocities, rotations, positions, delta_velocities, delta_positions, dts, gravity,
            scale, params_.ric, params_.tic, params_.g_norm)) {
        spdlog::warn("VIO initialization failed: gravity refinement rejected the window");
        return false;
    }
    if (!std::isfinite(scale) || scale < params_.init_min_scale) {
        spdlog::warn("VIO initialization failed: invalid scale {:.6f}", scale);
        return false;
    }

    for (int i = kFirstActiveFrameIndex; i <= last_frame_index; ++i) {
        state_->frames[i].gyro_bias = gyro_bias;
    }
    preintegrators_ = std::move(working_preintegrators);

    Eigen::Matrix3d gravity_rotation =
        Eigen::Quaterniond::FromTwoVectors(gravity.normalized(), Eigen::Vector3d(0, 0, 1))
            .toRotationMatrix();
    const double yaw = std::atan2(gravity_rotation(1, 0), gravity_rotation(0, 0));
    gravity_rotation =
        Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * gravity_rotation;

    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    for (int local_index = 0; local_index < frame_count; ++local_index) {
        const int frame_index = kFirstActiveFrameIndex + local_index;
        state_->frames[frame_index].rot_w_i =
            Eigen::Quaterniond(
                gravity_rotation * params_.ric * rotations[local_index] * params_.ric.transpose())
                .normalized()
                .toRotationMatrix();
        state_->frames[frame_index].pos_w_i =
            gravity_rotation *
            (params_.ric * scale * positions[local_index] -
             params_.ric * rotations[local_index] * params_.ric.transpose() * params_.tic +
             params_.tic);
        state_->frames[frame_index].vel_w = gravity_rotation * velocities[local_index];
    }
    spdlog::info(
        "VI init: |g|={:.4f} s={:.4f} R0_yaw={:.2f}°", tassel_utils::G.norm(), scale,
        yaw * 180.0 / M_PI);
    return true;
}

}  // namespace tassel_core
