#include "estimator.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>
#include <cstddef>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <vector>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/visual_factor.h"
#include "marg/marg_helper.h"
#include "marg/marginlization_sqrt.h"
#include "tassel_utils/se3_right_manifold.h"

#include "initial/initial_alignment.h"
#include "initial/initial_sfm.h"

namespace tassel_core {
Estimator::Estimator(
    const tassel_tools::Parameters& params, std::shared_ptr<State> state,
    std::shared_ptr<FeatureManager> fm)
    : params_(params), state_(std::move(state)), feature_manager_(std::move(fm)) {
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    cv::setNumThreads(params_.num_threads);
    noise_ = initNoise();
    state_->visual_sqrt_info = Eigen::Matrix2d::Identity() * params_.visual_factor_weight;
    reset();
}

void Estimator::reset() {
    gravity_initialized_ = false;
    last_ts_ = -1;
    last_imu_acc_ = Eigen::Vector3d::Zero();
    last_imu_gyro_ = Eigen::Vector3d::Zero();
    preintegrators_.clear();
    preintegrators_.resize(
        state_->max_frame_count - 1,
        MidPointIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_));
    Rs_.resize(state_->max_frame_count, Eigen::Matrix3d::Identity());
    Ps_.resize(state_->max_frame_count, Eigen::Vector3d::Zero());
    Vs_.resize(state_->max_frame_count, Eigen::Vector3d::Zero());
    marg_data_.reset();
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    state_->reset();
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    int& frame_count = state_->cur_frame_count;
    if (last_ts_ < 0 && !imu_measurements.empty()) {
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc - params_.acc_bias;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

    if (gravity_initialized_) {
        return;
    }
    bool is_keyframe = feature_manager_->checkParallax(frame_count, feature_frame);

    if (frame_count > 0) {
        Eigen::Matrix3d R = state_->Rs[frame_count];
        Eigen::Vector3d P = state_->Ps[frame_count];
        Eigen::Vector3d V = state_->Vs[frame_count];
        Eigen::Vector3d Ba = state_->Bas[frame_count];
        Eigen::Vector3d Bg = state_->Bgs[frame_count];
        auto& preintegrator = preintegrators_[frame_count - 1];

        for (const auto& imu : imu_measurements) {
            tassel_utils::IMUMeasurement imu_cal = imu;
            imu_cal.acc = imu.acc - params_.acc_bias;
            preintegrator.propagate(imu_cal);

            double dt = imu_cal.timestamp - last_ts_;
            Eigen::Vector3d acc_0 = R * (last_imu_acc_ - Ba) - tassel_utils::G;
            Eigen::Vector3d gyr = 0.5 * (last_imu_gyro_ + imu_cal.gyro) - Bg;
            R = R * Sophus::SO3d::exp(gyr * dt).matrix();
            Eigen::Vector3d acc_1 = R * (imu_cal.acc - Ba) - tassel_utils::G;
            Eigen::Vector3d acc = 0.5 * (acc_0 + acc_1);
            P += V * dt + 0.5 * acc * dt * dt;
            V += acc * dt;
            last_ts_ = imu_cal.timestamp;
            last_imu_gyro_ = imu_cal.gyro;
            last_imu_acc_ = imu_cal.acc;
        }

        Eigen::Quaterniond q(R);
        q.normalize();
        state_->Rs[frame_count] = q.matrix();
        state_->Ps[frame_count] = P;
        state_->Vs[frame_count] = V;
    }

    state_->acc_vec.push_back(last_imu_acc_);
    state_->gyro_vec.push_back(last_imu_gyro_);
    if (is_keyframe) {
        feature_manager_->triangulate(
            *state_, params_.ric, params_.tic, params_.ric1, params_.tic1);

        if (frame_count == state_->max_frame_count - 1) {
            if (!gravity_initialized_) {
                if (!tryInitialize()) {
                    spdlog::warn("VI initialization failed, resetting");
                    reset();
                    return;
                }
                feature_manager_->invalidateDepths();
                feature_manager_->triangulate(
                    *state_, params_.ric, params_.tic, params_.ric1, params_.tic1);
            }
            optimize();
            feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
            buildPrior();
            feature_manager_->removeOldest(*state_, params_.ric, params_.tic);
            slideWindow();
        } else {
            ++frame_count;
            state_->Rs[frame_count] = state_->Rs[frame_count - 1];
            state_->Ps[frame_count] = state_->Ps[frame_count - 1];
            state_->Vs[frame_count] = state_->Vs[frame_count - 1];
            state_->Bas[frame_count] = state_->Bas[frame_count - 1];
            state_->Bgs[frame_count] = state_->Bgs[frame_count - 1];
            int next_idx = frame_count - 1;
            if (next_idx < static_cast<int>(preintegrators_.size())) {
                preintegrators_[next_idx].reset(
                    state_->Bas[frame_count - 1], state_->Bgs[frame_count - 1], noise_);
            }
        }
    } else {
        state_->acc_vec.erase(state_->acc_vec.begin());
        state_->gyro_vec.erase(state_->gyro_vec.begin());
        if (frame_count == state_->max_frame_count - 1 && gravity_initialized_) {
            optimize();
            feature_manager_->removeOutliers(*state_, params_.ric, params_.tic);
        }
        feature_manager_->removeNewest(frame_count);
    }

    if (cloud_callback_ && gravity_initialized_) {
        auto pts = feature_manager_->getPointCloud(*state_, params_.ric, params_.tic);
        cloud_callback_(ts, pts);
    }
    if (pose_callback_ && gravity_initialized_) {
        pose_callback_(
            ts,
            Sophus::SE3d(state_->Rs[state_->cur_frame_count], state_->Ps[state_->cur_frame_count]));
    }
}

void Estimator::optimize() {
    double delay_time_pre = state_->delay_time;

    state_->stateToParams();
    auto features = feature_manager_->collectLandmarks();

    ceres::Problem problem;

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->params_pose[i].data(), 6, se3_manifold);
        problem.AddParameterBlock(state_->params_speed_bias[i].data(), 9);
    }

    problem.AddParameterBlock(&state_->param_delay_time, 1);
    bool set_dt_constant_flag = true;
    for (int i = 0; i < state_->cur_frame_count; ++i) {
        if (state_->gyro_vec[i].norm() > params_.dt_gyro_threshold) {
            set_dt_constant_flag = false;
            break;
        }
    }

    if (set_dt_constant_flag) {
        problem.SetParameterBlockConstant(&state_->param_delay_time);
    }

    if (marg_data_) {
        auto* prior_cost = new MarginalizationPriorFactor(*marg_data_);
        std::vector<double*> prior_blocks;
        int num_kept = static_cast<int>(marg_data_->linearization_poses.size());
        for (int i = 0; i < num_kept; ++i) {
            prior_blocks.push_back(state_->params_pose[i].data());
            prior_blocks.push_back(state_->params_speed_bias[i].data());
        }
        problem.AddResidualBlock(prior_cost, nullptr, prior_blocks);
    }

    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < params_.max_depth) ? (1.0 / d) : 1.0;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
    }

    Eigen::Matrix2d sqrt_info = state_->visual_sqrt_info;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        int host_id = f->start_frame_id;
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            int target_id = host_id + static_cast<int>(obs_idx);
            Eigen::Vector2d pt_j(f->observations[obs_idx].pt.x, f->observations[obs_idx].pt.y);
            auto* cost = new VisualFactor(
                f->observations[0].uv, pt_j, params_.ric, params_.tic, state_->gyro_vec[host_id],
                state_->gyro_vec[target_id], state_->acc_vec[host_id], state_->acc_vec[target_id],
                state_->params_speed_bias[host_id].data(),
                state_->params_speed_bias[target_id].data(),
                state_->params_speed_bias[host_id].data() + 6,
                state_->params_speed_bias[target_id].data() + 6,
                state_->params_speed_bias[host_id].data() + 3,
                state_->params_speed_bias[target_id].data() + 3, sqrt_info, state_->camera);
            problem.AddResidualBlock(
                cost, loss, state_->params_pose[host_id].data(),
                state_->params_pose[target_id].data(), &state_->param_delay_time,
                &inv_depth_params[k]);
        }
    }

    for (int i = 0; i < state_->cur_frame_count; ++i) {
        if (preintegrators_[i].buffer.size() < 2) continue;
        auto pint_ptr =
            std::shared_ptr<MidPointIntegrator>(&preintegrators_[i], [](MidPointIntegrator*) {});
        auto* imu_cost = new IMUFactor<MidPointIntegrator>(pint_ptr);
        problem.AddResidualBlock(
            imu_cost, nullptr, state_->params_pose[i].data(), state_->params_speed_bias[i].data(),
            state_->params_pose[i + 1].data(), state_->params_speed_bias[i + 1].data());
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = params_.num_iterations;
    opts.num_threads = params_.num_threads;
    opts.minimizer_progress_to_stdout = false;
    opts.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    state_->paramsToState();

    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
        features[k]->estimated_depth = d;
    }

    for (int i = 0; i < preintegrators_.size(); ++i) {
        preintegrators_[i].repropagate(state_->Bas[i], state_->Bgs[i], noise_);
    }

    spdlog::info(
        "Ba {:.4f} {:.4f} {:.4f} Bg {:.4f} {:.4f} {:.4f} delay_time {:.2f} ms", state_->Bas[0].x(),
        state_->Bas[0].y(), state_->Bas[0].z(), state_->Bgs[0].x(), state_->Bgs[0].y(),
        state_->Bgs[0].z(), state_->param_delay_time * 1e3);
}

void Estimator::buildPrior() {
    const int n = state_->max_frame_count;
    state_->stateToParams();

    auto marg_features = feature_manager_->collectMargFeatures();
    spdlog::debug("{} feature marginals", static_cast<int>(marg_features.size()));
    std::vector<IntegratorBase<MidPointIntegrator>*> pint_ptrs;
    pint_ptrs.push_back(&preintegrators_[0]);

    auto marg = MarginlizationSqrt<MidPointIntegrator>(
        std::move(marg_features), std::make_unique<ceres::HuberLoss>(1.0), state_, pint_ptrs,
        params_.ric, params_.tic, marg_data_.get());
    marg.allocate();
    marg.linearize();
    marg.performQRAll();

    Eigen::MatrixXd Q2Jp;
    Eigen::VectorXd Q2r;
    marg.get_dense_Jp_b(Q2Jp, Q2r);

    Eigen::MatrixXd marg_H;
    Eigen::VectorXd marg_b;
    MargHelper::marginalizeSqrtToSqrt(15, (n - 1) * 15, Q2Jp, Q2r, marg_H, marg_b);

    auto new_prior = std::make_unique<MargLinData>();
    new_prior->H = marg_H;
    new_prior->b = marg_b;
    new_prior->linearization_poses.resize(n - 1);
    new_prior->linearization_speed_bias.resize(n - 1);
    for (int i = 1; i < n; ++i) {
        new_prior->linearization_poses[i - 1] = state_->params_pose[i];
        new_prior->linearization_speed_bias[i - 1] = state_->params_speed_bias[i];
    }
    marg_data_ = std::move(new_prior);
    feature_manager_->removeMargFeatures();
    spdlog::debug("Built prior: {} residuals, {} kept frames", marg_b.size(), n - 1);
}

void Estimator::slideWindow() {
    const int n = state_->max_frame_count;

    for (int i = 0; i < n - 1; ++i) {
        state_->Rs[i] = state_->Rs[i + 1];
        state_->Ps[i] = state_->Ps[i + 1];
        state_->Vs[i] = state_->Vs[i + 1];
        state_->Bas[i] = state_->Bas[i + 1];
        state_->Bgs[i] = state_->Bgs[i + 1];
    }
    // 滑动预积分器
    for (int i = 0; i < static_cast<int>(preintegrators_.size()) - 1; ++i) {
        preintegrators_[i] = std::move(preintegrators_[i + 1]);
    }
    // 重置最后一个预积分器
    preintegrators_.back().reset(state_->Bas[n - 2], state_->Bgs[n - 2], noise_);

    state_->acc_vec.erase(state_->acc_vec.begin());
    state_->gyro_vec.erase(state_->gyro_vec.begin());
}

Eigen::Matrix<double, 18, 18> Estimator::initNoise() const {
    Eigen::Matrix<double, 18, 18> noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (params_.acc_n * params_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (params_.gyr_n * params_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (params_.acc_n * params_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (params_.gyr_n * params_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (params_.acc_w * params_.acc_w) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (params_.gyr_w * params_.gyr_w) * Eigen::Matrix3d::Identity();
    return noise;
}

bool Estimator::tryInitialize() {
    int frame_count = state_->cur_frame_count;
    int n_frames = frame_count + 1;

    InitialSFM sfm(
        params_.min_depth, params_.max_depth, params_.sfm_min_seed_pts, params_.sfm_min_e_inliers,
        params_.sfm_e_ransac_threshold, params_.sfm_min_pnp_pts, params_.sfm_pnp_reproj_threshold,
        params_.sfm_max_bad_pnp_ratio, params_.sfm_ba_max_iterations, params_.sfm_ba_num_threads);
    if (!sfm.construct(
            *state_, *feature_manager_, params_.ric, params_.tic, params_.ric1, params_.tic1, Rs_,
            Ps_)) {
        spdlog::warn("VIO init: SFM failed");
        return false;
    }

    {
        std::vector<Eigen::Matrix3d> dq_dbgs, delta_qs;
        for (int i = 0; i < frame_count; ++i) {
            dq_dbgs.push_back(preintegrators_[i].get_dq_dbg());
            delta_qs.push_back(preintegrators_[i].final_delta_q);
        }
        Eigen::Vector3d bg = solveGyroBias(Rs_, dq_dbgs, delta_qs, params_.ric);
        for (int i = 0; i < frame_count; ++i) state_->Bgs[i] = bg;
    }

    for (int i = 0; i < frame_count; ++i) {
        preintegrators_[i].repropagate(state_->Bas[i], state_->Bgs[i], noise_);
    }

    std::vector<Eigen::Vector3d> delta_ps, delta_vs;
    std::vector<double> dts;
    for (int i = 0; i < n_frames - 1; ++i) {
        delta_ps.push_back(preintegrators_[i].final_delta_p);
        delta_vs.push_back(preintegrators_[i].final_delta_v);
        dts.push_back(preintegrators_[i].sum_dt);
    }

    Eigen::Vector3d g;
    double s;
    if (!linearAlignment(
            Rs_, Ps_, Vs_, delta_vs, delta_ps, dts, g, s, params_.ric, params_.tic,
            params_.gravity_diff_threshold, params_.g_norm)) {
        spdlog::warn("VI init: linear alignment failed");
        return false;
    }

    refineGravitySpeeds(
        Vs_, Rs_, Ps_, delta_vs, delta_ps, dts, g, s, params_.ric, params_.tic, params_.g_norm);

    Eigen::Matrix3d R0 =
        Eigen::Quaterniond::FromTwoVectors((-g).normalized(), Eigen::Vector3d(0, 0, 1))
            .toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
    for (int i = 0; i <= frame_count; ++i) {
        state_->Rs[i] = Eigen::Quaterniond(R0 * params_.ric * Rs_[i] * params_.ric.transpose())
                            .normalized()
                            .toRotationMatrix();
        state_->Ps[i] =
            R0 * (params_.ric * s * Ps_[i] + params_.ric * Rs_[i] * params_.tic - params_.tic);
        state_->Vs[i] = R0 * Vs_[i];
    }
    tassel_utils::G = Eigen::Vector3d(0, 0, params_.g_norm);
    gravity_initialized_ = true;

    spdlog::info("VI init: |g|={:.4f} s={:.4f}", tassel_utils::G.norm(), s);
    return true;
}

}  // namespace tassel_core
