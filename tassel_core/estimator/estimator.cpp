#include "estimator.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <set>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/se3_right_manifold.h"
#include "factor/visual_factor.h"

namespace tassel_core {

Estimator::Estimator(
    const EstimatorOption& option, std::shared_ptr<State> state, std::shared_ptr<FeatureManager> fm,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1,
    const Eigen::Vector3d& tic1)
    : option_(option),
      state_(std::move(state)),
      feature_manager_(std::move(fm)),
      ric_(ric),
      tic_(tic),
      ric1_(ric1),
      tic1_(tic1),
      imu_initialized_(false),
      init_ts_(-1) {
    cv::setNumThreads(option_.num_threads);
    noise_ = initNoise();
    preintegrators_.resize(
        state_->max_frame_count - 1,
        MidPointIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_));
    state_->cur_frame_count = 0;
    state_->ric = ric_;
    state_->tic = tic_;
}

void Estimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    if (!imu_initialized_) {
        if (init_ts_ < 0) {
            init_ts_ = ts;
        }
        imu_init_buf_.insert(imu_init_buf_.end(), imu_measurements.begin(), imu_measurements.end());
        if (ts - init_ts_ >= option_.init_time_span) {
            initializeImu(imu_init_buf_);
        }
        return;
    }

    int& frame_count = state_->cur_frame_count;
    bool is_keyframe = feature_manager_->checkKeyFrameByParallax(frame_count, feature_frame);

    if (frame_count > 0) {
        Eigen::Matrix3d R = state_->Rs[frame_count];
        Eigen::Vector3d P = state_->Ps[frame_count];
        Eigen::Vector3d V = state_->Vs[frame_count];
        Eigen::Vector3d Ba = state_->Bas[frame_count];
        Eigen::Vector3d Bg = state_->Bgs[frame_count];
        auto& preintegrator = preintegrators_[frame_count - 1];

        for (const auto& imu : imu_measurements) {
            preintegrator.update(imu);

            double dt = imu.timestamp - last_ts_;
            Eigen::Vector3d acc_0 = R * (last_imu_acc_ - Ba) - tassel_utils::G;
            Eigen::Vector3d gyr = 0.5 * (last_imu_gyro_ + imu.gyro) - Bg;
            R = R * Sophus::SO3d::exp(gyr * dt).matrix();
            Eigen::Vector3d acc_1 = R * (imu.acc - Ba) - tassel_utils::G;
            Eigen::Vector3d acc = 0.5 * (acc_0 + acc_1);
            P += V * dt + 0.5 * acc * dt * dt;
            V += acc * dt;
            last_ts_ = imu.timestamp;
            last_imu_gyro_ = imu.gyro;
            last_imu_acc_ = imu.acc;
        }

        Eigen::Quaterniond q(R);
        q.normalize();
        state_->Rs[frame_count] = q.matrix();
        state_->Ps[frame_count] = P;
        state_->Vs[frame_count] = V;
    }
    if (is_keyframe) {
        state_->acc_vec.push_back(last_imu_acc_);
        state_->gyro_vec.push_back(last_imu_gyro_);

        feature_manager_->triangulate(*state_, ric1_, tic1_);
        if (frame_count == state_->max_frame_count - 1) {
            optimize();
            feature_manager_->removeMarginalizedFeatures();
            feature_manager_->removeOldest(*state_);
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
        // if (frame_count == state_->max_frame_count - 1) {
        // optimize();
        // }
        feature_manager_->removeNewest(frame_count);
    }

    if (state_->cur_frame_count > 0) {
        int latest_idx = state_->cur_frame_count == state_->max_frame_count
                             ? state_->cur_frame_count
                             : state_->cur_frame_count - 1;

        Sophus::SE3d latest_pose(state_->Rs[latest_idx], state_->Ps[latest_idx]);

        if (pose_callback_) {
            pose_callback_(ts, latest_pose);
        }
        if (mono_cloud_callback_) {
            auto pts = feature_manager_->getMonoPointCloud(*state_);
            mono_cloud_callback_(ts, pts);
        }
        if (stereo_cloud_callback_) {
            auto pts = feature_manager_->getStereoPointCloud(*state_);
            stereo_cloud_callback_(ts, pts);
        }
    }
}

void Estimator::optimize() {
    double delay_time_pre = state_->delay_time;

    state_->stateToParams();
    state_->saveStateWithLinearized();
    auto features = feature_manager_->collectOptimizedFeatures();

    ceres::Problem problem;

    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < option_.max_depth) ? (1.0 / d) : 1.0;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
    }

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->params_pose[i].data(), 6, se3_manifold);
        problem.AddParameterBlock(state_->params_speed_bias[i].data(), 9);
    }

    problem.AddParameterBlock(&state_->param_delay_time, 1);

    problem.SetParameterBlockConstant(state_->params_pose[0].data());

    ceres::LossFunction* loss = new ceres::HuberLoss(0.005);
    Eigen::Matrix2d sqrt_info = Eigen::Matrix2d::Identity() * 320;
    std::set<int> involved_indices;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        int host_id = f->start_frame_id;
        involved_indices.insert(host_id);
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            int target_id = host_id + static_cast<int>(obs_idx);
            involved_indices.insert(target_id);
            auto* cost = new VisualFactor(
                f->observations[0].uv, f->observations[obs_idx].uv, ric_, tic_,
                state_->gyro_vec[host_id], state_->gyro_vec[target_id], state_->acc_vec[host_id],
                state_->acc_vec[target_id], state_->params_speed_bias[host_id].data(),
                state_->params_speed_bias[target_id].data(),
                state_->params_speed_bias[host_id].data() + 6,
                state_->params_speed_bias[target_id].data() + 6, sqrt_info);
            problem.AddResidualBlock(
                cost, loss, state_->params_pose[host_id].data(),
                state_->params_pose[target_id].data(), &inv_depth_params[k],
                &state_->param_delay_time);
        }
    }

    if (static_cast<int>(involved_indices.size()) == state_->max_frame_count) {
        spdlog::info("Visual factors cover all {} frames in the window", state_->max_frame_count);
    }

    int imu_count = 0;
    for (int i = 0; i < state_->cur_frame_count; ++i) {
        if (preintegrators_[i].buffer.size() < 2) continue;
        auto pint_ptr =
            std::shared_ptr<MidPointIntegrator>(&preintegrators_[i], [](MidPointIntegrator*) {});
        auto* imu_cost = new IMUFactor<MidPointIntegrator>(pint_ptr);
        problem.AddResidualBlock(
            imu_cost, nullptr, state_->params_pose[i].data(), state_->params_speed_bias[i].data(),
            state_->params_pose[i + 1].data(), state_->params_speed_bias[i + 1].data());
        ++imu_count;
    }
    spdlog::info("Added {} IMU factors", imu_count);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = option_.num_iterations;
    opts.num_threads = option_.num_threads;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    state_->paramsToState();

    for (int i = 0; i < state_->cur_frame_count; ++i) {
        preintegrators_[i].reintegrate(state_->Bas[i], state_->Bgs[i], noise_);
    }

    std::vector<double> depths;
    depths.reserve(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
        features[k]->estimated_depth = d;
        if (d > 0) depths.push_back(d);
    }
}

void Estimator::initializeImu(const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d avg_gyro = Eigen::Vector3d::Zero();
    for (const auto& m : imu_measurements) {
        avg_acc += m.acc;
        avg_gyro += m.gyro;
    }
    avg_acc /= static_cast<double>(imu_measurements.size());
    avg_gyro /= static_cast<double>(imu_measurements.size());

    // VINS-Fusion style gravity alignment: align to +Z then zero yaw
    Eigen::Vector3d ng1 = avg_acc.normalized();
    Eigen::Vector3d ng2(0.0, 0.0, 1.0);
    Eigen::Matrix3d R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
    state_->Rs[0] = R0;
    state_->Ps[0] = Eigen::Vector3d::Zero();
    state_->Vs[0] = Eigen::Vector3d::Zero();
    state_->Bas[0] = avg_acc - R0.transpose() * tassel_utils::G;
    state_->Bgs[0] = avg_gyro;
    last_ts_ = imu_measurements.back().timestamp;
    last_imu_acc_ = imu_measurements.back().acc;
    last_imu_gyro_ = imu_measurements.back().gyro;
    imu_initialized_ = true;
    spdlog::info("IMU gravity initialized");
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
    preintegrators_.back().reset(state_->Bas[n - 1], state_->Bgs[n - 1], noise_);

    state_->acc_vec.erase(state_->acc_vec.begin());
    state_->gyro_vec.erase(state_->gyro_vec.begin());
}

Eigen::Matrix<double, 18, 18> Estimator::initNoise() const {
    Eigen::Matrix<double, 18, 18> noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (option_.acc_n * option_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (option_.gyr_n * option_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (option_.acc_n * option_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (option_.gyr_n * option_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (option_.acc_w * option_.acc_w) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (option_.gyr_w * option_.gyr_w) * Eigen::Matrix3d::Identity();
    return noise;
}

}  // namespace tassel_core
