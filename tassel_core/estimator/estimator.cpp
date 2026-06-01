#include "estimator.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <spdlog/spdlog.h>
#include <cstddef>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "factor/imu_factor.h"
#include "factor/integrator_base.h"
#include "factor/marg_helper.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/marginlization_sqrt.h"
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
      tic1_(tic1) {
    cv::setNumThreads(option_.num_threads);
    noise_ = initNoise();
    preintegrators_.resize(
        state_->max_frame_count - 1,
        MidPointIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_));
    state_->cur_frame_count = 0;
    state_->ric = ric_;
    state_->tic = tic_;
    Rs_.resize(state_->max_frame_count);
    Ps_.resize(state_->max_frame_count);
    Rs_[0] = Eigen::Matrix3d::Identity();
    Ps_[0] = Eigen::Vector3d::Zero();
}

void Estimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    int& frame_count = state_->cur_frame_count;

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

    // if (last_ts_ < 0 && !imu_measurements.empty()) {
    //     last_ts_ = imu_measurements.back().timestamp;
    //     last_imu_acc_ = imu_measurements.back().acc;
    //     last_imu_gyro_ = imu_measurements.back().gyro;
    // }

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

        // if (!gravity_initialized_) {
        //     feature_manager_->initPoseByPNP(frame_count, Rs_, Ps_);
        // }

        feature_manager_->triangulate(*state_, ric1_, tic1_);
        if (frame_count == state_->max_frame_count - 1) {
            // if (!gravity_initialized_) {
            //     solveGyroBias();
            //     for (int i = 0; i < state_->cur_frame_count; ++i) {
            //         preintegrators_[i].reintegrate(Eigen::Vector3d::Zero(), state_->Bgs[i],
            //         noise_);
            //     }
            //     Eigen::Vector3d G_body;
            //     linearAlignment(G_body);
            //     G_body = G_body.normalized() * tassel_utils::G.norm();
            //     refineGravity(G_body);

            //     Eigen::Matrix3d R0 = Eigen::Quaterniond::FromTwoVectors(
            //                              G_body.normalized(), Eigen::Vector3d(0, 0, 1))
            //                              .toRotationMatrix();
            //     double yaw = std::atan2(R0(1, 0), R0(0, 0));
            //     R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
            //     for (int i = 0; i <= frame_count; ++i) {
            //         Ps_[i] = R0 * Ps_[i];
            //         Rs_[i] = R0 * Rs_[i];
            //         state_->Rs[i] = Rs_[i] * ric_.transpose();
            //         state_->Ps[i] = Ps_[i] - state_->Rs[i] * tic_;
            //         state_->Vs[i] = R0 * state_->Vs[i];
            //     }
            //     gravity_initialized_ = true;
            //     spdlog::info("Gravity aligned via visual-inertial initialization");
            // }
            optimize();
            feature_manager_->removeOutliers(*state_);
            buildPrior();
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
        // slideWindow(false);
    }

    int latest_idx = state_->cur_frame_count - 1 == state_->max_frame_count
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
        // problem.SetParameterBlockConstant(&inv_depth_params[k]);
    }

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->params_pose[i].data(), 6, se3_manifold);
        problem.AddParameterBlock(state_->params_speed_bias[i].data(), 9);
    }

    problem.AddParameterBlock(&state_->param_delay_time, 1);
    bool set_dt_constant_flag = true;
    for (auto V : state_->Vs) {
        if (V.norm() > 0.3) {
            set_dt_constant_flag = false;
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

    // ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
    Eigen::Matrix2d sqrt_info = state_->visual_sqrt_info;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        int host_id = f->start_frame_id;
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            int target_id = host_id + static_cast<int>(obs_idx);
            auto* cost = new VisualFactor(
                f->observations[0].uv, f->observations[obs_idx].uv, ric_, tic_,
                state_->gyro_vec[host_id], state_->gyro_vec[target_id], state_->acc_vec[host_id],
                state_->acc_vec[target_id], state_->params_speed_bias[host_id].data(),
                state_->params_speed_bias[target_id].data(),
                state_->params_speed_bias[host_id].data() + 6,
                state_->params_speed_bias[target_id].data() + 6, sqrt_info);
            problem.AddResidualBlock(
                cost, nullptr, state_->params_pose[host_id].data(),
                state_->params_pose[target_id].data(), &inv_depth_params[k],
                &state_->param_delay_time);
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
    opts.max_num_iterations = option_.num_iterations;
    opts.num_threads = option_.num_threads;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    state_->paramsToState();
    std::vector<double> depths;
    depths.reserve(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
        features[k]->estimated_depth = d;
        if (d > 0) depths.push_back(d);
    }

    for (int i = 0; i < preintegrators_.size(); ++i) {
        preintegrators_[i].reintegrate(state_->Bas[i], state_->Bgs[i], noise_);
    }

    spdlog::info(
        "Ba {:.4f} {:.4f} {:.4f} Bg {:.4f} {:.4f} {:.4f} delay_time {:.2f} ms", state_->Bas[0].x(),
        state_->Bas[0].y(), state_->Bas[0].z(), state_->Bgs[0].x(), state_->Bgs[0].y(),
        state_->Bgs[0].z(), state_->param_delay_time * 1e3);
}

void Estimator::buildPrior() {
    const int n = state_->max_frame_count;
    state_->stateToParams();

    std::vector<IntegratorBase<MidPointIntegrator>*> pint_ptrs;
    pint_ptrs.reserve(preintegrators_.size());
    for (auto& p : preintegrators_) {
        pint_ptrs.push_back(&p);
    }

    auto marg = MarginlizationSqrt<MidPointIntegrator>(
        feature_manager_, nullptr, state_, pint_ptrs, marg_data_.get());
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
    feature_manager_->removeMarginalizedFeatures();
    spdlog::info("Built prior: {} residuals, {} kept frames", marg_b.size(), n - 1);
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

void Estimator::solveGyroBias() {
    // 忽略时间延迟的影响，使用一阶线性化估计偏置，需要角速度激励充足，否则估计结果不可靠
    // 影响ceres优化的初始值
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A.setZero();
    b.setZero();
    for (int i = 0; i < state_->cur_frame_count; ++i) {
        Eigen::Matrix3d R_imu_i = Rs_[i];
        Eigen::Matrix3d R_imu_j = Rs_[i + 1];
        Eigen::Matrix3d R_ij = R_imu_i.transpose() * R_imu_j;
        auto preintegrator = preintegrators_[i];
        Eigen::Matrix3d R_diff = preintegrator.final_delta_q.transpose() * R_ij;
        Eigen::Quaterniond q_diff(R_diff);
        q_diff.normalize();
        Eigen::Vector3d phi = Sophus::SO3d(q_diff).log();
        A += preintegrator.get_dq_dbg();
        b += phi;
    }
    Eigen::Vector3d bg = A.ldlt().solve(b);
    for (auto& bg_i : state_->Bgs) {
        bg_i += bg;
    }
    spdlog::info("Gyro bias solved: ({:.6f}, {:.6f}, {:.6f})", bg.x(), bg.y(), bg.z());
}

bool Estimator::linearAlignment(Eigen::Vector3d& G_body) {
    int n = state_->max_frame_count;
    int n_pairs = state_->cur_frame_count;

    int rows = 6 * n_pairs;
    int cols = 3 * n + 3;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

    for (int i = 0; i < n_pairs; ++i) {
        double dt = preintegrators_[i].sum_dt;
        Eigen::Matrix3d R_imu_i = Rs_[i] * ric_.transpose();
        const Eigen::Matrix3d Ri_T = R_imu_i.transpose();
        const Eigen::Vector3d Pi = Ps_[i] - R_imu_i * tic_;
        const Eigen::Vector3d Pj = Ps_[i + 1] - Rs_[i + 1] * ric_.transpose() * tic_;
        const auto& pint = preintegrators_[i];

        // δp: R_i^T (P_j - P_i - V_i*dt + 0.5*g*dt²) = δp_measured
        int row = 6 * i;
        A.block<3, 3>(row, 3 * i) = -Ri_T * dt;
        A.block<3, 3>(row, cols - 3) = Ri_T * (0.5 * dt * dt);
        b.segment<3>(row) = pint.final_delta_p - Ri_T * (Pj - Pi);

        // δv: R_i^T (V_j - V_i + g*dt) = δv_measured
        A.block<3, 3>(row + 3, 3 * i) = -Ri_T;
        A.block<3, 3>(row + 3, 3 * (i + 1)) = Ri_T;
        A.block<3, 3>(row + 3, cols - 3) = Ri_T * dt;
        b.segment<3>(row + 3) = pint.final_delta_v;
    }

    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    for (int i = 0; i < n; ++i) {
        state_->Vs[i] = x.segment<3>(3 * i);
    }
    G_body = x.segment<3>(3 * n);

    spdlog::info(
        "LinearAlignment: |g|={:.4f} g=({:.3f},{:.3f},{:.3f})", G_body.norm(), G_body.x(),
        G_body.y(), G_body.z());
    return true;
}

void Estimator::refineGravity(Eigen::Vector3d& G_body) {
    const double g_mag = tassel_utils::G.norm();
    Eigen::Vector3d g_dir = G_body.normalized();

    int n = state_->max_frame_count;
    int n_pairs = state_->cur_frame_count;

    for (int iter = 0; iter < 4; ++iter) {
        Eigen::Matrix<double, 2, 3> tangent_base = computeTangentBasis(g_dir);
        Eigen::Matrix<double, 3, 2> dg_dw = tangent_base.transpose();

        // x = [V_0, ..., V_n, w1, w2]  where g = g_mag*ĝ + w1*b1 + w2*b2
        int rows = 6 * n_pairs;
        int cols = 3 * n + 2;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

        for (int i = 0; i < n_pairs; ++i) {
            double dt = preintegrators_[i].sum_dt;
            Eigen::Matrix3d R_imu_i = Rs_[i] * ric_.transpose();
            const Eigen::Matrix3d Ri_T = R_imu_i.transpose();
            const Eigen::Vector3d Pi = Ps_[i] - R_imu_i * tic_;
            const Eigen::Vector3d Pj = Ps_[i + 1] - Rs_[i + 1] * ric_.transpose() * tic_;
            const auto& pint = preintegrators_[i];

            int row = 6 * i;
            A.block<3, 3>(row, 3 * i) = -Ri_T * dt;
            A.block<3, 2>(row, cols - 2) = Ri_T * (0.5 * dt * dt) * dg_dw;
            b.segment<3>(row) =
                pint.final_delta_p - Ri_T * (Pj - Pi) - Ri_T * g_mag * g_dir * (0.5 * dt * dt);

            A.block<3, 3>(row + 3, 3 * i) = -Ri_T;
            A.block<3, 3>(row + 3, 3 * (i + 1)) = Ri_T;
            A.block<3, 2>(row + 3, cols - 2) = Ri_T * dt * dg_dw;
            b.segment<3>(row + 3) = pint.final_delta_v - Ri_T * g_mag * g_dir * dt;
        }

        Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

        for (int i = 0; i < n; ++i) {
            state_->Vs[i] = x.segment<3>(3 * i);
        }

        Eigen::Vector2d w = x.segment<2>(3 * n);
        g_dir = (g_mag * g_dir + dg_dw * w).normalized();
        G_body = g_mag * g_dir;
    }

    spdlog::info(
        "RefineGravity: |g|={:.4f} g=({:.3f},{:.3f},{:.3f})", G_body.norm(), G_body.x(), G_body.y(),
        G_body.z());
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

    Eigen::Vector3d ng1 = avg_acc.normalized();
    Eigen::Vector3d ng2(0.0, 0.0, 1.0);
    Eigen::Matrix3d R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
    state_->Rs[0] = R0;
    state_->Ps[0] = Eigen::Vector3d::Zero();
    state_->Vs[0] = Eigen::Vector3d::Zero();
    // state_->Bas[0] = avg_acc - R0.transpose() * tassel_utils::G;
    state_->Bgs[0] = avg_gyro;
    last_ts_ = imu_measurements.back().timestamp;
    last_imu_acc_ = imu_measurements.back().acc;
    last_imu_gyro_ = imu_measurements.back().gyro;
    imu_initialized_ = true;
    spdlog::info(
        "IMU static init: Bg=({:.4f},{:.4f},{:.4f}) rad/s  Ba=({:.4f},{:.4f},{:.4f}) m/s²",
        avg_gyro.x(), avg_gyro.y(), avg_gyro.z(), state_->Bas[0].x(), state_->Bas[0].y(),
        state_->Bas[0].z());
}
}  // namespace tassel_core
