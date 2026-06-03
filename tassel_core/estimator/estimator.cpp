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
namespace {

// ── Ceres cost for VI initialization ──────────────────────────────────────
//
// Residual per adjacent frame pair (i, i+1) — 6D:
//   r_p = R_iᵀ(Pⱼ-Pᵢ-Vᵢ·dt-½g·dt²) - δp
//   r_v = R_iᵀ(Vⱼ-Vᵢ-g·dt)       - δv
//
// Parameters: [V_i, V_j, w(2-DOF gravity tangent)]  — Ba deferred to backend

struct ViInitResidual {
    Eigen::Vector3d dp_meas, dv_meas;
    Eigen::Vector3d Pi, Pj;
    Eigen::Matrix3d Ri_T;
    double sum_dt;
    double g_mag;
    Eigen::Matrix<double, 3, 2> dg_dw;
    Eigen::Vector3d g0_dir;

    ViInitResidual(
        const MidPointIntegrator& pint, const Eigen::Vector3d& P_i, const Eigen::Vector3d& P_j,
        const Eigen::Matrix3d& R_i, const Eigen::Matrix<double, 2, 3>& tangent_base,
        const Eigen::Vector3d& g_dir, double gm, double dt)
        : dp_meas(pint.final_delta_p),
          dv_meas(pint.final_delta_v),
          Pi(P_i),
          Pj(P_j),
          Ri_T(R_i.transpose()),
          sum_dt(dt),
          g_mag(gm),
          dg_dw(tangent_base.transpose()),
          g0_dir(g_dir) {}

    template <typename T>
    bool operator()(const T* const V_i, const T* const V_j, const T* const w, T* residual) const {
        // Gravity on sphere: g = g_mag * (g0_dir + dg/dw * w) / |·|
        T g_dir_T[3], g[3];
        for (int k = 0; k < 3; ++k)
            g_dir_T[k] = T(g0_dir[k]) + w[0] * T(dg_dw(k, 0)) + w[1] * T(dg_dw(k, 1));
        T inv_norm = ceres::sqrt(
            g_dir_T[0] * g_dir_T[0] + g_dir_T[1] * g_dir_T[1] + g_dir_T[2] * g_dir_T[2]);
        T gm = T(g_mag);
        for (int k = 0; k < 3; ++k) g[k] = gm * g_dir_T[k] / inv_norm;

        // Predicted delta (no bias correction — Ba deferred to backend)
        T dt = T(sum_dt);
        T half_dt2 = T(0.5) * dt * dt;

        T pred_p_w[3], pred_v_w[3];
        for (int c = 0; c < 3; ++c) {
            pred_p_w[c] = T(Pj[c]) - T(Pi[c]) - V_i[c] * dt - half_dt2 * g[c];
            pred_v_w[c] = V_j[c] - V_i[c] - dt * g[c];
        }
        for (int r = 0; r < 3; ++r) {
            T rot_p = T(0), rot_v = T(0);
            for (int c = 0; c < 3; ++c) {
                rot_p += T(Ri_T(r, c)) * pred_p_w[c];
                rot_v += T(Ri_T(r, c)) * pred_v_w[c];
            }
            residual[r] = rot_p - T(dp_meas[r]);
            residual[r + 3] = rot_v - T(dv_meas[r]);
        }
        return true;
    }
};

// Check if ≥ min_frames preintegrators show enough rotation
bool hasEnoughExcitation(
    const std::vector<MidPointIntegrator>& preints, int n_pairs, int min_frames,
    double rot_thresh_rad) {
    int count = 0;
    for (int i = 0; i < n_pairs; ++i) {
        double angle = Sophus::SO3d(preints[i].final_delta_q).log().norm();
        if (angle > rot_thresh_rad) ++count;
    }
    return count >= min_frames;
}

}  // namespace

// ── Estimator public interface ─────────────────────────────────────────────

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
    Grav_ = Eigen::Vector3d(0, 0, option_.g_norm);
    tassel_utils::G = Grav_;
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

void Estimator::resetInit() {
    gravity_initialized_ = false;
    imu_initialized_ = false;
    init_ts_ = -1;
    last_ts_ = -1;
    last_imu_acc_ = Eigen::Vector3d::Zero();
    last_imu_gyro_ = Eigen::Vector3d::Zero();
    imu_init_buf_.clear();
    preintegrators_.clear();
    preintegrators_.resize(
        state_->max_frame_count - 1,
        MidPointIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_));
    Rs_.clear();
    Ps_.clear();
    Rs_.resize(state_->max_frame_count);
    Ps_.resize(state_->max_frame_count);
    Rs_[0] = Eigen::Matrix3d::Identity();
    Ps_[0] = Eigen::Vector3d::Zero();
    marg_data_.reset();
    is_first_optimization_ = true;
    Grav_ = Eigen::Vector3d(0, 0, option_.g_norm);
    tassel_utils::G = Grav_;
    state_->reset();
    feature_manager_->reset();
}

void Estimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    int& frame_count = state_->cur_frame_count;

    if (gravity_initialized_) return;

    // if (!imu_initialized_) {
    //     if (init_ts_ < 0) {
    //         init_ts_ = ts;
    //     }
    //     imu_init_buf_.insert(imu_init_buf_.end(), imu_measurements.begin(),
    //     imu_measurements.end()); if (ts - init_ts_ >= option_.init_time_span) {
    //         initializeImu(imu_init_buf_);
    //     }
    //     return;
    // }

    if (last_ts_ < 0 && !imu_measurements.empty()) {
        //  initializeImu(imu_measurements);
        last_ts_ = imu_measurements.back().timestamp;
        last_imu_acc_ = imu_measurements.back().acc;
        last_imu_gyro_ = imu_measurements.back().gyro;
    }

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
            Eigen::Vector3d acc_0 = R * (last_imu_acc_ - Ba) - Grav_;
            Eigen::Vector3d gyr = 0.5 * (last_imu_gyro_ + imu.gyro) - Bg;
            R = R * Sophus::SO3d::exp(gyr * dt).matrix();
            Eigen::Vector3d acc_1 = R * (imu.acc - Ba) - Grav_;
            Eigen::Vector3d acc = 0.5 * (acc_0 + acc_1);
            P += V * dt + 0.5 * acc * dt * dt;
            V += acc * dt;
            last_ts_ = imu.timestamp;
            last_imu_gyro_ = imu.gyro;
            last_imu_acc_ = imu.acc;
            if (pose_callback_) {
                Eigen::Quaterniond q_pose(R);
                q_pose.normalize();
                pose_callback_(ts, Sophus::SE3d(q_pose.toRotationMatrix(), P));
            }
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

        if (!gravity_initialized_) {
            if (!feature_manager_->initPoseByPNP(frame_count, Rs_, Ps_)) {
                spdlog::warn("PnP failed, resetting initialization");
                resetInit();
                return;
            }
        }

        feature_manager_->triangulate(*state_, ric1_, tic1_);
        if (frame_count == state_->max_frame_count - 1) {
            if (!gravity_initialized_) {
                solveGyroBias();
                for (int i = 0; i < state_->cur_frame_count; ++i) {
                    preintegrators_[i].reintegrate(Eigen::Vector3d::Zero(), state_->Bgs[i], noise_);
                }

                // ── Rotation gating ─────────────────────────────────
                static constexpr double kMinRotRad = 0.2;  // ~12° per pair
                static constexpr int kMinExcitedFrames = 3;
                int n_pairs = state_->cur_frame_count;

                if (!hasEnoughExcitation(preintegrators_, n_pairs, kMinExcitedFrames, kMinRotRad)) {
                    spdlog::warn(
                        "Insufficient rotation excitation ({}/{} pairs > {:.0f}°). "
                        "Please move the device to initialize.",
                        std::count_if(
                            preintegrators_.begin(), preintegrators_.begin() + n_pairs,
                            [](const auto& p) {
                                return Sophus::SO3d(p.final_delta_q).log().norm() > kMinRotRad;
                            }),
                        n_pairs, kMinRotRad * 180.0 / M_PI);
                    // Reset to re‑collect with new motion
                    frame_count = 0;
                    state_->cur_frame_count = 0;
                    state_->acc_vec.clear();
                    state_->gyro_vec.clear();
                    Rs_.clear();
                    Ps_.clear();
                    Rs_.resize(state_->max_frame_count);
                    Ps_.resize(state_->max_frame_count);
                    feature_manager_->reset();
                    return;
                }

                // ── Ceres joint initialization: [V, g(2-DOF tangent)] ──
                int n_frames = n_pairs + 1;
                std::vector<std::array<double, 3>> V_blocks(n_frames);
                double w_block[2] = {0.0, 0.0};

                // Initial guesses
                {
                    // V: accumulate PnP position deltas
                    for (int i = 0; i < n_frames; ++i) {
                        V_blocks[i] = {0.0, 0.0, 0.0};
                    }
                    for (int i = 0; i < n_pairs; ++i) {
                        double dt = preintegrators_[i].sum_dt;
                        if (dt > 1e-6) {
                            Eigen::Vector3d dP_w = (Ps_[i + 1] - Ps_[i]) / dt;
                            for (int d = 0; d < 3; ++d) V_blocks[i][d] = dP_w[d];
                        }
                    }
                }

                // Initial gravity direction: rotate first acc to world, negate
                // acc_body ≈ -R^T * g_world + Ba, so g_world_dir ≈ average(-R_imu_i * acc_i)
                Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
                for (int i = 0; i < n_pairs; ++i) {
                    Eigen::Matrix3d R_imu_i = Rs_[i] * ric_.transpose();
                    avg_acc += R_imu_i * (-preintegrators_[i].buffer.front().acc);
                }
                avg_acc /= n_pairs;
                Eigen::Vector3d g0_dir = avg_acc.normalized();
                if (g0_dir.z() < 0) g0_dir = -g0_dir;

                double g_mag = Grav_.norm();

                // Iterative refinement with tangent-basis gravity
                for (int iter = 0; iter < 4; ++iter) {
                    Eigen::Matrix<double, 2, 3> tangent_base = computeTangentBasis(g0_dir);

                    ceres::Problem init_problem;
                    ceres::LossFunction* huber = new ceres::HuberLoss(1.0);

                    for (int i = 0; i < n_pairs; ++i) {
                        Eigen::Matrix3d R_imu_i = Rs_[i] * ric_.transpose();
                        Eigen::Vector3d Pi = Ps_[i] - R_imu_i * tic_;
                        Eigen::Vector3d Pj = Ps_[i + 1] - Rs_[i + 1] * ric_.transpose() * tic_;

                        auto* cost = new ceres::AutoDiffCostFunction<ViInitResidual, 6, 3, 3, 2>(
                            new ViInitResidual(
                                preintegrators_[i], Pi, Pj, R_imu_i, tangent_base, g0_dir, g_mag,
                                preintegrators_[i].sum_dt));
                        init_problem.AddResidualBlock(
                            cost, huber, V_blocks[i].data(), V_blocks[i + 1].data(), w_block);
                    }

                    ceres::Solver::Options opts;
                    opts.linear_solver_type = ceres::DENSE_SCHUR;
                    opts.max_num_iterations = 50;
                    opts.num_threads = option_.num_threads;
                    opts.minimizer_progress_to_stdout = false;
                    opts.logging_type = ceres::SILENT;

                    ceres::Solver::Summary summary;
                    ceres::Solve(opts, &init_problem, &summary);

                    // Recenter tangent at new optimal direction
                    Eigen::Vector3d dg =
                        tangent_base.transpose() * Eigen::Vector2d(w_block[0], w_block[1]);
                    g0_dir = (g0_dir + dg).normalized();
                    w_block[0] = w_block[1] = 0.0;

                    if (iter == 0) {
                        spdlog::info(
                            "VI Ceres init (iter {}): iters={} init_cost={:.2f} final_cost={:.2f}",
                            iter, summary.iterations.size(), summary.initial_cost,
                            summary.final_cost);
                    }
                }

                Eigen::Vector3d G_body = g_mag * g0_dir;
                spdlog::info(
                    "  |g|={:.4f} g=({:.3f},{:.3f},{:.3f})", G_body.norm(), G_body.x(), G_body.y(),
                    G_body.z());

                // Store results
                for (int i = 0; i < n_frames; ++i) {
                    state_->Vs[i] = Eigen::Vector3d(V_blocks[i][0], V_blocks[i][1], V_blocks[i][2]);
                }

                // ── Gravity alignment ──────────────────────────────
                // G_body points DOWN (gravity direction).
                // Align the UP direction (−G_body) to world +Z.
                Eigen::Matrix3d R0 = Eigen::Quaterniond::FromTwoVectors(
                                         (-G_body).normalized(), Eigen::Vector3d(0, 0, 1))
                                         .toRotationMatrix();
                double yaw = std::atan2(R0(1, 0), R0(0, 0));
                R0 = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
                for (int i = 0; i <= frame_count; ++i) {
                    Ps_[i] = R0 * Ps_[i];
                    Rs_[i] = R0 * Rs_[i];
                    Eigen::Quaterniond q_tmp(Rs_[i]);
                    q_tmp.normalize();
                    Rs_[i] = q_tmp.toRotationMatrix();
                    state_->Rs[i] = Rs_[i] * ric_.transpose();
                    Eigen::Quaterniond q_state(state_->Rs[i]);
                    q_state.normalize();
                    state_->Rs[i] = q_state.toRotationMatrix();
                    state_->Ps[i] = Ps_[i] - state_->Rs[i] * tic_;
                    state_->Vs[i] = R0 * state_->Vs[i];
                }
                Grav_ = Eigen::Vector3d(0, 0, G_body.norm());
                tassel_utils::G = Grav_;
                gravity_initialized_ = true;
                spdlog::info("Gravity aligned: |g|={:.4f}", Grav_.norm());
                spdlog::info(
                    "G_body(est): ({:.3f},{:.3f},{:.3f}) → aligned to Z-up", G_body.x(), G_body.y(),
                    G_body.z());
                spdlog::info(
                    "R0(yaw={:.1f}°):\n{:.4f} {:.4f} {:.4f}\n{:.4f} {:.4f} {:.4f}\n{:.4f} {:.4f} "
                    "{:.4f}",
                    yaw * 180.0 / M_PI, R0(0, 0), R0(0, 1), R0(0, 2), R0(1, 0), R0(1, 1), R0(1, 2),
                    R0(2, 0), R0(2, 1), R0(2, 2));
                for (int i = 0; i <= frame_count; ++i) {
                    spdlog::info(
                        "Frame {}: V=({:.3f},{:.3f},{:.3f}) P=({:.3f},{:.3f},{:.3f})", i,
                        state_->Vs[i].x(), state_->Vs[i].y(), state_->Vs[i].z(), state_->Ps[i].x(),
                        state_->Ps[i].y(), state_->Ps[i].z());
                }

                // Publish first-frame pose and point cloud for verification
                if (pose_callback_) {
                    Eigen::Quaterniond q0(state_->Rs[0]);
                    q0.normalize();
                    pose_callback_(ts, Sophus::SE3d(q0.toRotationMatrix(), state_->Ps[0]));
                }
                if (cloud_callback_) {
                    auto pts = feature_manager_->getPointCloud(*state_);
                    cloud_callback_(ts, pts);
                }
                return;
            }
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
        state_->acc_vec.erase(state_->acc_vec.begin());
        state_->gyro_vec.erase(state_->gyro_vec.begin());
        state_->acc_vec.push_back(last_imu_acc_);
        state_->gyro_vec.push_back(last_imu_gyro_);
        feature_manager_->removeNewest(frame_count);
        // slideWindow(false);
    }

    if (cloud_callback_ && gravity_initialized_) {
        auto pts = feature_manager_->getPointCloud(*state_);
        cloud_callback_(ts, pts);
    }
}

void Estimator::optimize() {
    double delay_time_pre = state_->delay_time;

    state_->stateToParams();
    state_->saveStateWithLinearized();
    auto features = feature_manager_->collectOptimizedFeatures();

    ceres::Problem problem;

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

    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < option_.max_depth) ? (1.0 / d) : 1.0;
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
                f->observations[0].uv, pt_j, f->estimated_depth, ric_, tic_,
                state_->gyro_vec[host_id], state_->gyro_vec[target_id], state_->acc_vec[host_id],
                state_->acc_vec[target_id], state_->params_speed_bias[host_id].data(),
                state_->params_speed_bias[target_id].data(),
                state_->params_speed_bias[host_id].data() + 6,
                state_->params_speed_bias[target_id].data() + 6, sqrt_info, Grav_, state_->camera);
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
        auto* imu_cost = new IMUFactor<MidPointIntegrator>(pint_ptr, Grav_);
        problem.AddResidualBlock(
            imu_cost, nullptr, state_->params_pose[i].data(), state_->params_speed_bias[i].data(),
            state_->params_pose[i + 1].data(), state_->params_speed_bias[i + 1].data());
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = option_.num_iterations;
    opts.num_threads = option_.num_threads;
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

    int max_obs_len = 0;
    auto marg_features = feature_manager_->collectMarginalizationFeatures(max_obs_len);

    int num_marg_pints =
        max_obs_len > 1 ? max_obs_len - 1 : static_cast<int>(preintegrators_.size());
    std::vector<IntegratorBase<MidPointIntegrator>*> pint_ptrs;
    pint_ptrs.reserve(num_marg_pints);
    for (int i = 0; i < num_marg_pints; ++i) {
        pint_ptrs.push_back(&preintegrators_[i]);
    }

    auto marg = MarginlizationSqrt<MidPointIntegrator>(
        std::move(marg_features), std::make_unique<ceres::HuberLoss>(1.0), state_, pint_ptrs,
        marg_data_.get());
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
        Eigen::Matrix3d R_imu_i = Rs_[i] * ric_.transpose();
        Eigen::Matrix3d R_imu_j = Rs_[i + 1] * ric_.transpose();
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
    state_->Bas[0] = avg_acc - R0.transpose() * Grav_;
    state_->Bgs[0] = avg_gyro;
    last_ts_ = imu_measurements.back().timestamp;
    last_imu_acc_ = imu_measurements.back().acc;
    last_imu_gyro_ = imu_measurements.back().gyro;
    imu_initialized_ = true;
    spdlog::info(
        "IMU static init: Bg=({:.4f},{:.4f},{:.4f}) rad/s  Ba=({:.4f},{:.4f},{:.4f}) m/s²",
        state_->Bgs[0].x(), state_->Bgs[0].y(), state_->Bgs[0].z(), state_->Bas[0].x(),
        state_->Bas[0].y(), state_->Bas[0].z());
}
}  // namespace tassel_core
