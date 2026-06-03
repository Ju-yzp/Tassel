// =============================================================================
// test_vi_init_sim.cpp — Ceres VI joint init simulation
// =============================================================================
//
// Optimize [V, g] only; Ba is handled by the backend sliding-window optimization.
// Parameters: V(3n) + g(2-DOF tangent) = 3n + 2
//
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <sophus/so3.hpp>

#include <ceres/ceres.h>

#include "factor/integrator_base.h"
#include "factor/visual_factor.h"  // computeTangentBasis
#include "tassel_utils/types.h"

namespace tassel_core {
namespace {

int g_max_frames = 30;  // CLI overridable

// ── Simulation config ─────────────────────────────────────────────────────

struct SimConfig {
    // OAK-D-Lite BMI270 noise (from stereo_vins.yaml)
    double acc_n = 1.9291286492253038e-02;  // acc white noise [m/s²/√Hz]
    double acc_w = 8.0363303593887979e-04;  // acc random walk [m/s³/√Hz]
    double gyr_n = 2.6401761065710893e-03;  // gyro white noise [rad/s/√Hz]
    double gyr_w = 7.0304129470456934e-05;  // gyro random walk [rad/s²/√Hz]

    double g_mag = 9.80;      // gravity magnitude
    double imu_rate = 200.0;  // IMU frequency [Hz]
    double cam_rate = 20.0;   // camera frequency [Hz]
    double duration = 3.5;    // total simulation [s]

    // Motion params
    Eigen::Vector3d ang_vel = Eigen::Vector3d(0.3, 0.5, -0.2);  // rad/s body frame

    // True biases (|Ba| ≈ 0.3, realistic for BMI270 at startup)
    Eigen::Vector3d Ba_true = Eigen::Vector3d(0.25, -0.15, 0.10);
    Eigen::Vector3d Bg_true = Eigen::Vector3d(0.0, 0.0, 0.0);  // assume solved
};

// ── Cost function: preintegration residual per frame pair ──────────────────
// Parameters: [V_i(3), V_j(3), w(2)]  — Ba is NOT optimized, deferred to backend
// Residual: 6D [dp_err(3), dv_err(3)]

struct ViInitCost {
    Eigen::Vector3d dp_meas, dv_meas;
    Eigen::Vector3d Pi, Pj;
    Eigen::Matrix3d Ri_T;
    double sum_dt;
    double g_mag;
    Eigen::Matrix<double, 3, 2> dg_dw;
    Eigen::Vector3d g0_dir;

    ViInitCost(
        const MidPointIntegrator& pint, const Eigen::Vector3d& P_i, const Eigen::Vector3d& P_j,
        const Eigen::Matrix3d& R_i, const Eigen::Matrix<double, 2, 3>& tangent_base,
        const Eigen::Vector3d& g_dir, double gm)
        : dp_meas(pint.final_delta_p),
          dv_meas(pint.final_delta_v),
          Pi(P_i),
          Pj(P_j),
          Ri_T(R_i.transpose()),
          sum_dt(pint.sum_dt),
          g_mag(gm),
          dg_dw(tangent_base.transpose()),
          g0_dir(g_dir) {}

    template <typename T>
    bool operator()(const T* const V_i, const T* const V_j, const T* const w, T* residual) const {
        // Gravity on tangent plane: g = g_mag * (g0_dir + dg/dw * w) / |·|
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

// ── Trajectory state ──────────────────────────────────────────────────────

struct TrajState {
    double ts;
    Eigen::Matrix3d R_wb;  // world -> body
    Eigen::Vector3d P_w;   // world position
    Eigen::Vector3d V_w;   // world velocity
};

// ── Generate synthetic trajectory + IMU data ───────────────────────────────

struct SimData {
    std::vector<TrajState> traj;                        // one per camera frame
    std::vector<tassel_utils::IMUMeasurement> imu_all;  // all IMU measurements
    std::vector<MidPointIntegrator> preints;
    std::vector<Eigen::Matrix3d> Rs;  // camera poses
    std::vector<Eigen::Vector3d> Ps;
    std::vector<Eigen::Vector3d> V_true;  // ground-truth velocities per camera frame
};

SimData generateData(const SimConfig& cfg) {
    SimData data;

    double imu_dt = 1.0 / cfg.imu_rate;
    int n_imu = static_cast<int>(cfg.duration / imu_dt);
    int cam_skip = static_cast<int>(cfg.imu_rate / cfg.cam_rate);
    int n_cam = std::min(n_imu / cam_skip, g_max_frames);

    std::mt19937 rng(42);
    std::normal_distribution<double> n01(0.0, 1.0);

    double sigma_acc = cfg.acc_n / std::sqrt(imu_dt);
    double sigma_gyr = cfg.gyr_n / std::sqrt(imu_dt);
    double sigma_baw = cfg.acc_w * std::sqrt(imu_dt);
    double sigma_bgw = cfg.gyr_w * std::sqrt(imu_dt);

    Eigen::Vector3d g_world(0, 0, cfg.g_mag);

    Eigen::Matrix3d R_wb = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d V_w = Eigen::Vector3d(1.5, 0.7, -0.3);
    Eigen::Vector3d Ba = cfg.Ba_true;
    Eigen::Vector3d Bg = cfg.Bg_true;

    data.imu_all.reserve(n_imu);
    data.Rs.resize(n_cam);
    data.Ps.resize(n_cam);
    data.V_true.resize(n_cam);

    int cam_idx = 0;
    for (int k = 0; k < n_imu; ++k) {
        double ts = k * imu_dt;

        Eigen::Vector3d gyro_body = cfg.ang_vel;
        Eigen::Vector3d acc_body = -R_wb.transpose() * g_world;

        tassel_utils::IMUMeasurement m;
        m.timestamp = ts;
        for (int i = 0; i < 3; ++i) {
            m.acc[i] = acc_body[i] + Ba[i] + sigma_acc * n01(rng);
            m.gyro[i] = gyro_body[i] + Bg[i] + sigma_gyr * n01(rng);
        }
        data.imu_all.push_back(m);

        for (int i = 0; i < 3; ++i) {
            Ba[i] += sigma_baw * n01(rng);
            Bg[i] += sigma_bgw * n01(rng);
        }

        P_w += V_w * imu_dt;
        R_wb = R_wb * Sophus::SO3d::exp(gyro_body * imu_dt).matrix();

        if (k % cam_skip == 0) {
            data.Rs[cam_idx] = R_wb;
            data.Ps[cam_idx] = P_w;
            data.V_true[cam_idx] = V_w;
            ++cam_idx;
            if (cam_idx >= n_cam) break;
        }
    }

    for (int c = 0; c < n_cam - 1; ++c) {
        MidPointIntegrator pint(
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Matrix<double, 18, 18>::Identity() * 1e-6);
        int start_imu = c * cam_skip;
        int end_imu = (c + 1) * cam_skip;
        for (int k = start_imu; k <= end_imu; ++k) {
            pint.update(data.imu_all[k]);
        }
        data.preints.push_back(pint);
    }

    return data;
}

// ── Ceres joint init: [V, g(2-DOF)] only ──────────────────────────────────

struct InitResult {
    std::vector<Eigen::Vector3d> V;
    Eigen::Vector3d g;
    double final_cost;
};

InitResult runCeresInit(
    const std::vector<MidPointIntegrator>& preints, const std::vector<Eigen::Matrix3d>& Rs,
    const std::vector<Eigen::Vector3d>& Ps, double g_mag, int num_threads) {
    int n_pairs = static_cast<int>(preints.size());
    int n_frames = n_pairs + 1;

    std::vector<std::array<double, 3>> V_arr(n_frames);
    double w_arr[2] = {0.0, 0.0};

    for (int i = 0; i < n_frames; ++i) V_arr[i] = {0.0, 0.0, 0.0};

    // Initial gravity direction from first acc per frame
    Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
    for (int i = 0; i < n_pairs; ++i) {
        if (!preints[i].buffer.empty()) avg_acc += Rs[i] * (-preints[i].buffer.front().acc);
    }
    avg_acc /= n_pairs;
    Eigen::Vector3d g0_dir = avg_acc.normalized();
    if (g0_dir.z() < 0) g0_dir = -g0_dir;

    InitResult result;
    result.final_cost = 0.0;

    for (int iter = 0; iter < 4; ++iter) {
        Eigen::Matrix<double, 2, 3> tangent_base = computeTangentBasis(g0_dir);

        ceres::Problem problem;
        ceres::LossFunction* huber = new ceres::HuberLoss(1.0);

        for (int i = 0; i < n_pairs; ++i) {
            auto* cost = new ceres::AutoDiffCostFunction<ViInitCost, 6, 3, 3, 2>(
                new ViInitCost(preints[i], Ps[i], Ps[i + 1], Rs[i], tangent_base, g0_dir, g_mag));
            problem.AddResidualBlock(cost, huber, V_arr[i].data(), V_arr[i + 1].data(), w_arr);
        }

        ceres::Solver::Options opts;
        opts.linear_solver_type = ceres::DENSE_SCHUR;
        opts.max_num_iterations = 50;
        opts.num_threads = num_threads;
        opts.minimizer_progress_to_stdout = false;
        opts.logging_type = ceres::SILENT;

        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);

        Eigen::Vector3d dg = tangent_base.transpose() * Eigen::Vector2d(w_arr[0], w_arr[1]);
        g0_dir = (g0_dir + dg).normalized();
        w_arr[0] = w_arr[1] = 0.0;
        result.final_cost = summary.final_cost;
    }

    result.g = g_mag * g0_dir;
    result.V.resize(n_frames);
    for (int i = 0; i < n_frames; ++i)
        result.V[i] = Eigen::Vector3d(V_arr[i][0], V_arr[i][1], V_arr[i][2]);

    return result;
}

// ── Test ──────────────────────────────────────────────────────────────────

TEST(ViInitSim, RecoverGravityAndBias) {
    SimConfig cfg;
    SimData data = generateData(cfg);

    int n_pairs = static_cast<int>(data.preints.size());
    ASSERT_GE(n_pairs, 5) << "Need at least 5 frame pairs";

    std::cout << "\n=== VI Init Simulation (V+g only) ===\n"
              << "Frames: " << n_pairs + 1 << ", pairs: " << n_pairs << "\n"
              << "True Ba: " << cfg.Ba_true.transpose() << " (NOT optimized)\n"
              << "IMU rate: " << cfg.imu_rate << " Hz, cam rate: " << cfg.cam_rate << " Hz\n";

    InitResult result = runCeresInit(data.preints, data.Rs, data.Ps, cfg.g_mag, 1);

    // │ Gravity direction error
    Eigen::Vector3d g_true(0, 0, cfg.g_mag);
    double g_dot = result.g.normalized().dot(g_true.normalized());
    g_dot = std::clamp(g_dot, -1.0, 1.0);
    double g_deg = std::acos(g_dot) * 180.0 / M_PI;
    std::cout << "Gravity dir err: " << g_deg << " deg\n"
              << "  true: (" << g_true.transpose() << ")\n"
              << "  est:  (" << result.g.transpose() << ")\n";

    // } Velocity recovery
    double v0_err = (result.V[0] - data.V_true[0]).norm();
    std::cout << "V[0] true: " << data.V_true[0].transpose() << "\n"
              << "V[0] est:  " << result.V[0].transpose() << "\n"
              << "V[0] |err|: " << v0_err << "\n";
    std::cout << "Final cost: " << result.final_cost << "\n\n";

    EXPECT_LT(g_deg, 5.0) << "Gravity direction should be within 5 deg";
    EXPECT_LT(g_deg, 15.0) << "Gravity direction unrecoverable";
    EXPECT_LT(v0_err, 1.0) << "V[0] within 1 m/s";
}

}  // namespace
}  // namespace tassel_core

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1) {
        tassel_core::g_max_frames = std::atoi(argv[1]);
    }
    std::cout << "max_frames = " << tassel_core::g_max_frames << "\n";
    return RUN_ALL_TESTS();
}
