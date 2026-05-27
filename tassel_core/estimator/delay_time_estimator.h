#ifndef TASSEL_CORE_DELAY_TIME_ESTIMATOR_H_
#define TASSEL_CORE_DELAY_TIME_ESTIMATOR_H_

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <vector>

#include "factor/integrator_manager.h"
#include "factor/visual_reprojection.h"
#include "frond_end/feature_manager.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

class DelayTimeEstimator {
public:
    DelayTimeEstimator(
        std::shared_ptr<IntegratorManager> imu_manager,
        std::shared_ptr<FeatureManager> feature_manager, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic, int max_probe_iters = 20)
        : imu_manager_(std::move(imu_manager)),
          feature_manager_(std::move(feature_manager)),
          ric_(ric),
          tic_(tic),
          max_probe_iters_(max_probe_iters) {}

    double estimate(
        const State& state, const std::vector<double>& cam_ts, double initial_td = 0.0) {
        // TASSEL_ASSERT(state.max_frame_count == cam_ts.size());

        double dt_imu = imu_manager_->computeImuDt();
        double dt_step = (2.0 / 3.0) * dt_imu;

        auto features = feature_manager_->collectOptimizedFeatures();
        auto edges = buildEdges(features, static_cast<int>(cam_ts.size()));

        auto computeAllPoses = [&](double td_val) {
            std::vector<IMUState> poses(cam_ts.size());
            for (size_t i = 0; i < cam_ts.size(); ++i) {
                IMUState imu_state = getIMUStateAt(state, i, cam_ts[i]);
                if (!integrateTo(cam_ts[i] + td_val, dt_step, imu_state)) {
                    return poses;
                }
                poses[i] = imu_state;
            }
            return poses;
        };

        double td = std::clamp(initial_td, -0.1, 0.1);
        double probe_step = dt_imu;
        const double min_step = 0.1e-3;
        const double refine_ratio = 0.005;  // 代价相对下降低于0.5%时细化

        while (probe_step >= min_step) {
            for (int iter = 0; iter < max_probe_iters_; ++iter) {
                auto poses_cur = computeAllPoses(td);
                double cost_cur = computeReprojectionCost(poses_cur, edges);

                bool improved = false;

                double td_pos = std::clamp(td + probe_step, -0.1, 0.1);
                auto poses_pos = computeAllPoses(td_pos);
                double cost_pos = computeReprojectionCost(poses_pos, edges);

                if (cost_pos < cost_cur) {
                    td = td_pos;
                    improved = true;
                } else {
                    double td_neg = std::clamp(td - probe_step, -0.1, 0.1);
                    auto poses_neg = computeAllPoses(td_neg);
                    double cost_neg = computeReprojectionCost(poses_neg, edges);

                    if (cost_neg < cost_cur) {
                        td = td_neg;
                        improved = true;
                    }
                }

                if (!improved) {
                    break;  // 两个方向都未改进
                }
                // 改进幅度太小时提前细化，避免在当前分辨率浪费迭代
                double new_cost = computeReprojectionCost(computeAllPoses(td), edges);
                if ((cost_cur - new_cost) / cost_cur < refine_ratio) {
                    break;
                }
            }
            probe_step *= 0.5;
        }

        return td;
    }

private:
    struct IMUState {
        double t = 0;
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Vector3d P = Eigen::Vector3d::Zero();
        Eigen::Vector3d w = Eigen::Vector3d::Zero();  // 体坐标系，已减偏置
        Eigen::Vector3d a = Eigen::Vector3d::Zero();  // 体坐标系，已减偏置
    };

    struct Edge {
        int host_id;
        int target_id;
        Eigen::Vector3d uv_host;
        Eigen::Vector3d uv_target;
        double depth;  // 路标在 host 帧的深度
    };

    IMUState getIMUStateAt(const State& state, int frame_id, double t) const {
        IMUState s;
        s.t = t;
        s.R = state.Rs[frame_id];
        s.P = state.Ps[frame_id];
        s.v = state.Vs[frame_id];
        s.w = imu_manager_->getGyro(t);
        s.a = imu_manager_->getAcceleration(t);
        return s;
    }

    // 小步长中点积分: state.t → t_target
    // a_global = 0.5 * (R_cur * a_cur + R_next * a_next)
    bool integrateTo(double t_target, double dt_step, IMUState& state) const {
        double direction = (t_target > state.t) ? 1.0 : -1.0;

        while (std::abs(t_target - state.t) > 1e-12) {
            double dt = direction * std::min(dt_step, std::abs(t_target - state.t));
            double t_next = state.t + dt;

            Eigen::Vector3d w_next = imu_manager_->getGyro(t_next);
            Eigen::Vector3d a_next = imu_manager_->getAcceleration(t_next);

            Eigen::Vector3d w_mid = 0.5 * (state.w + w_next);
            Eigen::Matrix3d R_next = state.R * Sophus::SO3d::exp(w_mid * dt).matrix();

            // 分别用 R_cur 和 R_next 旋转，取平均
            Eigen::Vector3d a_global = 0.5 * (state.R * state.a + R_next * a_next);

            Eigen::Vector3d v_next = state.v + a_global * dt;
            Eigen::Vector3d P_next = state.P + state.v * dt + 0.5 * a_global * dt * dt;

            state.R = R_next;
            state.v = v_next;
            state.P = P_next;
            state.w = w_next;
            state.a = a_next;
            state.t = t_next;
        }
        return true;
    }

    std::vector<Edge> buildEdges(const std::vector<Feature*>& features, int num_frames) const {
        std::vector<Edge> edges;
        for (const auto* f : features) {
            if (f->observations.size() < 2) {
                continue;
            }
            if (f->estimated_depth <= 0) {
                continue;
            }
            int host_id = static_cast<int>(f->start_frame_id);
            for (size_t k = 1; k < f->observations.size(); ++k) {
                int target_id = host_id + static_cast<int>(k);
                if (target_id >= num_frames) {
                    break;
                }
                edges.push_back(Edge{
                    host_id, target_id, f->observations[0].uv, f->observations[k].uv,
                    f->estimated_depth});
            }
        }
        return edges;
    }

    double computeReprojectionCost(
        const std::vector<IMUState>& poses, const std::vector<Edge>& edges) const {
        double total_cost = 0.0;

        for (const auto& e : edges) {
            if (e.host_id >= static_cast<int>(poses.size()) ||
                e.target_id >= static_cast<int>(poses.size())) {
                continue;
            }

            const auto& pose_i = poses[e.host_id];
            const auto& pose_j = poses[e.target_id];

            Eigen::Matrix<double, 2, 3> tangent_base = computeTangentBasis(e.uv_target);

            Eigen::Vector3d pi_in_H = e.uv_host * e.depth;
            Eigen::Vector3d pi_in_I = ric_ * pi_in_H + tic_;
            Eigen::Vector3d pi_in_W = pose_i.R * pi_in_I + pose_i.P;
            Eigen::Vector3d pj_in_I = pose_j.R.transpose() * (pi_in_W - pose_j.P);
            Eigen::Vector3d pj_in_C = ric_.transpose() * (pj_in_I - tic_);

            double norm = pj_in_C.norm();
            Eigen::Vector2d r = tangent_base * (pj_in_C / norm - e.uv_target);
            total_cost += r.squaredNorm();
        }

        return edges.empty() ? 0.0 : total_cost / edges.size();
    }

    std::shared_ptr<IntegratorManager> imu_manager_;
    std::shared_ptr<FeatureManager> feature_manager_;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    int max_probe_iters_;
};

}  // namespace tassel_core

#endif /* TASSEL_CORE_DELAY_TIME_ESTIMATOR_H_ */
