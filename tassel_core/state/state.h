#ifndef TASSEL_CORE_STATE_H_
#define TASSEL_CORE_STATE_H_

#include <Eigen/Core>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tassel_core {

class FeatureManager;

struct State {
    explicit State(size_t max_frame_count_)
        : max_frame_count(max_frame_count_), cur_frame_count(0) {
        if (max_frame_count == 0) {
            throw std::runtime_error("State: max_frame_count must be > 0");
        }
        Rs.resize(max_frame_count + 1, Eigen::Matrix3d::Identity());
        Ps.resize(max_frame_count + 1, Eigen::Vector3d::Zero());
        Vs.resize(max_frame_count + 1, Eigen::Vector3d::Zero());
        Bas.resize(max_frame_count + 1, Eigen::Vector3d::Zero());
        Bgs.resize(max_frame_count + 1, Eigen::Vector3d::Zero());
        param_pose.resize(max_frame_count + 1, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        param_speed.resize(max_frame_count + 1);
    }

    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State() = delete;
    void paramsToState(bool use_imu = false);

    void stateToParams(bool use_imu = false);

    void forwardSlide();

    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> Ps;
    std::vector<Eigen::Vector3d> Vs;
    std::vector<Eigen::Vector3d> Bas;
    std::vector<Eigen::Vector3d> Bgs;

    size_t max_frame_count;
    size_t cur_frame_count;

    std::vector<std::array<double, 7>> param_pose;

    std::vector<std::array<double, 9>> param_speed;
    std::vector<double> param_inv_depth;
    std::vector<double*> depth_ptrs;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_STATE_H_
