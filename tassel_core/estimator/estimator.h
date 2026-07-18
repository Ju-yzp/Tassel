#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include "estimator/window_marginalization_plan.h"
#include "factor/integrator_base.h"
#include "frond_end/feature_manager.h"
#include "marg/marg_lin_data.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/types.h"

#include <sophus/se3.hpp>

#include <ceres/ceres.h>

namespace tassel_core {

class CameraBase;

struct OptimizationStats {
    double total_cost_before = 0.0;
    double total_cost_after = 0.0;
    double visual_cost_before = 0.0;
    double visual_cost_after = 0.0;
    double prior_cost_before = 0.0;
    double prior_cost_after = 0.0;
    double imu_cost_before = 0.0;
    double imu_cost_after = 0.0;
    std::vector<int> visual_factors_per_frame;
};

class Estimator {
public:
    Estimator(
        const tassel_tools::Parameters& params, std::shared_ptr<State> state,
        std::shared_ptr<FeatureManager> fm);

    void processMeasurement(
        tassel_utils::FrameId frame_id,
        const std::unordered_map<int, FeaturePerFrame>& feature_frame,
        const std::vector<tassel_utils::IMUMeasurement>& imu_measurements = {},
        double applied_delay = 0.0);

    void setPoseCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        pose_callback_ = std::move(cb);
    }
    void setRealtimePoseCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        realtime_pose_callback_ = std::move(cb);
    }
    void setCloudCallback(std::function<void(double, const std::vector<Eigen::Vector3d>&)> cb) {
        cloud_callback_ = std::move(cb);
    }
    void setOptimizationCallback(std::function<void(double, const OptimizationStats&)> cb) {
        optimization_callback_ = std::move(cb);
    }
    void setCamera(const CameraBase* camera) {
        camera_ = camera;
        if (state_) state_->camera = camera;
    }

    void optimize(double timestamp = -1.0);

    void reset();

private:
    template <typename Integrator>
    using IntegratorVector = std::vector<Integrator>;
    using PreintegratorStorage =
        std::variant<IntegratorVector<MidPointIntegrator>, IntegratorVector<EulerIntegrator>>;

    void buildPrior(const WindowMarginalizationPlan& plan);

    void slideInitializationWindow();

    void slideWindow(const WindowMarginalizationPlan& plan);

    bool tryInitialize();

    Eigen::Matrix<double, 18, 18> initNoise() const;

    template <typename Fn>
    decltype(auto) visitPreintegrators(Fn&& fn) {
        return std::visit(std::forward<Fn>(fn), preintegrators_);
    }

    template <typename Fn>
    decltype(auto) visitPreintegrators(Fn&& fn) const {
        return std::visit(std::forward<Fn>(fn), preintegrators_);
    }

    const tassel_tools::Parameters& params_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> feature_manager_;

    const CameraBase* camera_ = nullptr;

    Eigen::Matrix<double, 18, 18> noise_;

    bool gravity_initialized_ = false;

    std::function<void(double, const Sophus::SE3d&)> pose_callback_;
    std::function<void(double, const Sophus::SE3d&)> realtime_pose_callback_;
    std::function<void(double, const std::vector<Eigen::Vector3d>&)> cloud_callback_;
    std::function<void(double, const OptimizationStats&)> optimization_callback_;

    PreintegratorStorage preintegrators_;
    double last_ts_ = -1;
    Eigen::Vector3d last_imu_acc_;
    Eigen::Vector3d last_imu_gyro_;

    std::unique_ptr<MargLinData> marg_data_;

    // 动态初始化使用,存储sfm位姿以及imu在体坐标系下的速度
    std::vector<Eigen::Matrix3d> Rs_;
    std::vector<Eigen::Vector3d> Ps_;
    std::vector<Eigen::Vector3d> Vs_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
