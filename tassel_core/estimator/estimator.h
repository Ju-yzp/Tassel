#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "estimator/estimator_option.h"
#include "factor/integrator_base.h"
#include "factor/marg_lin_data.h"
#include "frond_end/feature_manager.h"
#include "state/state.h"
#include "tassel_utils/types.h"

#include <sophus/se3.hpp>

#include <ceres/ceres.h>

namespace tassel_core {

class CameraBase;

class Estimator {
public:
    Estimator(
        const EstimatorOption& option, std::shared_ptr<State> state,
        std::shared_ptr<FeatureManager> fm,
        const Eigen::Matrix3d& ric = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& tic = Eigen::Vector3d::Zero(),
        const Eigen::Matrix3d& ric1 = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& tic1 = Eigen::Vector3d::Zero());

    void processMeasurement(
        double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
        const std::vector<tassel_utils::IMUMeasurement>& imu_measurements = {});

    void setPoseCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        pose_callback_ = std::move(cb);
    }
    void setCloudCallback(std::function<void(double, const std::vector<Eigen::Vector3d>&)> cb) {
        cloud_callback_ = std::move(cb);
    }
    void setCamera(const CameraBase* camera) {
        camera_ = camera;
        if (state_) state_->camera = camera;
    }

    void optimize();

    void reset();

private:
    void buildPrior();

    void slideWindow();

    void solveGyroBias();

    bool solveGravityVelocity(
        Eigen::Vector3d& g, const std::vector<Eigen::Vector3d>& Ps_cam,
        const std::vector<Eigen::Matrix3d>& Rs_cam);

    Eigen::Vector3d buildGVsOptimizeProblem(const Eigen::Vector3d& g_B0_init);

    Eigen::Matrix<double, 18, 18> initNoise() const;

    EstimatorOption option_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> feature_manager_;

    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Matrix3d ric1_;
    Eigen::Vector3d tic1_;

    const CameraBase* camera_ = nullptr;

    Eigen::Matrix<double, 18, 18> noise_;

    bool gravity_initialized_ = false;

    std::function<void(double, const Sophus::SE3d&)> pose_callback_;
    std::function<void(double, const std::vector<Eigen::Vector3d>&)> cloud_callback_;

    std::vector<MidPointIntegrator> preintegrators_;
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
