#ifndef TASSEL_CORE_ESTIMATOR_VIO_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_VIO_ESTIMATOR_H_

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "delay_time_estimator.h"
#include "estimator/estimator_option.h"
#include "factor/integrator_manager.h"
#include "factor/marg_linearized_data.h"
#include "frond_end/feature_manager.h"
#include "state/state.h"
#include "tassel_utils/types.h"

#include <sophus/se3.hpp>

namespace tassel_core {

class VioEstimator {
public:
    VioEstimator(
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
    void setPathCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        path_callback_ = std::move(cb);
    }
    void setMonoCloudCallback(std::function<void(double, const std::vector<Eigen::Vector3d>&)> cb) {
        mono_cloud_callback_ = std::move(cb);
    }
    void setStereoCloudCallback(
        std::function<void(double, const std::vector<Eigen::Vector3d>&)> cb) {
        stereo_cloud_callback_ = std::move(cb);
    }

    std::shared_ptr<State> getState() const { return state_; }
    std::shared_ptr<FeatureManager> getFeatureManager() const { return feature_manager_; }
    const std::vector<double>& getCamTimestamps() const { return cam_timestamps_; }
    bool isTdEstimated() const { return td_estimated_; }

    void dumpToFile(const std::string& path) const;

private:
    void optimize();
    void marginalize();
    void slideWindow();
    void initializeImu(const std::vector<tassel_utils::IMUMeasurement>& imu_measurements);

    Eigen::Matrix<double, 18, 18> initNoise() const;

    EstimatorOption option_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> feature_manager_;
    std::unique_ptr<MargLinData> marg_lin_data_;
    std::vector<std::array<double, 6>> marg_poses_linearized_;

    std::shared_ptr<IntegratorManager> imu_manager_;
    std::unique_ptr<DelayTimeEstimator> delay_time_estimator_;

    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    Eigen::Matrix3d ric1_;
    Eigen::Vector3d tic1_;

    Eigen::Matrix<double, 18, 18> noise_;

    // IMU gravity initialization
    bool imu_initialized_ = false;
    bool td_estimated_ = false;
    double init_ts_ = -1;
    std::vector<tassel_utils::IMUMeasurement> imu_init_buf_;

    // callbacks
    std::function<void(double, const Sophus::SE3d&)> pose_callback_;
    std::function<void(double, const Sophus::SE3d&)> path_callback_;
    std::function<void(double, const std::vector<Eigen::Vector3d>&)> mono_cloud_callback_;
    std::function<void(double, const std::vector<Eigen::Vector3d>&)> stereo_cloud_callback_;

    double last_ts_ = -1;
    Eigen::Vector3d last_imu_acc_;
    Eigen::Vector3d last_imu_gyro_;

    // 滑窗内各帧的相机时间戳，供时间延迟估计器使用
    std::vector<double> cam_timestamps_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
