#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include "factor/integrator_base.h"
#include "frond_end/feature_manager.h"
#include "marg/marg_lin_data.h"
#include "marg/window_action.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/types.h"

#include <sophus/se3.hpp>

#include <ceres/ceres.h>

namespace tassel_core {

class CameraBase;
class WindowOptimizer;

// 整帧优化和窗口搬迁完成后复制发布；跨线程使用时不得再引用 FeatureManager 内部对象。
struct TrackingPredictionSnapshot {
    tassel_utils::FrameId source_frame_id = tassel_utils::kInvalidFrameId;
    FrameState source_state;
    double imu_timestamp = -1.0;
    Eigen::Vector3d imu_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gyro = Eigen::Vector3d::Zero();
    double delay_time = 0.0;
    std::unordered_map<int, Eigen::Vector3d> world_landmarks;
};

// 仅在帧被迁入长期 retained slot 时发布，位姿对应同一 frame_id 的 IMU 状态。
struct RetainedKeyframe {
    tassel_utils::FrameId frame_id = tassel_utils::kInvalidFrameId;
    Sophus::SE3d pose;
    std::vector<ObservedLandmark> landmarks;
};

std::unordered_map<int, cv::Point2f> predictLandmarkPixelsFromSnapshot(
    const TrackingPredictionSnapshot& snapshot, tassel_utils::FrameId target_frame_id,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay,
    const CameraBase& camera, const tassel_tools::Parameters& params);

class Estimator {
public:
    Estimator(
        const tassel_tools::Parameters& params, std::shared_ptr<State> state,
        std::shared_ptr<FeatureManager> fm);
    ~Estimator();

    void processMeasurement(
        tassel_utils::FrameId frame_id,
        const std::unordered_map<int, FeaturePerFrame>& feature_frame,
        const std::vector<tassel_utils::IMUMeasurement>& imu_measurements = {},
        double sync_delay = 0.0);

    void setPoseCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        pose_callback_ = std::move(cb);
    }
    void setVisualFactorCallback(std::function<void(double, const std::vector<int>&)> cb) {
        visual_factor_callback_ = std::move(cb);
    }
    void setCamera(const CameraBase* camera) {
        camera_ = camera;
        if (state_) {
            state_->camera = camera;
        }
    }

    bool lastMeasurementWasKeyframe() const { return last_measurement_was_keyframe_; }
    const std::optional<RetainedKeyframe>& lastRetainedKeyframe() const {
        return last_retained_keyframe_;
    }

    std::optional<TrackingPredictionSnapshot> makeTrackingPredictionSnapshot() const;

    std::unordered_map<int, cv::Point2f> predictLandmarkPixels(
        const TrackingPredictionSnapshot& snapshot, tassel_utils::FrameId target_frame_id,
        const std::vector<tassel_utils::IMUMeasurement>& imu_measurements, double sync_delay) const;

    void optimize();

    void reset();

private:
    template <typename Integrator>
    using IntegratorVector = std::vector<Integrator>;
    using PreintegratorStorage =
        std::variant<IntegratorVector<MidPointIntegrator>, IntegratorVector<EulerIntegrator>>;

    void updateMarginalizationPrior(RetainedHostAction action);

    RetainedHostAction selectMarginalizationAction() const;

    void predictFrameState(
        int frame_index, const std::vector<tassel_utils::IMUMeasurement>& imu_measurements);

    void runSlidingWindowUpdate(int latest_frame_index, double timestamp);

    void slideInitializationWindow();

    void migrateMarginalizedData(RetainedHostAction action);

    void normalizeGaugeAfterOptimization(int reference_frame_index);

    bool isStationaryWindow() const;

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
    std::unique_ptr<WindowOptimizer> window_optimizer_;

    Eigen::Matrix<double, 18, 18> noise_;

    bool initialized_ = false;
    bool last_measurement_was_keyframe_ = false;
    std::optional<RetainedKeyframe> last_retained_keyframe_;
    std::function<void(double, const Sophus::SE3d&)> pose_callback_;
    std::function<void(double, const std::vector<int>&)> visual_factor_callback_;
    PreintegratorStorage preintegrators_;
    double last_ts_ = -1;
    Eigen::Vector3d last_imu_acc_;
    Eigen::Vector3d last_imu_gyro_;

    std::unique_ptr<MargLinData> marginalization_prior_;
    // 动态初始化使用，存储 SFM 位姿以及 IMU 在体坐标系下的速度。
    std::vector<Eigen::Matrix3d> Rs_;
    std::vector<Eigen::Vector3d> Ps_;
    std::vector<Eigen::Vector3d> Vs_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
