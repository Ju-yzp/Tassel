#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "frond_end/feature_manager.h"
#include "initial/dynamic_initializer.h"
#include "marg/marg_lin_data.h"
#include "marg/window_action.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/types.h"

#include <sophus/se3.hpp>

#include <ceres/ceres.h>

namespace tassel_core {

class CameraBase;

// 整帧优化和窗口搬迁完成后复制发布；跨线程使用时不得再引用 FeatureManager 内部对象。
struct TrackingPredictionSnapshot {
    tassel_utils::FrameId source_frame_id = tassel_utils::kInvalidFrameId;
    FrameState source_state;
    double imu_timestamp = -1.0;
    Eigen::Vector3d imu_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gyro = Eigen::Vector3d::Zero();
    double time_delay = 0.0;
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

    void reset();

private:
    using PreintegratorStorage = DynamicInitializer::PreintegratorStorage;

    void updatePrior(RetainedHostAction action);

    RetainedHostAction selectMarginalizationAction() const;

    void predictLatestFrame(
        int frame_index, const std::vector<tassel_utils::IMUMeasurement>& imu_measurements);

    void processWindow(int latest_active_frame_index, double timestamp);

    void slideInitializationWindow();

    void slideWindow(RetainedHostAction action);

    void normalizeGauge(
        int reference_frame_index, const Eigen::Matrix3d& reference_rotation,
        const Eigen::Vector3d& reference_position);

    void optimizeWindow();

    Eigen::Matrix<double, 18, 18> initNoise() const;

    template <typename Fn>
    decltype(auto) withPreintegrators(Fn&& fn) {
        return std::forward<Fn>(fn)(preintegrators_);
    }

    const tassel_tools::Parameters& params_;
    std::shared_ptr<State> state_;
    std::shared_ptr<FeatureManager> feature_manager_;
    const CameraBase* camera_ = nullptr;

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
    std::unique_ptr<DynamicInitializer> dynamic_initializer_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
