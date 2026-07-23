#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_H_

#include <Eigen/Core>
#include <functional>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <variant>
#include <vector>

#include "factor/integrator_base.h"
#include "frond_end/feature_manager.h"
#include "loop_closure.h"
#include "marg/marg_helper.h"
#include "marg/marg_lin_data.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/types.h"

#include <sophus/se3.hpp>

#include <ceres/ceres.h>

namespace tassel_core {

class CameraBase;

class Estimator {
public:
    Estimator(
        const tassel_tools::Parameters& params, std::shared_ptr<State> state,
        std::shared_ptr<FeatureManager> fm);

    void processMeasurement(
        tassel_utils::FrameId frame_id,
        const std::unordered_map<int, FeaturePerFrame>& feature_frame,
        const std::vector<tassel_utils::IMUMeasurement>& imu_measurements = {},
        double sync_delay = 0.0);

    void setPoseCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        pose_callback_ = std::move(cb);
    }
    void setRealtimePoseCallback(std::function<void(double, const Sophus::SE3d&)> cb) {
        realtime_pose_callback_ = std::move(cb);
    }
    void setVisualFactorCallback(std::function<void(double, const std::vector<int>&)> cb) {
        visual_factor_callback_ = std::move(cb);
    }
    void setLoopClosure(std::shared_ptr<tassel_loop::LoopClosure> loop_closure) {
        loop_closure_ = std::move(loop_closure);
    }
    void submitFrameImage(tassel_utils::FrameId frame_id, const cv::Mat& image) {
        if (loop_closure_) {
            frame_images_[frame_id] = image.clone();
        }
    }
    void setCamera(const CameraBase* camera) {
        camera_ = camera;
        if (state_) {
            state_->camera = camera;
        }
    }

    bool lastMeasurementWasKeyframe() const { return last_measurement_was_keyframe_; }

    void optimize(double timestamp = -1.0);

    void reset();

private:
    template <typename Integrator>
    using IntegratorVector = std::vector<Integrator>;
    using PreintegratorStorage =
        std::variant<IntegratorVector<MidPointIntegrator>, IntegratorVector<EulerIntegrator>>;

    void updateMarginalizationPrior(RetainedHostAction action);

    void predictFrameState(
        int frame_index, const std::vector<tassel_utils::IMUMeasurement>& imu_measurements);

    void slideInitializationWindow();

    void shiftWindowAfterMarginalization(RetainedHostAction action);

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

    bool initialized_ = false;
    bool last_measurement_was_keyframe_ = false;

    std::function<void(double, const Sophus::SE3d&)> pose_callback_;
    std::function<void(double, const Sophus::SE3d&)> realtime_pose_callback_;
    std::function<void(double, const std::vector<int>&)> visual_factor_callback_;
    std::shared_ptr<tassel_loop::LoopClosure> loop_closure_;
    std::map<tassel_utils::FrameId, cv::Mat> frame_images_;
    std::map<tassel_utils::FrameId, tassel_loop::KeyframeId> loop_keyframes_;

    PreintegratorStorage preintegrators_;
    double last_ts_ = -1;
    Eigen::Vector3d last_imu_acc_;
    Eigen::Vector3d last_imu_gyro_;

    std::unique_ptr<MargLinData> marginalization_prior_;

    // 动态初始化使用,存储sfm位姿以及imu在体坐标系下的速度
    std::vector<Eigen::Matrix3d> Rs_;
    std::vector<Eigen::Vector3d> Ps_;
    std::vector<Eigen::Vector3d> Vs_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_VO_ESTIMATOR_H_
