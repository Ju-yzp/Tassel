#include "vo_estimator.h"

#include <spdlog/spdlog.h>

#include <Eigen/Core>

#include "linearization/linearization_abs_qr.h"
#include "lm_optimizer/lm_optimizer.h"
#include "marginalization/marg_helper.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

VoEstimator::VoEstimator(
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
    state_->cur_frame_count = 0;
    initializeMargPrior();
}

void VoEstimator::initializeMargPrior() {
    int total_size = (state_->max_frame_count - 1) * tassel_utils::POSE_SIZE;

    cur_marg_lin_data_ = std::make_shared<MargLinData>();

    // Weak diagonal prior on frame 0 to fix gauge freedom.
    // H is sqrt-form: H^T * H ≈ weak prior on the first pose.
    // topLeftCorner(total_size, total_size) in linearizeMargPrior covers frames 0..(n-2),
    // so cols 0..5 (frame 0) receive the weak prior.
    cur_marg_lin_data_->H = Eigen::MatrixXd::Zero(tassel_utils::POSE_SIZE, total_size);
    cur_marg_lin_data_->H.diagonal().head(tassel_utils::POSE_SIZE).array() = 1e-4;
    cur_marg_lin_data_->b = Eigen::VectorXd::Zero(tassel_utils::POSE_SIZE);
}

bool VoEstimator::processMeasurement(
    const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    int& frame_count = state_->cur_frame_count;

    // Pose prediction: copy previous pose
    if (frame_count > 0) {
        state_->poses[frame_count] = state_->poses[frame_count - 1];
        state_->poses[frame_count].set_optimized_pose(
            state_->poses[frame_count - 1].get_optimized_pose());
    } else {
        state_->poses[0] = PoseStateWithLin(Sophus::SE3d());
        state_->poses[0].set_optimized_pose(Sophus::SE3d());
    }

    bool is_keyframe = feature_manager_->checkKeyFrameByParallax(frame_count, feature_frame);

    if (is_keyframe) {
        if (frame_count + 1 >= state_->max_frame_count) {
            feature_manager_->initPoseByPNP(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            feature_manager_->triangulate(*state_, ric_, tic_, ric1_, tic1_);
            ++frame_count;

            if (option_.optimize_enabled) {
                optimize();
            }
            if (option_.marginalization_enabled) {
                marginalizeOldestFrame();
            }
            feature_manager_->removeOutliers(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            feature_manager_->removeOldest(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            slideWindow();
        } else {
            feature_manager_->initPoseByPNP(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            feature_manager_->triangulate(*state_, ric_, tic_, ric1_, tic1_);
            ++frame_count;
        }
    } else {
        feature_manager_->removeNewest(frame_count);
    }

    return is_keyframe;
}

void VoEstimator::optimize() {
    // Set linearization point for kept frames before building the linear system.
    // Frame n-1 (newest) is excluded — it is not in the marg prior.
    for (int i = 0; i < state_->cur_frame_count - 1; i++) {
        if (!state_->poses[i].isLinearized()) {
            state_->poses[i].setLinearized();
        }
    }

    LinearizationAbsQR linearization(
        4, state_, feature_manager_, option_.reprojection_loss, option_.depth_loss,
        cur_marg_lin_data_);

    double initial_cost = linearization.computeError();

    LMOptions opts;
    opts.max_iterations = option_.num_iterations;
    opts.lambda_initial = option_.lambda_initial;
    LMOptimizer optimizer(opts);
    int iterations = optimizer.optimize(&linearization);

    double final_cost = linearization.computeError();

    spdlog::info(
        "optimize: initial_cost={:.6f}, final_cost={:.6f}, iterations={}", initial_cost, final_cost,
        iterations);

    for (int i = 0; i < state_->cur_frame_count; i++) {
        state_->poses[i].set_optimized_pose(state_->poses[i].get_pose());
    }
}

void VoEstimator::marginalizeOldestFrame() {
    MargLinData new_mld;

    // Build linear system. cur_marg_lin_data_ (from previous marg) is included
    // as prior rows, accumulating over multiple marginalizations.
    LinearizationAbsQR linearization(
        4, state_, feature_manager_, option_.reprojection_loss, option_.depth_loss,
        cur_marg_lin_data_);
    linearization.linearizeProbelm();
    linearization.performQR();

    Eigen::MatrixXd Q2Jp;
    Eigen::VectorXd Q2r;
    linearization.get_dense_Q2Jp_Q2r(Q2Jp, Q2r);

    int marg_size = tassel_utils::POSE_SIZE;
    int keep_size = state_->max_frame_count * tassel_utils::POSE_SIZE - marg_size;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;
    MargHelper::marginalizeOldest(marg_size, keep_size, Q2Jp, Q2r, marg_sqrt_H, marg_sqrt_b);

    new_mld.H = marg_sqrt_H;
    new_mld.b = marg_sqrt_b;

    cur_marg_lin_data_ = std::make_shared<MargLinData>(std::move(new_mld));
}

void VoEstimator::slideWindow() {
    for (int i = 0; i < state_->max_frame_count - 1; i++) {
        // Absorb delta into linearization point: the actual current pose
        // becomes the new baseline for the next optimization round.
        Pose actual_pose = state_->poses[i + 1].get_pose();
        if (state_->poses[i + 1].isLinearized()) {
            actual_pose = actual_pose * Sophus::SE3d::exp(state_->poses[i + 1].get_delta());
        }
        Pose optimized_pose = state_->poses[i + 1].get_optimized_pose();

        state_->poses[i] = PoseStateWithLin(actual_pose);
        state_->poses[i].set_optimized_pose(optimized_pose);
    }
    state_->poses[state_->max_frame_count - 1].reset();
    --state_->cur_frame_count;
}

}  // namespace tassel_core
