#include "vo_estimator.h"

#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>

#include <Eigen/Core>

#include "linearization/linearization_abs_qr.h"
#include "marginalization/marg_helper.h"
#include "optimizer/lm_optimizer.h"
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
      tic1_(tic1),
      arena_(option_.num_threads),
      init_ts_(-1) {
    cv::setNumThreads(option_.num_threads);

    state_->cur_frame_count = 0;
    initializeMargPrior();
}

void VoEstimator::initializeMargPrior() {
    int total_size = (state_->max_frame_count - 1) * tassel_utils::POSE_SIZE;

    cur_marg_lin_data_ = std::make_shared<MargLinData>();
    cur_marg_lin_data_->H = Eigen::MatrixXd::Zero(tassel_utils::POSE_SIZE, total_size);
    cur_marg_lin_data_->H.diagonal().head(tassel_utils::POSE_SIZE).array() = 1e-4;
    cur_marg_lin_data_->b = Eigen::VectorXd::Zero(tassel_utils::POSE_SIZE);
}

void VoEstimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame) {
    if (init_ts_ == -1) {
        init_ts_ = ts;
    }
    if (ts - init_ts_ < 5.0) {
        return;
    }
    int& frame_count = state_->cur_frame_count;

    bool is_keyframe = feature_manager_->checkKeyFrameByParallax(frame_count, feature_frame);

    if (is_keyframe) {
        if (frame_count + 1 == state_->max_frame_count) {
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
            --frame_count;
        } else {
            feature_manager_->initPoseByPNP(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            feature_manager_->triangulate(*state_, ric_, tic_, ric1_, tic1_);
            ++frame_count;
        }
    } else {
        feature_manager_->removeNewest(frame_count);
    }
}

void VoEstimator::optimize() {
    arena_.execute([&] {
        for (int i = 0; i < state_->max_frame_count; ++i) {
            if (!state_->poses[i].isLinearized()) {
                state_->poses[i].setLinearized();
            }
        }

        LinearizationAbsQR linearization(
            1, state_, feature_manager_->collectOptimizationFeatures(), option_.reprojection_loss,
            option_.min_depth, option_.max_depth, cur_marg_lin_data_);

        LMOptions opts;
        opts.max_iterations = option_.num_iterations;
        opts.lambda_initial = option_.lambda_initial;
        LMOptimizer optimizer(opts);
        optimizer.optimize(&linearization);
        optimizer.log_summary();
    });
}

void VoEstimator::marginalizeOldestFrame() {
    arena_.execute([&] {
        MargLinData new_mld;
        LinearizationAbsQR linearization(
            1, state_, feature_manager_->collectMarginalizationFeatures(),
            option_.reprojection_loss, option_.min_depth, option_.max_depth, cur_marg_lin_data_);
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

        feature_manager_->removeMarginalizedFeatures();
    });
}

void VoEstimator::slideWindow() {
    for (int i = 0; i < state_->max_frame_count - 1; i++) {
        state_->poses[i] = state_->poses[i + 1];
    }
    state_->poses[state_->max_frame_count - 1].reset();
}

}  // namespace tassel_core
