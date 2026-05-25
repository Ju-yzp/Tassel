#include "vo_estimator.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <set>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>

#include "factor/landmark_block.h"
#include "factor/marg_helper.h"
#include "factor/marg_linearized_data.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/se3_right_manifold.h"
#include "factor/visual_factor.h"

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
      imu_initialized_(false),
      init_ts_(-1) {
    cv::setNumThreads(option_.num_threads);
    state_->cur_frame_count = 0;
    state_->ric = ric_;
    state_->tic = tic_;
}

void VoEstimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    if (!imu_initialized_) {
        if (init_ts_ < 0) {
            init_ts_ = ts;
        }
        imu_init_buf_.insert(imu_init_buf_.end(), imu_measurements.begin(), imu_measurements.end());
        if (ts - init_ts_ >= option_.init_time_span) {
            initializeImu(imu_init_buf_);
        }
        return;
    }

    int& frame_count = state_->cur_frame_count;
    bool is_keyframe = feature_manager_->checkKeyFrameByParallax(frame_count, feature_frame);

    if (is_keyframe) {
        if (frame_count + 1 == state_->max_frame_count) {
            feature_manager_->initPoseByPNP(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            feature_manager_->triangulate(*state_, ric_, tic_, ric1_, tic1_);
            optimize();
            feature_manager_->removeOutliers(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            marginalize();
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
        // if (frame_count + 1 == state_->max_frame_count) {
        //     feature_manager_->initPoseByPNP(
        //         *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        // }
        feature_manager_->removeNewest(frame_count);
    }

    if (state_->cur_frame_count > 0) {
        int latest_idx = 0;
        if (state_->cur_frame_count + 1 == state_->max_frame_count) {
            latest_idx = state_->cur_frame_count;
        } else {
            latest_idx = state_->cur_frame_count - 1;
        }
        Sophus::SE3d latest_pose(state_->Rs[latest_idx], state_->Ps[latest_idx]);

        if (pose_callback_) pose_callback_(ts, latest_pose);
        if (path_callback_) path_callback_(ts, latest_pose);
        if (mono_cloud_callback_) {
            auto pts = feature_manager_->getMonoPointCloud(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            mono_cloud_callback_(ts, pts);
        }
        if (stereo_cloud_callback_) {
            auto pts = feature_manager_->getStereoPointCloud(
                *state_, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            stereo_cloud_callback_(ts, pts);
        }
    }
}

void VoEstimator::optimize() {
    state_->stateToParams();
    auto features = feature_manager_->collectOptimizedFeatures();

    ceres::Problem problem;

    // inverse depth for each feature
    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < option_.max_depth) ? (1.0 / d) : 1.0;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
        problem.SetParameterLowerBound(&inv_depth_params[k], 0, 1.0 / option_.max_depth);
        problem.SetParameterUpperBound(&inv_depth_params[k], 0, 1.0 / option_.min_depth);
    }

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->param_poses[i].data(), 6, se3_manifold);
    }

    problem.SetParameterBlockConstant(state_->param_poses[0].data());

    ceres::LossFunction* loss = new ceres::HuberLoss(0.005);
    std::set<int> involved_indices;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        int host_id = f->start_frame_id;
        involved_indices.insert(host_id);
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            int target_id = host_id + static_cast<int>(obs_idx);
            involved_indices.insert(target_id);
            auto* cost = new VisualFactor(
                f->observations[0].uv, f->observations[obs_idx].uv, Eigen::Matrix3d::Identity(),
                Eigen::Vector3d::Zero(), option_.min_depth);
            problem.AddResidualBlock(
                cost, loss, state_->param_poses[host_id].data(),
                state_->param_poses[target_id].data(), &inv_depth_params[k]);
        }
    }
    if (static_cast<int>(involved_indices.size()) == state_->max_frame_count) {
        spdlog::info(
            "Optimized features cover all {} frames in the sliding window",
            state_->max_frame_count);
    }

    // 添加边缘化先验
    if (marg_lin_data_ && marg_lin_data_->H.rows() > 0) {
        auto* prior =
            new MarginalizationPrior(marg_lin_data_->H, marg_lin_data_->b, marg_poses_linearized_);
        int num_kept = static_cast<int>(marg_poses_linearized_.size());
        std::vector<double*> param_blocks;
        for (int i = 0; i < num_kept; ++i) {
            param_blocks.push_back(state_->param_poses[i].data());
        }
        problem.AddResidualBlock(prior, nullptr, param_blocks);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = option_.num_iterations;
    opts.num_threads = option_.num_threads;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    spdlog::info("Ceres: {}", summary.FullReport());

    state_->paramsToState();

    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        features[k]->estimated_depth = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
    }
}

void VoEstimator::marginalize() {
    int num_frames = state_->max_frame_count;

    // 收集当前窗口所有帧的线性化点
    std::vector<std::array<double, 6>> poses_linearized;
    for (int i = 0; i < num_frames; ++i) {
        poses_linearized.push_back(state_->param_poses[i]);
    }

    // 收集需要边缘化的特征
    auto marg_features = feature_manager_->collectMarginalizationFeatures();

    int marg_size = 6;
    int keep_size = (num_frames - 1) * 6;

    ceres::LossFunction* loss = new ceres::HuberLoss(0.005);
    std::vector<LandmarkBlock> blocks;
    int total_new_rows = 0;
    for (const auto& feature : marg_features) {
        int num_obs = static_cast<int>(feature.observations.size()) - 1;
        LandmarkBlock lb(option_.min_depth, loss);
        lb.allocate(num_frames, num_obs);
        lb.linearize(
            feature, poses_linearized, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        lb.performQR();
        total_new_rows += lb.keptRows();
        blocks.push_back(std::move(lb));
    }

    int prev_rows = (marg_lin_data_ && marg_lin_data_->H.rows() > 0) ? marg_lin_data_->H.rows() : 0;
    int all_rows = total_new_rows + prev_rows;

    Eigen::MatrixXd Q2Jp(all_rows, marg_size + keep_size);
    Eigen::VectorXd Q2r(all_rows);
    Q2Jp.setZero();
    Q2r.setZero();

    int offset = 0;
    for (const auto& lb : blocks) {
        lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, offset);
        offset += lb.keptRows();
    }

    if (prev_rows > 0) {
        Q2Jp.block(total_new_rows, 0, prev_rows, marg_lin_data_->H.cols()) = marg_lin_data_->H;
        Q2r.segment(total_new_rows, prev_rows) = marg_lin_data_->b;
    }

    Eigen::MatrixXd H_new;
    Eigen::VectorXd b_new;
    MargHelper::marginalizeSqrtToSqrt(marg_size, keep_size, Q2Jp, Q2r, H_new, b_new);

    if (!marg_lin_data_) {
        marg_lin_data_ = std::make_unique<MargLinData>();
    }
    marg_lin_data_->H = H_new;
    marg_lin_data_->b = b_new;

    marg_poses_linearized_.clear();
    for (int i = 1; i < num_frames; ++i) {
        marg_poses_linearized_.push_back(poses_linearized[i]);
    }

    feature_manager_->removeMarginalizedFeatures();
    delete loss;
}

void VoEstimator::initializeImu(const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
    for (const auto& m : imu_measurements) {
        avg_acc += m.acc;
    }
    avg_acc /= static_cast<double>(imu_measurements.size());

    Eigen::Vector3d g_imu = avg_acc.normalized();
    Eigen::Vector3d g_world = tassel_utils::G.normalized();

    Eigen::Matrix3d R_w_i = Eigen::Quaterniond::FromTwoVectors(g_imu, g_world).toRotationMatrix();
    Eigen::Matrix3d R_w_c0 = R_w_i * ric_.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_w_c0, Eigen::ComputeFullU | Eigen::ComputeFullV);
    state_->Rs[0] = svd.matrixU() * svd.matrixV().transpose();
    state_->Ps[0] = Eigen::Vector3d::Zero();

    imu_initialized_ = true;
    spdlog::info("IMU gravity initialized");
}

void VoEstimator::slideWindow() {
    const int n = state_->max_frame_count;
    for (int i = 0; i < n - 1; ++i) {
        state_->Rs[i] = state_->Rs[i + 1];
        state_->Ps[i] = state_->Ps[i + 1];
    }
}

}  // namespace tassel_core
