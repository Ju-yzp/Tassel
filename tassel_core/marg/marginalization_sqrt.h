#ifndef TASSEL_CORE_MARG_MARGINALIZATION_SQRT_H_
#define TASSEL_CORE_MARG_MARGINALIZATION_SQRT_H_

#include <ceres/loss_function.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "factor/integrator_base.h"
#include "factor/visual_frame_cache.h"
#include "frond_end/feature.h"
#include "marg/imu_block.h"
#include "marg/landmark_block.h"
#include "marg/marg_lin_data.h"
#include "state/state.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

template <typename Derived>
class MarginalizationSqrt {
public:
    // retiring_features 是本次边缘化独占的值快照，其顺序与 landmark_blocks_ 一致。
    // landmark_target_frame_index < 0 表示消去每个宿主路标的全部后续观测。
    MarginalizationSqrt(
        std::vector<std::pair<int, Feature>> retiring_features, int landmark_target_frame_index,
        std::unique_ptr<ceres::LossFunction> loss_function, std::shared_ptr<State> state,
        std::vector<IntegratorBase<Derived>*>& preintegrators, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic, const MargLinData* prior = nullptr,
        int first_imu_factor_index = 0)
        : retiring_features_(std::move(retiring_features)),
          landmark_target_frame_index_(landmark_target_frame_index),
          loss_function_(std::move(loss_function)),
          state_(std::move(state)),
          preintegrators_(preintegrators),
          prior_(prior),
          first_imu_factor_index_(first_imu_factor_index),
          ric_(ric),
          tic_(tic) {
        if (!preintegrators_.empty()) {
            TASSEL_ASSERT(
                preintegrators.size() <= static_cast<size_t>(state_->max_frame_count - 1));
            TASSEL_ASSERT(state_->latest_frame_index == state_->max_frame_count - 1);
        }
    }

    void allocate() {
        landmark_blocks_.clear();
        landmark_blocks_.reserve(retiring_features_.size());
        for (const auto& [_, feature] : retiring_features_) {
            const int num_observations = landmark_target_frame_index_ < 0
                                             ? static_cast<int>(feature.observations.size()) - 1
                                             : 1;
            landmark_blocks_.emplace_back(preintegrators_.empty() ? 6 : 15, loss_function_.get());
            auto& landmark_block = landmark_blocks_.back();
            landmark_block.allocate(
                state_->max_frame_count, num_observations, preintegrators_.empty() ? 6 : 15);
        }
        imu_blocks_.resize(preintegrators_.size());
        for (size_t idx = 0; idx < preintegrators_.size(); ++idx) {
            imu_blocks_[idx].allocate(preintegrators_[idx]);
        }
    }

    void linearize() {
        num_cols_ = state_->max_frame_count * 15 + 1;
        prepareVisualCache();
        for (size_t idx = 0; idx < retiring_features_.size(); ++idx) {
            auto& landmark_block = landmark_blocks_[idx];
            const int landmark_cache_index =
                visual_landmark_cache_indices_.empty() ? -1 : visual_landmark_cache_indices_[idx];
            landmark_block.linearize(
                retiring_features_[idx].second, landmark_target_frame_index_, *state_, ric_, tic_,
                visual_frame_cache_.get(), landmark_cache_index);
        }

        for (size_t i = 0; i < imu_blocks_.size(); ++i) {
            auto& imu_block = imu_blocks_[i];
            const int state_i = first_imu_factor_index_ + static_cast<int>(i);
            const int state_j = state_i + 1;
            const Eigen::Vector3d Q_i = Sophus::SO3d(state_->frames[state_i].R).log();
            const Eigen::Vector3d Q_j = Sophus::SO3d(state_->frames[state_j].R).log();
            imu_block.linearize(
                state_->frames[state_i].V, state_->frames[state_j].V, state_->frames[state_i].P,
                state_->frames[state_j].P, Q_i, Q_j, state_->frames[state_i].Ba,
                state_->frames[state_j].Ba, state_->frames[state_i].Bg, state_->frames[state_j].Bg);
        }
    }

    void marginalizeLandmarks() {
        // 每个逆深度独立边缘化，再与 IMU 因子及旧先验组装。
        for (auto& landmark_block : landmark_blocks_) {
            landmark_block.marginalizeLandmark();
        }
        num_rows_ = static_cast<int>(imu_blocks_.size()) * 15;
        for (const auto& landmark_block : landmark_blocks_) {
            num_rows_ += landmark_block.get_kept_rows();
        }
    }

    void buildReducedSystem(Eigen::MatrixXd& jacobian, Eigen::VectorXd& residual) {
        int total_rows = num_rows_;
        int prior_rows = 0;
        if (prior_) {
            prior_rows = static_cast<int>(prior_->b.size());
            total_rows += prior_rows;
        }

        jacobian = Eigen::MatrixXd::Zero(total_rows, num_cols_);
        residual = Eigen::VectorXd::Zero(total_rows);
        // 全局列布局固定为 [frame0(15), frame1(15), ..., delay(1)]。
        int rows = 0;
        for (size_t idx = 0; idx < landmark_blocks_.size(); ++idx) {
            auto& landmark_block = landmark_blocks_[idx];
            landmark_block.writeReducedSystem(jacobian, residual, rows);
            rows += landmark_block.get_kept_rows();
        }

        for (size_t i = 0; i < imu_blocks_.size(); ++i) {
            auto& imu_block = imu_blocks_[i];
            imu_block.get_dense_Jp_b(
                jacobian, residual, rows, (first_imu_factor_index_ + static_cast<int>(i)) * 15);
            rows += 15;
        }

        if (prior_) {
            // 旧先验使用当前残差和边缘化时冻结的雅各比。
            prior_->validate();
            for (int frame_index = 0; frame_index < prior_->stateCount(); ++frame_index) {
                const int compact_pose = prior_->poseColumn(frame_index);
                const int window_pose = frame_index * MargLinData::StateSize;
                jacobian.block(rows, window_pose, prior_rows, MargLinData::PoseSize) =
                    prior_->H.middleCols(compact_pose, MargLinData::PoseSize);
                if (frame_index > 0) {
                    jacobian.block(
                        rows, window_pose + MargLinData::PoseSize, prior_rows,
                        MargLinData::SpeedBiasSize) =
                        prior_->H.middleCols(
                            prior_->speedBiasColumn(frame_index), MargLinData::SpeedBiasSize);
                }
            }
            jacobian.col(state_->max_frame_count * MargLinData::StateSize)
                .segment(rows, prior_rows) = prior_->H.col(prior_->delayColumn());
            residual.segment(rows, prior_rows) = prior_->b;
        }
    }

private:
    void prepareVisualCache() {
        visual_frame_cache_.reset();
        visual_inverse_depth_params_.clear();
        visual_landmark_cache_indices_.clear();
        if (retiring_features_.empty()) {
            return;
        }

        std::vector<VisualFrameCacheInput> inputs;
        inputs.reserve(static_cast<size_t>(state_->max_frame_count));
        for (int frame_index = 0; frame_index < state_->max_frame_count; ++frame_index) {
            const auto& frame = state_->frames[frame_index];
            VisualFrameCacheInput input;
            input.pose = frame.pose.data();
            input.velocity = Eigen::Map<const Eigen::Vector3d>(frame.speed_bias.data());
            input.accel_bias = Eigen::Map<const Eigen::Vector3d>(frame.speed_bias.data() + 3);
            input.gyro_bias = Eigen::Map<const Eigen::Vector3d>(frame.speed_bias.data() + 6);
            input.gyro = frame.gyro;
            input.acceleration = frame.acc;
            input.sync_delay = frame.sync_delay;
            inputs.push_back(input);
        }

        visual_frame_cache_ = std::make_unique<VisualFrameCache>(
            std::move(inputs), &state_->param_delay_time, ric_, tic_);
        visual_frame_cache_->reserveLandmarks(retiring_features_.size());
        visual_inverse_depth_params_.resize(retiring_features_.size());
        visual_landmark_cache_indices_.reserve(retiring_features_.size());

        for (size_t index = 0; index < retiring_features_.size(); ++index) {
            const Feature& feature = retiring_features_[index].second;
            const std::vector<FeaturePerFrame>& observations = feature.observations;
            if (observations.empty()) {
                throw std::logic_error("Marginalized feature has no observations");
            }
            if (observations[0].sync_delay != state_->frames[feature.host_frame_index].sync_delay) {
                throw std::logic_error("Feature sync delay does not match its host frame state");
            }
            visual_inverse_depth_params_[index] = 1.0 / feature.estimated_depth;
            visual_landmark_cache_indices_.push_back(visual_frame_cache_->addLandmark(
                observations[0].uv, &visual_inverse_depth_params_[index]));

            const int first_observation_index =
                landmark_target_frame_index_ < 0
                    ? 1
                    : landmark_target_frame_index_ - feature.host_frame_index;
            const int last_observation_index = landmark_target_frame_index_ < 0
                                                   ? static_cast<int>(observations.size())
                                                   : first_observation_index + 1;
            for (int observation_index = first_observation_index;
                 observation_index < last_observation_index; ++observation_index) {
                const int target_frame = feature.observationFrameIndex(observation_index);
                if (observations[observation_index].sync_delay !=
                    state_->frames[target_frame].sync_delay) {
                    throw std::logic_error(
                        "Feature sync delay does not match its target frame state");
                }
                visual_frame_cache_->addPair(feature.host_frame_index, target_frame);
            }
        }
        // 边缘化不走 Ceres callback，必须在当前线性化点显式刷新共享中间量。
        visual_frame_cache_->PrepareForEvaluation(true, true);
    }

    std::vector<std::pair<int, Feature>> retiring_features_;
    int landmark_target_frame_index_;
    std::unique_ptr<ceres::LossFunction> loss_function_;

    std::shared_ptr<State> state_;

    std::vector<LandmarkBlock> landmark_blocks_;
    std::vector<IMUBlock<Derived>> imu_blocks_;
    std::vector<IntegratorBase<Derived>*> preintegrators_;
    std::unique_ptr<VisualFrameCache> visual_frame_cache_;
    std::vector<double> visual_inverse_depth_params_;
    std::vector<int> visual_landmark_cache_indices_;

    const MargLinData* prior_ = nullptr;
    int first_imu_factor_index_ = 0;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARGINALIZATION_SQRT_H_
