#ifndef TASSEL_CORE_MARG_MARGINALIZATION_SQRT_H_
#define TASSEL_CORE_MARG_MARGINALIZATION_SQRT_H_

#include <ceres/loss_function.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <sophus/so3.hpp>
#include <vector>

#include "factor/integrator_base.h"
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
    MarginalizationSqrt(
        std::vector<MarginalizedFeatureObservation> retiring_observations,
        std::unique_ptr<ceres::LossFunction> loss_function, std::shared_ptr<State> state,
        std::vector<IntegratorBase<Derived>*>& preintegrators, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic, const MargLinData* prior = nullptr,
        int first_imu_factor_slot = 0)
        : retiring_observations_(std::move(retiring_observations)),
          loss_function_(std::move(loss_function)),
          state_(std::move(state)),
          preintegrators_(preintegrators),
          prior_(prior),
          first_imu_factor_slot_(first_imu_factor_slot),
          ric_(ric),
          tic_(tic) {
        if (!preintegrators_.empty()) {
            TASSEL_ASSERT(
                preintegrators.size() <= static_cast<size_t>(state_->max_frame_count - 1));
            TASSEL_ASSERT(state_->newest_slot == state_->max_frame_count - 1);
        }
    }

    void allocate() {
        landmark_blocks_.clear();
        landmark_blocks_.reserve(retiring_observations_.size());
        for (size_t idx = 0; idx < retiring_observations_.size(); ++idx) {
            landmark_blocks_.emplace_back(preintegrators_.empty() ? 6 : 15, loss_function_.get());
            auto& landmark_block = landmark_blocks_.back();
            landmark_block.allocate(state_->max_frame_count, 1, preintegrators_.empty() ? 6 : 15);
        }
        imu_blocks_.resize(preintegrators_.size());
        for (size_t idx = 0; idx < preintegrators_.size(); ++idx) {
            imu_blocks_[idx].allocate(preintegrators_[idx]);
        }
    }

    void linearize() {
        num_cols_ = state_->max_frame_count * 15 + 1;
        for (size_t idx = 0; idx < retiring_observations_.size(); ++idx) {
            auto& landmark_block = landmark_blocks_[idx];
            landmark_block.linearize(retiring_observations_[idx], *state_, ric_, tic_);
        }

        for (size_t i = 0; i < imu_blocks_.size(); ++i) {
            auto& imu_block = imu_blocks_[i];
            const int state_i = first_imu_factor_slot_ + static_cast<int>(i);
            const int state_j = state_i + 1;
            Eigen::Vector3d Q_i = Sophus::SO3d(state_->frames[state_i].R).log();
            Eigen::Vector3d Q_j = Sophus::SO3d(state_->frames[state_j].R).log();
            imu_block.linearize(
                state_->frames[state_i].V, state_->frames[state_j].V, state_->frames[state_i].P,
                state_->frames[state_j].P, Q_i, Q_j, state_->frames[state_i].Ba,
                state_->frames[state_j].Ba, state_->frames[state_i].Bg, state_->frames[state_j].Bg);
        }
    }

    void eliminateLandmarks() {
        for (auto& landmark_block : landmark_blocks_) {
            landmark_block.eliminateLandmark();
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
        int rows = 0;
        for (size_t idx = 0; idx < landmark_blocks_.size(); ++idx) {
            auto& landmark_block = landmark_blocks_[idx];
            landmark_block.writeReducedSystem(jacobian, residual, rows);
            rows += landmark_block.get_kept_rows();
        }

        for (size_t i = 0; i < imu_blocks_.size(); ++i) {
            auto& imu_block = imu_blocks_[i];
            imu_block.get_dense_Jp_b(
                jacobian, residual, rows, (first_imu_factor_slot_ + static_cast<int>(i)) * 15);
            rows += 15;
        }

        if (prior_) {
            int prior_cols = static_cast<int>(prior_->H.cols());
            const int state_cols = std::min(prior_cols, (state_->max_frame_count - 1) * 15);
            if (state_cols > 0) {
                jacobian.block(rows, 0, prior_rows, state_cols) = prior_->H.leftCols(state_cols);
            }
            if (prior_cols == state_cols + 1) {
                jacobian.col(num_cols_ - 1).segment(rows, prior_rows) =
                    prior_->H.col(prior_cols - 1);
            }
            residual.segment(rows, prior_rows) = prior_->b;
        }
    }

private:
    std::vector<MarginalizedFeatureObservation> retiring_observations_;
    std::unique_ptr<ceres::LossFunction> loss_function_;

    std::shared_ptr<State> state_;

    std::vector<LandmarkBlock> landmark_blocks_;
    std::vector<IMUBlock<Derived>> imu_blocks_;
    std::vector<IntegratorBase<Derived>*> preintegrators_;

    const MargLinData* prior_ = nullptr;
    int first_imu_factor_slot_ = 0;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARGINALIZATION_SQRT_H_
