#ifndef TASSEL_CORE_MARG_MARGINLIZATION_SQRT_H_
#define TASSEL_CORE_MARG_MARGINLIZATION_SQRT_H_

#include <ceres/loss_function.h>
#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <sophus/so3.hpp>
#include <vector>

#include "factor/integrator_base.h"
#include "frond_end/feature.h"
#include "marg/imu_block.h"
#include "marg/landmark_block.h"
#include "marg/marg_lin_data.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

// TODO:
// 用户在进行边缘化的sqrt操作时，需要保证预积分器使用最近的线性化点进行了更新，使用新的线性化点
template <typename Derived>
class MarginlizationSqrt {
public:
    MarginlizationSqrt(
        std::vector<Feature*> marg_features, std::unique_ptr<ceres::LossFunction> loss_function,
        std::shared_ptr<State> state, std::vector<IntegratorBase<Derived>*>& preintegrators,
        const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const MargLinData* prior = nullptr)
        : prior_(prior), ric_(ric), tic_(tic) {
        marg_features_ = std::move(marg_features);
        loss_function_ = std::move(loss_function);
        state_ = state;
        preintegrators_ = preintegrators;
        if (!preintegrators_.empty()) {
            TASSEL_ASSERT(preintegrators.size() <= state_->max_frame_count - 1);
            TASSEL_ASSERT(state_->cur_frame_count == state_->max_frame_count - 1);
        }
    }

    void allocate() {
        landmark_blocks_.clear();
        landmark_blocks_.reserve(marg_features_.size());
        for (size_t idx = 0; idx < marg_features_.size(); ++idx) {
            landmark_blocks_.emplace_back(preintegrators_.empty() ? 6 : 15, loss_function_.get());
            auto& lmb = landmark_blocks_.back();
            lmb.allocate(
                state_->max_frame_count,
                static_cast<int>(marg_features_[idx]->observations.size()) - 1,
                preintegrators_.empty() ? 6 : 15);
        }
        imu_blocks_.resize(preintegrators_.size());
        for (size_t idx = 0; idx < preintegrators_.size(); ++idx) {
            imu_blocks_[idx].allocate(preintegrators_[idx]);
        }
    }

    void linearize() {
        num_cols = state_->max_frame_count * 15;
        num_rows = 0;
        for (size_t idx = 0; idx < marg_features_.size(); ++idx) {
            auto& lmb = landmark_blocks_[idx];
            lmb.linearize(*marg_features_[idx], *state_, ric_, tic_);
            num_rows += lmb.get_kept_rows();
        }

        for (size_t i = 0; i < imu_blocks_.size(); ++i) {
            auto& imu_block = imu_blocks_[i];
            Eigen::Vector3d Q_i = Sophus::SO3d(state_->Rs[i]).log();
            Eigen::Vector3d Q_j = Sophus::SO3d(state_->Rs[i + 1]).log();
            imu_block.linearize(
                state_->Vs[i], state_->Vs[i + 1], state_->Ps[i], state_->Ps[i + 1], Q_i, Q_j,
                state_->Bas[i], state_->Bas[i + 1], state_->Bgs[i], state_->Bgs[i + 1]);
            num_rows += 15;
        }
    }

    void performQRAll() {
        for (auto& lmb : landmark_blocks_) {
            lmb.performQR();
        }
    }

    void get_dense_Jp_b(Eigen::MatrixXd& Jp, Eigen::VectorXd& b) {
        int total_rows = num_rows;
        int prior_rows = 0;
        if (prior_) {
            prior_rows = static_cast<int>(prior_->b.size());
            total_rows += prior_rows;
        }

        Jp = Eigen::MatrixXd::Zero(total_rows, num_cols);
        b = Eigen::VectorXd::Zero(total_rows);
        int rows = 0;
        for (size_t idx = 0; idx < landmark_blocks_.size(); ++idx) {
            auto& lmb = landmark_blocks_[idx];
            lmb.get_dense_Q2Jp_Q2r(Jp, b, rows);
            rows += lmb.get_kept_rows();
        }

        for (size_t i = 0; i < imu_blocks_.size(); ++i) {
            auto& imu_block = imu_blocks_[i];
            imu_block.get_dense_Jp_b(Jp, b, rows, i * 15);
            rows += 15;
        }

        if (prior_) {
            int prior_cols = static_cast<int>(prior_->H.cols());
            Jp.block(rows, 0, prior_rows, prior_cols) = prior_->H;
            b.segment(rows, prior_rows) = prior_->b;
        }
    }

private:
    std::vector<Feature*> marg_features_;
    std::unique_ptr<ceres::LossFunction> loss_function_;

    std::shared_ptr<State> state_;

    std::vector<LandmarkBlock> landmark_blocks_;
    std::vector<IMUBlock<Derived>> imu_blocks_;
    std::vector<IntegratorBase<Derived>*> preintegrators_;

    const MargLinData* prior_ = nullptr;
    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;
    int num_rows;
    int num_cols;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_MARGINLIZATION_SQRT_H_
