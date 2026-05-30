#ifndef TASSEL_CORE_FACTOR_MARGINLIZATION_SQRT_H_
#define TASSEL_CORE_FACTOR_MARGINLIZATION_SQRT_H_

#include <ceres/loss_function.h>
#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <vector>

#include "factor/integrator_base.h"
#include "factor/landmark_block.h"
#include "frond_end/feature_manager.h"
#include "imu_block.h"
#include "tassel_utils/macros.h"

namespace tassel_core {

struct MargLinData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
};

// TODO:
// 用户在进行边缘化的sqrt操作时，需要保证预积分器使用最近的线性化点进行了更新，使用新的线性化点
template <typename Derived>
class MarginlizationSqrt {
public:
    MarginlizationSqrt(
        std::shared_ptr<FeatureManager> feature_manager,
        std::unique_ptr<ceres::LossFunction> loss_function, std::shared_ptr<State> state,
        std::vector<IntegratorBase<Derived>*>& preintegrators) {
        feature_manager_ = feature_manager;
        loss_function_ = std::move(loss_function);
        state_ = state;
        preintegrators_ = preintegrators;
        if (!preintegrators_.empty()) {
            TASSEL_ASSERT(preintegrators.size() == state_->max_frame_count - 1);
            TASSEL_ASSERT(state_->cur_frame_count == state_->max_frame_count - 1);
        }
    }

    void allocate() {
        const std::vector<Feature*> marg_features =
            feature_manager_->collectMarginalizationFeatures();
        landmark_blocks_.resize(marg_features.size());
        for (size_t idx = 0; idx < marg_features.size(); ++idx) {
            auto& lmb = landmark_blocks_[idx];
            lmb.allocate(
                state_->max_frame_count,
                static_cast<int>(marg_features[idx]->observations.size()) - 1,
                preintegrators_.empty() ? 6 : 15);
        }
        imu_blocks_.resize(preintegrators_.size());
        for (size_t idx = 0; idx < preintegrators_.size(); ++idx) {
            imu_blocks_[idx].allocate(preintegrators_[idx]);
        }
    }

    void linearize() {
        const std::vector<Feature*> marg_features =
            feature_manager_->collectMarginalizationFeatures();

        for (size_t idx = 0; idx < marg_features.size(); ++idx) {
            auto& lmb = landmark_blocks_[idx];
            lmb.linearize(*marg_features[idx], *state_);
        }

        // for () {
        // }
    }

    void get_dense_Jp_b(Eigen::MatrixXd Jp, Eigen::VectorXd b) {
        // for () {

        // }
    }

private:
    std::shared_ptr<FeatureManager> feature_manager_;
    std::unique_ptr<ceres::LossFunction> loss_function_;

    std::shared_ptr<State> state_;

    std::vector<LandmarkBlock> landmark_blocks_;
    std::vector<IMUBlock<Derived>> imu_blocks_;
    std::vector<IntegratorBase<Derived>*> preintegrators_;
};
}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_MARGINLIZATION_SQRT_H_
