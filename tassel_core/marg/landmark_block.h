#ifndef TASSEL_CORE_MARG_LANDMARK_BLOCK_H_
#define TASSEL_CORE_MARG_LANDMARK_BLOCK_H_

#include <Eigen/Core>

#include <ceres/loss_function.h>
#include "frond_end/feature.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/types.h"

namespace tassel_core {

class LandmarkBlock {
public:
    LandmarkBlock(int dim = 6, ceres::LossFunction* loss = nullptr);

    void allocate(int num_frames, int num_obs, int dim);

    void linearize(
        const MarginalizedFeatureObservation& observation, const State& state,
        const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void marginalizeLandmark();

    void writeReducedSystem(
        Eigen::MatrixXd& jacobian, Eigen::VectorXd& residual, int start_row) const;

    inline int get_num_rows() const { return num_rows_; }

    inline int get_kept_rows() const {
        TASSEL_ASSERT(qr_performed_);
        return std::max(num_rows_ - marginalized_landmark_rank_, 0);
    }

    inline int get_padding_index() const { return padding_idx_; }

    inline int get_delay_index() const { return delay_idx_; }

    inline int get_landmark_index() const { return lm_idx_; }

    inline int get_residual_index() const { return res_idx_; }

    tassel_utils::MatrixRowMajor& get_mutable_storage() { return storage_; }

    const tassel_utils::MatrixRowMajor& get_storage() const { return storage_; }

private:
    // 列布局为 [各帧状态填充区, 时间延迟, 逆深度, 残差]。
    // 路标边缘化后仅将状态、时间延迟和残差写入全局平方根系统。
    tassel_utils::MatrixRowMajor storage_;
    int delay_idx_;
    int lm_idx_;
    int res_idx_;
    int padding_idx_;
    int num_rows_;
    int marginalized_landmark_rank_;
    bool qr_performed_;
    int dim_;
    ceres::LossFunction* loss_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_LANDMARK_BLOCK_H_
