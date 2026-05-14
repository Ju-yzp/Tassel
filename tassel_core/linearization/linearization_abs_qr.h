#ifndef TASSEL_CORE_LINEARIZATION_ABS_QR_H_
#define TASSEL_CORE_LINEARIZATION_ABS_QR_H_

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tbb/task_arena.h>

#include "frond_end/feature.h"
#include "frond_end/feature_manager.h"
#include "linearization/landmark_block.h"
#include "loss_fuction/loss_fuction_base.h"
#include "marginalization/marg_linearized_data.h"
#include "state/state.h"

namespace tassel_core {

class LinearizationAbsQR {
public:
    LinearizationAbsQR(
        int num_threads, std::shared_ptr<State> state, std::shared_ptr<FeatureManager> fm,
        LossVariant reprojection_loss, DepthLoss depth_loss, double min_depth, double max_depth,
        std::shared_ptr<MargLinData> marg_lin_data = nullptr);

    double linearizeProbelm(bool* numerically_valid = nullptr);

    void performQR();

    void setPoseDamping(double lambda);

    void get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r) const;

    void get_dense_H_b(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;

    bool hasPoseDamping() const { return pose_damping_diagonal_ > 0; }

    double backSubstitute(const Eigen::VectorXd& pose_inc);

    State* getState() const { return cur_state_.get(); }

    int getPoseDim() const;

    double computeError() const;

    void saveState();

    void restoreState();

    void applyPoseInc(const Eigen::VectorXd& inc);

protected:
    void get_dense_Q2Jp_Q2r_pose_damping(Eigen::MatrixXd& Q2Jp, size_t start_idx) const;

    void get_dense_Q2Jp_Q2r_marg_prior(
        Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, size_t start_idx) const;

    void add_dense_H_b_pose_damping(Eigen::MatrixXd& H) const;

    void add_dense_H_b_marg_prior(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;

private:
    mutable tbb::task_arena thread_pool_;

    std::shared_ptr<State> cur_state_;
    std::shared_ptr<FeatureManager> feature_manager_;
    std::shared_ptr<MargLinData> marg_lin_data_;

    LossVariant reprojection_loss_;
    DepthLoss depth_loss_;

    std::vector<LandmarkBlock> landmark_blocks_;

    double pose_damping_diagonal_ = 0;
    double pose_damping_diagonal_sqrt_ = 0;
    Eigen::VectorXd marg_scaling_;

    std::unordered_map<Feature*, double> saved_feature_depths_;
    std::vector<uint8_t> saved_lms_states_;

    double min_depth_, max_depth_;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_LINEARIZATION_ABS_QR_H_
