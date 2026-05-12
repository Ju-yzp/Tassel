#ifndef TASSEL_CORE_LINEARIZATION_ABS_QR_H_
#define TASSEL_CORE_LINEARIZATION_ABS_QR_H_

// cpp
#include <cstddef>
#include <memory>
#include <vector>

// tbb
#include <tbb/task_arena.h>

#include "frond_end/feature_manager.h"
#include "linearization/landmark_block.h"
#include "marginalization/marg_helper.h"
#include "marginalization/marg_linearized_data.h"
#include "state/state.h"

namespace tassel_core {
// VO 版本
class LinearizationAbsQR {
public:
    explicit LinearizationAbsQR(int num_threads = 1) : thread_pool_(num_threads) {}

    void setFeatureManager(std::shared_ptr<FeatureManager> fm) { feature_manager_ = std::move(fm); }
    void setState(std::shared_ptr<State> s) { state_ = std::move(s); }

    double linearizeProbelm();

    void performQR();

    void setPoseDamping(const double lambda);

    void get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r) const;

    void get_dense_H_b(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;

    bool hasPoseDamping() const { return pose_damping_diagonal_ > 0; }

    double backSubstitute(const Eigen::VectorXd& pose_inc);

    State* getState() const { return state_.get(); }

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

    std::shared_ptr<FeatureManager> feature_manager_;

    std::shared_ptr<MargLinData> marg_lin_data_;

    std::shared_ptr<State> state_;

    std::vector<LandmarkBlock> landmark_blocks_;

    double pose_damping_diagonal_;
    double pose_damping_diagonal_sqrt_;
    Eigen::VectorXd marg_scaling_;

    std::unique_ptr<MargHelper> marg_helper_;

    // backup for LM rejection
    std::unordered_map<Feature*, double> saved_feature_depths_;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_LINEARIZATION_ABS_QR_H_ */
