#ifndef TASSEL_CORE_LINEARIZATION_LANDMARK_BLOCK_H_
#define TASSEL_CORE_LINEARIZATION_LANDMARK_BLOCK_H_

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "frond_end/feature.h"
#include "loss_fuction/loss_fuction_base.h"
#include "state/state.h"

namespace tassel_core {
Eigen::Matrix<double, 2, 3> compute_tangent_base(Eigen::Vector3d uv);

void compute_feature_linearization_block(
    const Sophus::SE3d& T_w_h, const Sophus::SE3d& T_w_t, double depth,
    const Eigen::Vector3d& uv_host, const Eigen::Vector3d& uv_target,
    const Eigen::Matrix<double, 2, 3>& tangent_base, Eigen::Matrix<double, 2, 6>& jacobian_H,
    Eigen::Matrix<double, 2, 6>& jacobian_T, Eigen::Matrix<double, 2, 1>& jacobian_L,
    Eigen::Matrix<double, 2, 1>& residual);

class LandmarkBlock {
public:
    LandmarkBlock();

    void allocate(
        Feature* const feature, State* const state, const LossVariant& reprojection_loss,
        const DepthLoss& depth_loss);

    double linearize();

    void setLandmarkDamping(double lambda);

    void performQR();

    inline bool hasLandmarkDamping() const { return !damping_rotations_.empty(); }

    void get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, size_t start_idx) const;

    void add_dense_H_b(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& getStorage()
        const {
        return storage_;
    }

    void backSubstitute(const Eigen::VectorXd& pose_inc, double& l_diff);

    void addJp_diag2(Eigen::VectorXd& res) const;

    void scaleJl_cols();

    void scaleJp_cols(const Eigen::VectorXd& jacobian_scaling);

    inline size_t numQ2rows() const { return num_rows_ - 3; }

    int getLandmarkCol() const { return lm_idx_; }
    int getPaddingIdx() const { return padding_idx_; }
    int getNumCols() const { return num_cols_; }
    int getNumRows() const { return num_rows_; }

private:
    enum class LandmarkState {
        Uninitialized = 0,
        Initialized = 1,
        Linearized = 2,
        Marginalized = 3,
        NumbericalFailure = 4,
        Allocated = 5
    };

    void evaluate(
        int frame_id_H, int frame_id_T, Eigen::Matrix<double, 2, 6>& jacobian_H,
        Eigen::Matrix<double, 2, 6>& jacobian_T, Eigen::Matrix<double, 2, 1>& jacobian_L,
        Eigen::Matrix<double, 2, 1>& residual);

    // | J_p 2x6N | padding | J_l 2x1 | residual 2x1 |
    // | damping 3 x cols|
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage_;

    std::vector<Eigen::JacobiRotation<double>> damping_rotations_;

    int lm_idx_;
    int res_idx_;
    int padding_idx_;
    int padding_size_;

    int num_cols_, num_rows_;

    State* state_;
    Feature* feature_;

    std::vector<Eigen::Matrix<double, 2, 3>> tangent_base_vec_;

    double Jl_col_scale_;

    LandmarkState lms_;

    LossVariant reprojection_loss_;
    DepthLoss depth_loss_;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_LINEARIZATION_LANDMARK_BLOCK_H_
