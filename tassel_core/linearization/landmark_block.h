#ifndef TASSEL_CORE_LINEARIZATION_LANDMARK_BLOCK_H_
#define TASSEL_CORE_LINEARIZATION_LANDMARK_BLOCK_H_

#include <Eigen/Core>

#include "feature/feature.h"
#include "state/state.h"

namespace tassel_core {
class LandmarkBlock {
public:
    using DR_DQ = Eigen::Matrix<double, 2, 6>;
    using DR_DL = Eigen::Matrix<double, 2, 1>;
    using Residual = Eigen::Matrix<double, 2, 1>;

    const int POSE_SIZE = 6;

    LandmarkBlock();

    static Eigen::Matrix<double, 2, 3> compute_tangent_base(Eigen::Vector3d uv);

    static void compute_feature_linearization_block(
        const Eigen::Matrix3d& host_R, const Eigen::Vector3d& host_P,
        const Eigen::Matrix3d& target_R, const Eigen::Vector3d& target_P, const double inv_depth,
        const Eigen::Vector3d& uv_host, const Eigen::Vector3d& uv_target,
        const Eigen::Matrix<double, 2, 3>& tangent_base, Eigen::Matrix<double, 2, 6>& jacobian_H,
        Eigen::Matrix<double, 2, 6>& jacobian_T, Eigen::Matrix<double, 2, 1>& jacobian_L,
        Eigen::Matrix<double, 2, 1>& residual);

    void allocate(Feature* const feature, State* const state);

    double linearize();

    void performQR();

    void get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, size_t start_idx) const;

    void add_dense_H_b(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& getStorage()
        const {
        return storage_;
    }
    int getLandmarkCol() const { return lm_idx_; }
    int getPaddingIdx() const { return padding_idx_; }
    int getNumCols() const { return num_cols_; }
    int getNumRows() const { return num_rows_; }

private:
    void evaluate(
        int frame_id_H, int frame_id_T, DR_DQ& jacobian_H, DR_DQ& jacobian_T, DR_DL& jacobian_L,
        Residual& residual);

    // 存储残差对于位姿部分，残差对于逆深度的雅各比矩阵，以及残差
    // | J_p 2x6N | padding | J_l 2x1 | residual 2x1 |
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage_;

    int lm_idx_;   // 逆深度的列索引
    int res_idx_;  // 残差的列索引
    int padding_idx_;

    State* state_;      // 状态量
    Feature* feature_;  // 特征点

    int num_rows_, num_cols_;
    std::vector<Eigen::Matrix<double, 2, 3>> tangent_base_vec_;
};
}  // namespace tassel_core
#endif  // TASSEL_CORE_LINEARIZATION_LANDMARK_BLOCK_H_
