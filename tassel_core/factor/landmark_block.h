#ifndef TASSEL_CORE_FACTOR_LANDMARK_BLOCK_H_
#define TASSEL_CORE_FACTOR_LANDMARK_BLOCK_H_

#include <Eigen/Core>
#include <array>
#include <vector>

#include <ceres/loss_function.h>
#include "frond_end/feature.h"

namespace tassel_core {

using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class LandmarkBlock {
public:
    LandmarkBlock(double min_depth, ceres::LossFunction* loss);

    void allocate(int num_frames, int num_obs);

    double linearize(
        const Feature& feature, const std::vector<std::array<double, 6>>& lin_poses,
        const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);

    void performQR();

    void get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, int start_row) const;

    int numRows() const { return num_rows_; }
    int keptRows() const { return num_rows_ - 1; }
    int paddingIdx() const { return padding_idx_; }

private:
    MatrixRowMajor storage_;
    int lm_idx_ = 0;
    int res_idx_ = 0;
    int padding_idx_ = 0;
    int num_rows_ = 0;
    double min_depth_;
    ceres::LossFunction* loss_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_LANDMARK_BLOCK_H_
