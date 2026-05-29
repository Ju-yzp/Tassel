#ifndef TASSEL_CORE_FACTOR_LANDMARK_BLOCK_H_
#define TASSEL_CORE_FACTOR_LANDMARK_BLOCK_H_

#include <Eigen/Core>

#include <ceres/loss_function.h>
#include "frond_end/feature.h"

namespace tassel_core {

using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class LandmarkBlock {
public:
    LandmarkBlock(int dim = 6, ceres::LossFunction* loss = nullptr);

    void allocate(int num_frames, int num_obs, int dim);

    void linearize(
        const Feature& feature, const State& state, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic);

    void performQR();

    void get_dense_Q2Jp_Q2r(Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, int start_row) const;

    int numRows() const { return num_rows_; }

    int keptRows() const { return num_rows_ - 1; }

    int paddingIdx() const { return padding_idx_; }

    int landmarkIdx() const { return lm_idx_; }

    int residualIdx() const { return res_idx_; }

    MatrixRowMajor& mutableStorage() { return storage_; }

    const MatrixRowMajor& storage() const { return storage_; }

private:
    MatrixRowMajor storage_;
    int lm_idx_;
    int res_idx_;
    int padding_idx_;
    int num_rows_;
    int dim_;
    ceres::LossFunction* loss_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_FACTOR_LANDMARK_BLOCK_H_
