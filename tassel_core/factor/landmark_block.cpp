#include "landmark_block.h"
#include "factor/visual_factor.h"

#include <Eigen/Core>
#include <cmath>
#include <sophus/so3.hpp>
#include <vector>

namespace tassel_core {

LandmarkBlock::LandmarkBlock(int dim, ceres::LossFunction* loss)
    : lm_idx_(0), res_idx_(0), padding_idx_(0), num_rows_(0), dim_(dim), loss_(loss) {}

void LandmarkBlock::allocate(int num_frames, int num_obs, int dim) {
    padding_idx_ = num_frames * dim;
    num_rows_ = num_obs * 2;

    int padding_size = padding_idx_ % 4;
    int padding_cols = (padding_size == 0) ? 0 : 4 - padding_size;
    lm_idx_ = padding_idx_ + padding_cols;
    res_idx_ = lm_idx_ + 1;

    storage_.resize(num_rows_, res_idx_ + 1);
    storage_.setZero();
}

void LandmarkBlock::linearize(const Feature& feature, const State& state) {
    storage_.setZero();

    const std::vector<FeaturePerFrame>& observations = feature.observations;
    Eigen::Vector3d uv_i = observations[0].uv;
    int start_frame_id = feature.start_frame_id;
    double depth = feature.estimated_depth;

    Eigen::Matrix3d ric = state.ric;
    Eigen::Vector3d tic = state.tic;
    Eigen::Matrix<double, 2, 6> jacobian_pose_i, jacobian_pose_j;
    Eigen::Matrix<double, 2, 1> jacobian_dt;
    Eigen::Matrix<double, 2, 1> jacobian_landmark;
    Eigen::Matrix<double, 2, 1> residual;

    Eigen::Matrix2d sqrt_info = state.visual_sqrt_info;
    for (int offset = 1; offset < static_cast<int>(observations.size()); ++offset) {
        int target_id = start_frame_id + offset;
        Eigen::Vector2d pt_j(observations[offset].pt.x, observations[offset].pt.y);
        double inv_depth = 1.0 / depth;
        auto visual_factor = new VisualFactor(
            uv_i, pt_j, ric, tic, state.gyro_vec[start_frame_id], state.gyro_vec[target_id],
            state.acc_vec[start_frame_id], state.acc_vec[target_id],
            state.params_speed_bias[start_frame_id].data(),
            state.params_speed_bias[target_id].data(),
            state.params_speed_bias[start_frame_id].data() + 6,
            state.params_speed_bias[target_id].data() + 6,
            state.params_speed_bias[start_frame_id].data() + 3,
            state.params_speed_bias[target_id].data() + 3, sqrt_info, state.camera);

        std::vector<double*> jacobians;
        jacobians.push_back(jacobian_pose_i.data());
        jacobians.push_back(jacobian_pose_j.data());
        jacobians.push_back(jacobian_dt.data());
        jacobians.push_back(jacobian_landmark.data());

        std::vector<double const*> parameters;
        parameters.push_back(state.params_pose[start_frame_id].data());
        parameters.push_back(state.params_pose[target_id].data());
        parameters.push_back(&state.param_delay_time);
        parameters.push_back(&inv_depth);

        visual_factor->Evaluate(parameters.data(), residual.data(), jacobians.data());
        delete visual_factor;

        double sqrt_loss = 1.0;
        if (loss_) {
            double s = residual.squaredNorm();
            double rho[3];
            loss_->Evaluate(s, rho);
            sqrt_loss = std::sqrt(rho[1]);
        }
        double scale = sqrt_loss;
        int row = (offset - 1) * 2;
        storage_.block<2, 6>(row, start_frame_id * dim_) = scale * jacobian_pose_i;
        storage_.block<2, 6>(row, target_id * dim_) = scale * jacobian_pose_j;
        storage_.block<2, 1>(row, lm_idx_) = scale * jacobian_landmark;
        storage_.block<2, 1>(row, res_idx_) = scale * residual;
    }
}

void LandmarkBlock::performQR() {
    int n = num_rows_;
    if (n <= 1) return;

    Eigen::VectorXd v = storage_.col(lm_idx_).head(n);
    double norm = v.norm();
    if (norm < 1e-12) return;

    double alpha = (v(0) > 0) ? -norm : norm;
    v(0) -= alpha;
    double beta = 2.0 / v.squaredNorm();

    for (int j = 0; j <= res_idx_; ++j) {
        double gamma = beta * v.dot(storage_.col(j).head(n));
        storage_.col(j).head(n) -= gamma * v;
    }
}

void LandmarkBlock::get_dense_Q2Jp_Q2r(
    Eigen::MatrixXd& Q2Jp, Eigen::VectorXd& Q2r, int start_row) const {
    int kept_rows = num_rows_ - 1;
    Q2r.segment(start_row, kept_rows) = storage_.col(res_idx_).tail(kept_rows);
    Q2Jp.block(start_row, 0, kept_rows, padding_idx_) =
        storage_.block(1, 0, kept_rows, padding_idx_);
}

}  // namespace tassel_core
