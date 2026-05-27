#include "landmark_block.h"
#include "visual_reprojection.h"

#include <cmath>
#include <sophus/so3.hpp>

namespace tassel_core {

LandmarkBlock::LandmarkBlock(double min_depth, ceres::LossFunction* loss)
    : min_depth_(min_depth), loss_(loss) {}

void LandmarkBlock::allocate(int num_frames, int num_obs) {
    padding_idx_ = num_frames * 6;
    num_rows_ = num_obs * 2;

    int padding_size = padding_idx_ % 4;
    int padding_cols = (padding_size == 0) ? 0 : 4 - padding_size;
    lm_idx_ = padding_idx_ + padding_cols;
    res_idx_ = lm_idx_ + 1;

    storage_.resize(num_rows_, res_idx_ + 1);
    storage_.setZero();
}

double LandmarkBlock::linearize(
    const Feature& feature, const std::vector<std::array<double, 6>>& lin_poses,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    storage_.setZero();

    const std::vector<FeaturePerFrame>& observations = feature.observations;
    Eigen::Vector3d uv_i = observations[0].uv;
    int start_frame_id = feature.start_frame_id;
    double depth = feature.estimated_depth;

    // 深度信息矩阵权重 ∝ 1/depth
    // double weight = min_depth_ / depth;
    // weight = std::clamp(weight, 0.1, 5.0);
    // double sqrt_depth_weight = std::sqrt(weight);

    // host 位姿线性化点
    const auto& pose_arr = lin_poses[start_frame_id];
    Eigen::Vector3d phi_i(pose_arr[0], pose_arr[1], pose_arr[2]);
    Eigen::Vector3d P_i(pose_arr[3], pose_arr[4], pose_arr[5]);
    Eigen::Matrix3d R_i = Sophus::SO3d::exp(phi_i).matrix();

    int num_frames = static_cast<int>(lin_poses.size());
    double error_sum = 0.0;
    for (int offset = 1; offset < static_cast<int>(observations.size()); ++offset) {
        int target_id = start_frame_id + offset;
        const auto& pose_arr_j = lin_poses[target_id];
        Eigen::Vector3d phi_j(pose_arr_j[0], pose_arr_j[1], pose_arr_j[2]);
        Eigen::Vector3d P_j(pose_arr_j[3], pose_arr_j[4], pose_arr_j[5]);
        Eigen::Matrix3d R_j = Sophus::SO3d::exp(phi_j).matrix();

        Eigen::Matrix<double, 2, 6> J_i, J_j;
        Eigen::Matrix<double, 2, 1> J_l;
        Eigen::Vector2d r_raw;
        computeVisualReprojection(
            uv_i, observations[offset].uv, R_i, P_i, R_j, P_j, depth, ric, tic, J_i, J_j, J_l,
            r_raw);

        // IRLS：组合深度权重和鲁棒核函数权重
        double sqrt_loss = 1.0;
        if (loss_) {
            double s = r_raw.squaredNorm();
            double rho[3];
            loss_->Evaluate(s, rho);
            sqrt_loss = std::sqrt(rho[1]);
        }
        double scale = sqrt_loss;
        error_sum += r_raw.squaredNorm();

        int row = (offset - 1) * 2;
        storage_.block(row, start_frame_id * 6, 2, 6) = scale * J_i;
        storage_.block(row, target_id * 6, 2, 6) = scale * J_j;
        storage_.block<2, 1>(row, lm_idx_) = scale * J_l;
        storage_.block<2, 1>(row, res_idx_) = scale * r_raw;
    }
    return error_sum;
}

void LandmarkBlock::performQR() {
    Eigen::JacobiRotation<double> gr;
    for (int m = num_rows_ - 1; m > 0; --m) {
        double a = storage_(m - 1, lm_idx_);
        double b = storage_(m, lm_idx_);
        if (std::abs(a) < 1e-100 && std::abs(b) < 1e-100) {
            continue;
        }
        gr.makeGivens(a, b);
        storage_.applyOnTheLeft(m, m - 1, gr);
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
