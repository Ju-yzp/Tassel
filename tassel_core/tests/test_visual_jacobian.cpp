#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "linearization/landmark_block.h"
#include "tassel_tools/jacobian_checker.h"
#include "tassel_utils/utility.h"

namespace {

/// 自定义 SE3 + inv_depth 的 Plus 检查器
/// 观测 uv_host / uv_target 预计算后固定，不再随扰动变化
struct VisualJacobianChecker : public tassel_tools::JacobianChecker {
    VisualJacobianChecker()
        : JacobianChecker(
              /*num_residuals=*/2,
              /*tangent dims=*/std::vector<int>{6, 6, 1},
              /*storage dims=*/std::vector<int>{7, 7, 1}) {}

    /// 固定观测（在 set_observations 中由真值计算）
    Eigen::Vector3d uv_host;
    Eigen::Vector3d uv_target;

    /// 切空间扰动 → 参数空间
    bool plus(const double* x, const double* delta, double* x_out, int block_idx) const override {
        if (block_idx < 2) {
            // SE3: 平移(3) + 旋转(3) — 右扰动
            Eigen::Map<const Eigen::Vector3d> t(x);
            Eigen::Map<const Eigen::Vector3d> rho(delta);
            Eigen::Map<const Eigen::Vector3d> phi(delta + 3);

            Eigen::Map<Eigen::Vector3d> t_out(x_out);
            t_out = t + rho;

            Eigen::Map<const Eigen::Quaterniond> q(x + 3);
            Eigen::Quaterniond dq = tassel_utils::deltaQ(phi);
            Eigen::Map<Eigen::Quaterniond> q_out(x_out + 3);
            q_out = (q * dq).normalized();
            return true;
        } else {
            x_out[0] = x[0] + delta[0];
            return true;
        }
    }

    /// 调用 compute_feature_linearization_block 输出 analytic 结果
    void evaluate(
        const std::vector<double*>& params, double* residuals, double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> host_t(params[0]);
        Eigen::Map<const Eigen::Quaterniond> host_q(params[0] + 3);
        Eigen::Map<const Eigen::Vector3d> target_t(params[1]);
        Eigen::Map<const Eigen::Quaterniond> target_q(params[1] + 3);
        double inv_depth = *params[2];

        Eigen::Matrix3d host_R = host_q.toRotationMatrix();
        Eigen::Matrix3d target_R = target_q.toRotationMatrix();

        Eigen::Vector3d n = uv_target;
        auto tangent_base = tassel_core::LandmarkBlock::compute_tangent_base(n);

        Eigen::Matrix<double, 2, 6> J_host, J_target;
        Eigen::Matrix<double, 2, 1> J_inv, res;
        tassel_core::LandmarkBlock::compute_feature_linearization_block(
            host_R, host_t, target_R, target_t, inv_depth, uv_host, uv_target, tangent_base, J_host,
            J_target, J_inv, res);

        Eigen::Map<Eigen::Vector2d> r_out(residuals);
        r_out = res;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_h_out(jacobians[0]);
                J_h_out = J_host;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_t_out(jacobians[1]);
                J_t_out = J_target;
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Vector2d> J_l_out(jacobians[2]);
                J_l_out = J_inv;
            }
        }
    }
};

void packPose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q, double* dst) {
    dst[0] = p.x();
    dst[1] = p.y();
    dst[2] = p.z();
    dst[3] = q.x();
    dst[4] = q.y();
    dst[5] = q.z();
    dst[6] = q.w();
}

/// 从真值位姿 + inv_depth 预计算固定观测
void computeObservations(
    const Eigen::Vector3d& host_P, const Eigen::Quaterniond& host_Q,
    const Eigen::Vector3d& target_P, const Eigen::Quaterniond& target_Q, double inv_depth,
    Eigen::Vector3d& uv_host, Eigen::Vector3d& uv_target) {
    Eigen::Matrix3d host_R = host_Q.toRotationMatrix();
    Eigen::Matrix3d target_R = target_Q.toRotationMatrix();

    // host 观测: 归一化坐标 (固定)
    uv_host = Eigen::Vector3d(0.1, 0.1, 1.0);

    // target 观测: 由真值计算并固定
    Eigen::Vector3d pt_in_H = uv_host / inv_depth;
    Eigen::Vector3d pt_in_W = host_R * pt_in_H + host_P;
    uv_target = (target_R.transpose() * (pt_in_W - target_P)).normalized();
}

}  // namespace

// ────────────────────────────────────────────────────
// 雅可比检查: 位姿 + 逆深度
// ────────────────────────────────────────────────────
TEST(VisualJacobianTest, AllBlocksPass) {
    Eigen::Vector3d host_P(0, 0, 0);
    Eigen::Quaterniond host_Q = Eigen::Quaterniond::Identity();

    Eigen::Vector3d target_P(0.5, 0.02, 0.0);
    Eigen::Quaterniond target_Q(
        Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()));

    double inv_depth = 0.3333;

    double pose_i[7], pose_j[7];
    packPose(host_P, host_Q, pose_i);
    packPose(target_P, target_Q, pose_j);

    VisualJacobianChecker checker;
    computeObservations(
        host_P, host_Q, target_P, target_Q, inv_depth, checker.uv_host, checker.uv_target);
    checker.set_params({pose_i, pose_j, &inv_depth});
    ASSERT_TRUE(checker.check());
}

TEST(VisualJacobianTest, NonTrivialPose) {
    Eigen::Vector3d host_P(1.2, -0.3, 0.5);
    Eigen::Quaterniond host_Q(
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitX()));

    Eigen::Vector3d target_P(1.8, 0.1, 0.7);
    Eigen::Quaterniond target_Q(
        Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()));

    double inv_depth = 0.5;

    double pose_i[7], pose_j[7];
    packPose(host_P, host_Q, pose_i);
    packPose(target_P, target_Q, pose_j);

    VisualJacobianChecker checker;
    computeObservations(
        host_P, host_Q, target_P, target_Q, inv_depth, checker.uv_host, checker.uv_target);
    checker.set_params({pose_i, pose_j, &inv_depth});
    ASSERT_TRUE(checker.check());
}
