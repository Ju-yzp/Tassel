#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <random>

#include <sophus/so3.hpp>

#include <ceres/ceres.h>
#include <ceres/numeric_diff_cost_function.h>

#include "factor/visual_factor.h"

namespace tassel_core {
namespace {

constexpr double kJacRelTol = 5e-4;
constexpr double kTdRelTol = 3e-2;

class VisualFactorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ric_ = Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        tic_ = Eigen::Vector3d(0.04, -0.02, 0.01);

        R_h_ = (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitX()))
                   .toRotationMatrix();
        P_h_ = Eigen::Vector3d(0.2, -0.1, 0.05);

        R_t_ = (Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ()))
                   .toRotationMatrix();
        P_t_ = Eigen::Vector3d(1.2, 0.15, -0.08);

        depth_gt_ = 3.8;
        uv_i_ = Eigen::Vector3d(0.2, -0.12, 1.0);

        w_i_ = Eigen::Vector3d(1.2, -0.6, 0.3);
        w_j_ = Eigen::Vector3d(1.1, -0.5, 0.25);
        a_i_ = Eigen::Vector3d(0.3, 0.1, 9.81);
        a_j_ = Eigen::Vector3d(0.25, 0.15, 9.75);
        v_i_[0] = 0.8;
        v_i_[1] = -0.3;
        v_i_[2] = 0.1;
        v_j_[0] = 0.85;
        v_j_[1] = -0.35;
        v_j_[2] = 0.12;
        bg_i_[0] = 0.01;
        bg_i_[1] = -0.005;
        bg_i_[2] = 0.003;
        bg_j_[0] = 0.012;
        bg_j_[1] = -0.004;
        bg_j_[2] = 0.002;

        td_arr_[0] = 0.008;

        double dt = td_arr_[0];
        Eigen::Vector3d V_i(v_i_[0], v_i_[1], v_i_[2]);
        Eigen::Vector3d V_j(v_j_[0], v_j_[1], v_j_[2]);
        Eigen::Vector3d Bg_i(bg_i_[0], bg_i_[1], bg_i_[2]);
        Eigen::Vector3d Bg_j(bg_j_[0], bg_j_[1], bg_j_[2]);

        Eigen::Vector3d pi_in_C = uv_i_ * depth_gt_;
        Eigen::Vector3d pi_in_I = ric_ * pi_in_C + tic_;
        Eigen::Vector3d pi_in_G = R_h_ * Sophus::SO3d::exp((w_i_ - Bg_i) * dt).matrix() * pi_in_I +
                                  P_h_ + V_i * dt + 0.5 * R_h_ * a_i_ * dt * dt;
        Eigen::Vector3d pj_in_I = Sophus::SO3d::exp((Bg_j - w_j_) * dt).matrix() *
                                  R_t_.transpose() *
                                  (pi_in_G - (P_t_ + V_j * dt + 0.5 * R_t_ * a_j_ * dt * dt));
        Eigen::Vector3d pj_in_C = ric_.transpose() * (pj_in_I - tic_);
        uv_j_ = pj_in_C / pj_in_C.z();

        inv_depth_gt_ = 1.0 / depth_gt_;
        inv_depth_arr_[0] = inv_depth_gt_;

        Sophus::SO3d so3_h(R_h_);
        Sophus::SO3d so3_t(R_t_);
        Eigen::Vector3d phi_h = so3_h.log();
        Eigen::Vector3d phi_t = so3_t.log();
        for (int i = 0; i < 3; ++i) {
            host_pose_[i] = phi_h[i];
            host_pose_[i + 3] = P_h_[i];
            target_pose_[i] = phi_t[i];
            target_pose_[i + 3] = P_t_[i];
        }

        sqrt_info_ = Eigen::Matrix2d::Identity();
    }

    void evalAnalytical(double* J_pose_i, double* J_pose_j, double* J_depth, double* J_td) {
        const double* params[] = {host_pose_, target_pose_, inv_depth_arr_, td_arr_};
        double* jacs[] = {J_pose_i, J_pose_j, J_depth, J_td};
        double r[2];
        factor_->Evaluate(params, r, jacs);

        Eigen::Vector3d phi_h(host_pose_[0], host_pose_[1], host_pose_[2]);
        Eigen::Vector3d phi_t(target_pose_[0], target_pose_[1], target_pose_[2]);
        Eigen::Matrix3d Jr_h = Sophus::SO3d::leftJacobian(phi_h).transpose();
        Eigen::Matrix3d Jr_t = Sophus::SO3d::leftJacobian(phi_t).transpose();

        if (J_pose_i) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Ji(J_pose_i);
            Ji.block<2, 3>(0, 0) = Ji.block<2, 3>(0, 0) * Jr_h;
        }
        if (J_pose_j) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Jj(J_pose_j);
            Jj.block<2, 3>(0, 0) = Jj.block<2, 3>(0, 0) * Jr_t;
        }
    }

    void evalNumerical(double* J_pose_i, double* J_pose_j, double* J_depth, double* J_td) {
        const double* params[] = {host_pose_, target_pose_, inv_depth_arr_, td_arr_};
        double* jacs[] = {J_pose_i, J_pose_j, J_depth, J_td};
        double r[2];
        num_factor_->Evaluate(params, r, jacs);
    }

    void createFactor() {
        factor_ = std::make_unique<VisualFactor>(
            uv_i_, uv_j_, ric_, tic_, w_i_, w_j_, a_i_, a_j_, v_i_, v_j_, bg_i_, bg_j_, sqrt_info_);
        num_factor_ = std::make_unique<
            ceres::NumericDiffCostFunction<VisualFactor, ceres::CENTRAL, 2, 6, 6, 1, 1>>(
            new VisualFactor(
                uv_i_, uv_j_, ric_, tic_, w_i_, w_j_, a_i_, a_j_, v_i_, v_j_, bg_i_, bg_j_,
                sqrt_info_),
            ceres::TAKE_OWNERSHIP);
    }

    Eigen::Matrix3d ric_, R_h_, R_t_;
    Eigen::Vector3d tic_, P_h_, P_t_, uv_i_, uv_j_;
    double depth_gt_, inv_depth_gt_;
    double host_pose_[6], target_pose_[6], inv_depth_arr_[1], td_arr_[1];

    Eigen::Vector3d w_i_, w_j_, a_i_, a_j_;
    double v_i_[3], v_j_[3], bg_i_[3], bg_j_[3];
    Eigen::Matrix2d sqrt_info_;

    std::unique_ptr<VisualFactor> factor_;
    std::unique_ptr<ceres::NumericDiffCostFunction<VisualFactor, ceres::CENTRAL, 2, 6, 6, 1, 1>>
        num_factor_;
};

TEST_F(VisualFactorTest, ResidualAtGroundTruth) {
    createFactor();
    const double* params[] = {host_pose_, target_pose_, inv_depth_arr_, td_arr_};
    double r[2];
    factor_->Evaluate(params, r, nullptr);
    EXPECT_NEAR(r[0], 0.0, 1e-8);
    EXPECT_NEAR(r[1], 0.0, 1e-8);
}

TEST_F(VisualFactorTest, JacobianPoseHost) {
    createFactor();
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_ana, J_num;
    J_ana.setZero();
    J_num.setZero();
    evalAnalytical(J_ana.data(), nullptr, nullptr, nullptr);
    evalNumerical(J_num.data(), nullptr, nullptr, nullptr);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 6; ++c) {
            double d = std::max(1e-6, std::abs(J_num(r, c)));
            EXPECT_NEAR(J_ana(r, c), J_num(r, c), kJacRelTol * d)
                << "Host pose J[" << r << "," << c << "]";
        }
}

TEST_F(VisualFactorTest, JacobianPoseTarget) {
    createFactor();
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_ana, J_num;
    J_ana.setZero();
    J_num.setZero();
    evalAnalytical(nullptr, J_ana.data(), nullptr, nullptr);
    evalNumerical(nullptr, J_num.data(), nullptr, nullptr);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 6; ++c) {
            double d = std::max(1e-6, std::abs(J_num(r, c)));
            EXPECT_NEAR(J_ana(r, c), J_num(r, c), kJacRelTol * d)
                << "Target pose J[" << r << "," << c << "]";
        }
}

TEST_F(VisualFactorTest, JacobianDepth) {
    createFactor();
    Eigen::Matrix<double, 2, 1> J_ana, J_num;
    J_ana.setZero();
    J_num.setZero();
    evalAnalytical(nullptr, nullptr, J_ana.data(), nullptr);
    evalNumerical(nullptr, nullptr, J_num.data(), nullptr);
    for (int r = 0; r < 2; ++r) {
        double d = std::max(1e-6, std::abs(J_num(r, 0)));
        EXPECT_NEAR(J_ana(r, 0), J_num(r, 0), kJacRelTol * d) << "Depth J[" << r << "]";
    }
}

TEST_F(VisualFactorTest, JacobianTd) {
    createFactor();
    Eigen::Matrix<double, 2, 1> J_ana, J_num;
    J_ana.setZero();
    J_num.setZero();
    evalAnalytical(nullptr, nullptr, nullptr, J_ana.data());
    evalNumerical(nullptr, nullptr, nullptr, J_num.data());
    for (int r = 0; r < 2; ++r) {
        double d = std::max(1e-6, std::abs(J_num(r, 0)));
        EXPECT_NEAR(J_ana(r, 0), J_num(r, 0), kTdRelTol * d) << "TD J[" << r << "]";
    }
}

TEST_F(VisualFactorTest, JacobianAllRandomPerturb) {
    std::mt19937 rng(42);
    std::normal_distribution<double> pose_noise(0.0, 0.02);
    std::normal_distribution<double> depth_noise(0.0, 0.05);
    std::normal_distribution<double> td_noise(0.0, 0.002);

    for (int i = 0; i < 6; ++i) {
        host_pose_[i] += pose_noise(rng);
        target_pose_[i] += pose_noise(rng);
    }
    inv_depth_arr_[0] += depth_noise(rng);
    td_arr_[0] += td_noise(rng);

    createFactor();

    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_ana_pose_i, J_num_pose_i;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_ana_pose_j, J_num_pose_j;
    Eigen::Matrix<double, 2, 1> J_ana_depth, J_num_depth;
    Eigen::Matrix<double, 2, 1> J_ana_td, J_num_td;

    J_ana_pose_i.setZero();
    J_num_pose_i.setZero();
    J_ana_pose_j.setZero();
    J_num_pose_j.setZero();
    J_ana_depth.setZero();
    J_num_depth.setZero();
    J_ana_td.setZero();
    J_num_td.setZero();

    evalAnalytical(J_ana_pose_i.data(), J_ana_pose_j.data(), J_ana_depth.data(), J_ana_td.data());
    evalNumerical(J_num_pose_i.data(), J_num_pose_j.data(), J_num_depth.data(), J_num_td.data());

    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 6; ++c) {
            double d = std::max(1e-6, std::abs(J_num_pose_i(r, c)));
            EXPECT_NEAR(J_ana_pose_i(r, c), J_num_pose_i(r, c), kJacRelTol * d)
                << "Perturb pose i [" << r << "," << c << "]";
            d = std::max(1e-6, std::abs(J_num_pose_j(r, c)));
            EXPECT_NEAR(J_ana_pose_j(r, c), J_num_pose_j(r, c), kJacRelTol * d)
                << "Perturb pose j [" << r << "," << c << "]";
        }
    for (int r = 0; r < 2; ++r) {
        double d = std::max(1e-6, std::abs(J_num_depth(r, 0)));
        EXPECT_NEAR(J_ana_depth(r, 0), J_num_depth(r, 0), kJacRelTol * d)
            << "Perturb depth [" << r << "]";
        d = std::max(1e-6, std::abs(J_num_td(r, 0)));
        EXPECT_NEAR(J_ana_td(r, 0), J_num_td(r, 0), kTdRelTol * d) << "Perturb td [" << r << "]";
    }
}

}  // namespace
}  // namespace tassel_core
