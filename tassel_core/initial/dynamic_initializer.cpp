#include "dynamic_initializer.h"
#include "tassel_utils/macros.h"

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

namespace tassel_core {
Eigen::Vector3d DynamicInitializer::solveGyroBias(
    const std::vector<Eigen::Matrix3d>& delta_q_dbg, const std::vector<Eigen::Matrix3d>& delta_qs,
    const std::vector<Eigen::Matrix3d>& R_Cs_in_C0) const {
    TASSEL_ASSERT(
        delta_q_dbg.size() == delta_qs.size() && delta_qs.size() + 1 == R_Cs_in_C0.size());

    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    for (int i = 0; i < delta_q_dbg.size(); ++i) {
        Eigen::Matrix3d dq =
            delta_qs[i].transpose() * R_Cs_in_C0[i].transpose() * R_Cs_in_C0[i + 1];
        Eigen::Quaterniond q(dq);
        q.normalize();
        Eigen::Vector3d phi = Sophus::SO3d(q.toRotationMatrix()).log();
        A += delta_q_dbg[i];
        b += phi;
    }

    Eigen::Vector3d x = A.ldlt().solve(b);
    spdlog::info("GyroBias: {:.5f} {:.5f} {:.5f}", x.x(), x.y(), x.z());
    return x;
}

void DynamicInitializer::solveRealtiveRotation(
    std::vector<std::pair<bool, Eigen::Matrix3d>>& all_Rij_with_state) const {}

void DynamicInitializer::buildLinearSystem(
    std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps,
    std::vector<Eigen::Matrix3d>& s, std::vector<double> all_ts, Eigen::Matrix3d ric,
    Eigen::Vector3d tic) {
    auto features = feature_manager_->collectLandmarks();

    int valid_num = 0;
    for (int i = 1; i < Rs.size(); ++i) {
        Eigen::Vector3d phi = Sophus::SO3d(Rs[i]).log();
        if (phi.norm() > valid_num) {
            ++valid_num;
        }
    }
    if (valid_num == 0) {
        spdlog::warn("DynamicInitializer: gravity direction is unobservable");
    }

    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    int feature_num = features.size();
    int observation_num = 0;
    for (const auto& feature : features) {
        observation_num += feature->observations.size();
    }

    //[B0_P0,Bo_P1,.....Bo_PN,V0,ba,g]
    int system_size = feature_num * 2 + 9;
    A = Eigen::MatrixXd::Zero(observation_num * 2, system_size);
    b = Eigen::VectorXd::Zero(feature_num * 2);
    int cur_feature_idx = 0;
    int cur_observation_idx = 0;
    for (const auto& feature : features) {
        int start_frame_id = feature->start_frame_id;
        for (int idx = 0; idx < feature->observations.size(); ++idx) {
            int cur_frame_id = start_frame_id + idx;
            Eigen::Matrix<double, 2, 3> Y;
            Eigen::Vector3d uv = feature->observations[idx].uv;
            Y << 1, 0, -uv(0), 1, 0, -uv(1);
            Eigen::Matrix<double, 2, 3> J = Y * ric.transpose();
            double dt = all_ts[cur_frame_id] - all_ts[start_frame_id];
            A.block<2, 3>(observation_num * 2, cur_feature_idx * 3) =
                J * Rs[cur_frame_id].transpose();
            A.block<2, 3>(observation_num * 2, feature_num * 3) =
                -J * Rs[cur_frame_id].transpose() * dt;
            A.block<2, 3>(observation_num * 2, feature_num * 3 + 3) =
                J * Rs[cur_frame_id].transpose() * s[cur_frame_id];
            A.block<2, 3>(observation_num * 2, feature_num * 3 + 6) =
                J * Rs[cur_frame_id].transpose() * 0.5 * dt * dt;
            b.block(cur_feature_idx * 2, 0, 1, 2) =
                J * (Rs[cur_frame_id].transpose() * Ps[cur_frame_id] + tic);
            ++cur_observation_idx;
        }
        ++cur_feature_idx;
    }

    Eigen::MatrixXd A1 = A.block(0, 0, A.rows(), A.cols() - 3);
    Eigen::MatrixXd A1TA1_inv =
        (A1.transpose() * A1).llt().solve(Eigen::MatrixXd::Identity(A1.cols(), A1.cols()));
    Eigen::MatrixXd A2 = A.block(0, A.cols() - 3, A.rows(), 3);
    Eigen::MatrixXd Temp = A2.transpose() * (Eigen::MatrixXd::Identity(A1.rows(), A1.rows()) -
                                             A1 * A1TA1_inv * A1.transpose());
    Eigen::MatrixXd D = Temp * A2;
    Eigen::MatrixXd d = Temp * b;
}

Eigen::Vector<double, 7> DynamicInitializer::compute_coeff(
    Eigen::Matrix3d D, Eigen::Vector3d d, double g) {
    double D11 = D(0, 0);
    double D12 = D(0, 1);
    double D13 = D(0, 2);
    double D21 = D(1, 0);
    double D22 = D(1, 1);
    double D23 = D(1, 2);
    double D31 = D(2, 0);
    double D32 = D(2, 1);
    double D33 = D(2, 2);
    double d1 = d(0);
    double d2 = d(1);
    double d3 = d(2);

    Eigen::Vector<double, 7> coeff;
    const double g2 = (g * g);
    const double d1_2 = (d1 * d1);
    const double d2_2 = (d2 * d2);
    const double d3_2 = (d3 * d3);
    const double d1d2 = d1 * d2;
    const double d1d3 = d1 * d3;
    const double d2d3 = d2 * d3;
    const double D11_2 = (D11 * D11);
    const double D12_2 = (D12 * D12);
    const double D13_2 = (D13 * D13);
    const double D21_2 = (D21 * D21);
    const double D22_2 = (D22 * D22);
    const double D23_2 = (D23 * D23);
    const double D31_2 = (D31 * D31);
    const double D32_2 = (D32 * D32);
    const double D33_2 = (D33 * D33);
    const double D11_D12 = D11 * D12;
    const double D11_D13 = D11 * D13;
    const double D11_D21 = D11 * D21;
    const double D11_D22 = D11 * D22;
    const double D11_D23 = D11 * D23;
    const double D11_D31 = D11 * D31;
    const double D11_D32 = D11 * D32;
    const double D11_D33 = D11 * D33;
    const double D12_D13 = D12 * D13;
    const double D12_D21 = D12 * D21;
    const double D12_D22 = D12 * D22;
    const double D12_D23 = D12 * D23;
    const double D12_D31 = D12 * D31;
    const double D12_D32 = D12 * D32;
    const double D12_D33 = D12 * D33;
    const double D13_D21 = D13 * D21;
    const double D13_D22 = D13 * D22;
    const double D13_D23 = D13 * D23;
    const double D13_D31 = D13 * D31;
    const double D13_D32 = D13 * D32;
    const double D13_D33 = D13 * D33;
    const double D21_D22 = D21 * D22;
    const double D21_D23 = D21 * D23;
    const double D21_D31 = D21 * D31;
    const double D21_D32 = D21 * D32;
    const double D21_D33 = D21 * D33;
    const double D22_D23 = D22 * D23;
    const double D22_D31 = D22 * D31;
    const double D22_D32 = D22 * D32;
    const double D22_D33 = D22 * D33;
    const double D23_D31 = D23 * D31;
    const double D23_D32 = D23 * D32;
    const double D23_D33 = D23 * D33;
    const double D31_D32 = D31 * D32;
    const double D31_D33 = D31 * D33;
    const double D32_D33 = D32 * D33;
    coeff(0) = 1;
    coeff(1) = -2 * D11 - 2 * D22 - 2 * D33;
    coeff(2) = D11_2 + 4 * D11_D22 + 4 * D11_D33 - 2 * D12_D21 - 2 * D13_D31 + D22_2 + 4 * D22_D33 -
               2 * D23_D32 + D33_2 - d1_2 / g2 - d2_2 / g2 - d3_2 / g2;
    coeff(3) = -2 * D11 * D22_2 - 2 * D11 * D33_2 + 2 * D11 * d2_2 / g2 + 2 * D11 * d3_2 / g2 -
               2 * D11_2 * D22 - 2 * D11_2 * D33 + 2 * D11_D12 * D21 + 2 * D11_D13 * D31 -
               8 * D11_D22 * D33 + 4 * D11_D23 * D32 - 2 * D12 * d1d2 / g2 + 2 * D12_D21 * D22 +
               4 * D12_D21 * D33 - 2 * D12_D23 * D31 - 2 * D13 * d1d3 / g2 - 2 * D13_D21 * D32 +
               4 * D13_D22 * D31 + 2 * D13_D31 * D33 - 2 * D21 * d1d2 / g2 - 2 * D22 * D33_2 +
               2 * D22 * d1_2 / g2 + 2 * D22 * d3_2 / g2 - 2 * D22_2 * D33 + 2 * D22_D23 * D32 -
               2 * D23 * d2d3 / g2 + 2 * D23_D32 * D33 - 2 * D31 * d1d3 / g2 - 2 * D32 * d2d3 / g2 +
               2 * D33 * d1_2 / g2 + 2 * D33 * d2_2 / g2;
    coeff(4) =
        D11_2 * D22_2 + 4 * D11_2 * D22_D33 - 2 * D11_2 * D23_D32 + D11_2 * D33_2 -
        D11_2 * d2_2 / g2 - D11_2 * d3_2 / g2 - 2 * D11_D12 * D21_D22 - 4 * D11_D12 * D21_D33 +
        2 * D11_D12 * D23_D31 + D11_D12 * d1d2 / g2 + 2 * D11_D13 * D21_D32 -
        4 * D11_D13 * D22_D31 - 2 * D11_D13 * D31_D33 + D11_D13 * d1d3 / g2 + D11_D21 * d1d2 / g2 -
        4 * D11_D22 * D23_D32 + 4 * D11_D22 * D33_2 - 4 * D11_D22 * d3_2 / g2 +
        4 * D11_D23 * d2d3 / g2 + D11_D31 * d1d3 / g2 + 4 * D11_D32 * d2d3 / g2 +
        4 * D11_D33 * D22_2 - 4 * D11_D33 * d2_2 / g2 + D12_2 * D21_2 + 2 * D12_D13 * D21_D31 -
        4 * D12_D21 * D22_D33 + 2 * D12_D21 * D23_D32 - 2 * D12_D21 * D33_2 - D12_D21 * d1_2 / g2 -
        D12_D21 * d2_2 / g2 + 2 * D12_D21 * d3_2 / g2 + 2 * D12_D22 * D23_D31 +
        D12_D22 * d1d2 / g2 + 2 * D12_D23 * D31_D33 - 3 * D12_D23 * d1d3 / g2 -
        3 * D12_D31 * d2d3 / g2 + 4 * D12_D33 * d1d2 / g2 + D13_2 * D31_2 + 2 * D13_D21 * D22_D32 -
        3 * D13_D21 * d2d3 / g2 - 4 * D13_D22 * D31_D33 + 4 * D13_D22 * d1d3 / g2 +
        2 * D13_D23 * D31_D32 - 2 * D13_D31 * D22_2 - D13_D31 * d1_2 / g2 +
        2 * D13_D31 * d2_2 / g2 - D13_D31 * d3_2 / g2 - 3 * D13_D32 * d1d2 / g2 +
        D13_D33 * d1d3 / g2 + D21_D22 * d1d2 / g2 - 3 * D21_D32 * d1d3 / g2 +
        4 * D21_D33 * d1d2 / g2 + D22_2 * D33_2 - D22_2 * d1_2 / g2 - D22_2 * d3_2 / g2 +
        D22_D23 * d2d3 / g2 + 4 * D22_D31 * d1d3 / g2 + D22_D32 * d2d3 / g2 -
        4 * D22_D33 * d1_2 / g2 + D23_2 * D32_2 - 3 * D23_D31 * d1d2 / g2 +
        2 * D23_D32 * d1_2 / g2 - D23_D32 * d2_2 / g2 - D23_D32 * d3_2 / g2 + D23_D33 * d2d3 / g2 +
        D31_D33 * d1d3 / g2 + D32_D33 * (-4 * D11_D23 + 2 * D13_D21 - 2 * D22_D23 + d2d3 / g2) -
        D33_2 * d1_2 / g2 - D33_2 * d2_2 / g2;
    coeff(5) =
        -2 * D11 * D22_2 * D33_2 + 2 * D11 * D22_2 * d3_2 / g2 - 2 * D11 * D23_2 * D32_2 +
        2 * D11 * D33_2 * d2_2 / g2 - 2 * D11_2 * D22 * D33_2 + 2 * D11_2 * D22 * d3_2 / g2 -
        2 * D11_2 * D22_2 * D33 + 2 * D11_2 * D22_D23 * D32 - 2 * D11_2 * D23 * d2d3 / g2 +
        2 * D11_2 * D23_D32 * D33 - 2 * D11_2 * D32 * d2d3 / g2 + 2 * D11_2 * D33 * d2_2 / g2 +
        2 * D11_D12 * D21 * D33_2 - 2 * D11_D12 * D21 * d3_2 / g2 + 4 * D11_D12 * D21_D22 * D33 -
        2 * D11_D12 * D21_D23 * D32 - 2 * D11_D12 * D22_D23 * D31 + 2 * D11_D12 * D23 * d1d3 / g2 -
        2 * D11_D12 * D23_D31 * D33 + 2 * D11_D12 * D31 * d2d3 / g2 -
        2 * D11_D12 * D33 * d1d2 / g2 + 2 * D11_D13 * D21 * d2d3 / g2 -
        2 * D11_D13 * D21_D22 * D32 - 2 * D11_D13 * D21_D32 * D33 - 2 * D11_D13 * D22 * d1d3 / g2 +
        2 * D11_D13 * D22_2 * D31 + 4 * D11_D13 * D22_D31 * D33 - 2 * D11_D13 * D23_D31 * D32 -
        2 * D11_D13 * D31 * d2_2 / g2 + 2 * D11_D13 * D32 * d1d2 / g2 +
        2 * D11_D21 * D32 * d1d3 / g2 - 2 * D11_D21 * D33 * d1d2 / g2 -
        2 * D11_D22 * D23 * d2d3 / g2 + 4 * D11_D22 * D23_D32 * D33 -
        2 * D11_D22 * D31 * d1d3 / g2 - 2 * D11_D22 * D32 * d2d3 / g2 +
        2 * D11_D23 * D31 * d1d2 / g2 + 2 * D11_D23 * D32 * d2_2 / g2 +
        2 * D11_D23 * D32 * d3_2 / g2 - 2 * D11_D23 * D33 * d2d3 / g2 -
        2 * D11_D32 * D33 * d2d3 / g2 - 2 * D12 * D33_2 * d1d2 / g2 - 2 * D12_2 * D21_2 * D33 +
        2 * D12_2 * D21_D23 * D31 + 2 * D12_D13 * D21_2 * D32 - 2 * D12_D13 * D21_D22 * D31 -
        2 * D12_D13 * D21_D31 * D33 + 2 * D12_D13 * D23 * D31_2 + 2 * D12_D21 * D22 * D33_2 -
        2 * D12_D21 * D22 * d3_2 / g2 - 2 * D12_D21 * D23_D32 * D33 +
        2 * D12_D21 * D33 * d1_2 / g2 + 2 * D12_D21 * D33 * d2_2 / g2 +
        2 * D12_D22 * D23 * d1d3 / g2 - 2 * D12_D22 * D23_D31 * D33 +
        2 * D12_D22 * D31 * d2d3 / g2 - 2 * D12_D22 * D33 * d1d2 / g2 -
        2 * D12_D23 * D31 * d1_2 / g2 - 2 * D12_D23 * D31 * d2_2 / g2 -
        2 * D12_D23 * D31 * d3_2 / g2 + 2 * D12_D23 * D33 * d1d3 / g2 + 2 * D12_D31 * D23_2 * D32 +
        2 * D12_D31 * D33 * d2d3 / g2 - 2 * D13 * D22_2 * d1d3 / g2 + 2 * D13_2 * D21_D31 * D32 -
        2 * D13_2 * D22 * D31_2 + 2 * D13_D21 * D22 * d2d3 / g2 - 2 * D13_D21 * D22_D32 * D33 +
        2 * D13_D21 * D23 * D32_2 - 2 * D13_D21 * D32 * d1_2 / g2 - 2 * D13_D21 * D32 * d2_2 / g2 -
        2 * D13_D21 * D32 * d3_2 / g2 + 2 * D13_D21 * D33 * d2d3 / g2 -
        2 * D13_D22 * D23_D31 * D32 + 2 * D13_D22 * D31 * d1_2 / g2 +
        2 * D13_D22 * D31 * d3_2 / g2 + 2 * D13_D22 * D32 * d1d2 / g2 -
        2 * D13_D22 * D33 * d1d3 / g2 + 2 * D13_D31 * D22_2 * D33 - 2 * D13_D31 * D33 * d2_2 / g2 +
        2 * D13_D32 * D33 * d1d2 / g2 - 2 * D21 * D33_2 * d1d2 / g2 +
        2 * D21_D22 * D32 * d1d3 / g2 - 2 * D21_D22 * D33 * d1d2 / g2 +
        2 * D21_D32 * D33 * d1d3 / g2 + 2 * D22 * D33_2 * d1_2 / g2 - 2 * D22_2 * D31 * d1d3 / g2 +
        2 * D22_2 * D33 * d1_2 / g2 + 2 * D22_D23 * D31 * d1d2 / g2 -
        2 * D22_D23 * D32 * d1_2 / g2 - 2 * D22_D31 * D33 * d1d3 / g2 +
        2 * D23_D31 * D33 * d1d2 / g2 - 2 * D23_D32 * D33 * d1_2 / g2;
    coeff(6) =
        D11_2 * D22_2 * D33_2 - D11_2 * D22_2 * d3_2 / g2 + D11_2 * D22_D23 * d2d3 / g2 +
        D11_2 * D22_D32 * d2d3 / g2 + D11_2 * D23_2 * D32_2 - D11_2 * D23_D32 * d2_2 / g2 -
        D11_2 * D23_D32 * d3_2 / g2 + D11_2 * D23_D33 * d2d3 / g2 - D11_2 * D33_2 * d2_2 / g2 -
        2 * D11_D12 * D21_D22 * D33_2 + 2 * D11_D12 * D21_D22 * d3_2 / g2 -
        D11_D12 * D21_D23 * d2d3 / g2 - D11_D12 * D21_D32 * d2d3 / g2 +
        2 * D11_D12 * D22_D23 * D31_D33 - D11_D12 * D22_D23 * d1d3 / g2 -
        D11_D12 * D22_D31 * d2d3 / g2 - 2 * D11_D12 * D23_2 * D31_D32 +
        D11_D12 * D23_D31 * d2_2 / g2 + D11_D12 * D23_D31 * d3_2 / g2 +
        D11_D12 * D23_D32 * d1d2 / g2 - D11_D12 * D23_D33 * d1d3 / g2 -
        D11_D12 * D31_D33 * d2d3 / g2 + D11_D12 * D33_2 * d1d2 / g2 -
        D11_D13 * D21_D22 * d2d3 / g2 - 2 * D11_D13 * D21_D23 * D32_2 +
        D11_D13 * D21_D32 * d2_2 / g2 + D11_D13 * D21_D32 * d3_2 / g2 -
        D11_D13 * D21_D33 * d2d3 / g2 - 2 * D11_D13 * D22_2 * D31_D33 +
        D11_D13 * D22_2 * d1d3 / g2 + 2 * D11_D13 * D22_D23 * D31_D32 -
        D11_D13 * D22_D32 * d1d2 / g2 - D11_D13 * D23_D31 * d2d3 / g2 +
        D11_D13 * D23_D32 * d1d3 / g2 - D11_D13 * D31_D32 * d2d3 / g2 +
        2 * D11_D13 * D31_D33 * d2_2 / g2 - D11_D21 * D22_D32 * d1d3 / g2 +
        D11_D21 * D23_D32 * d1d2 / g2 + D11_D21 * D33_2 * d1d2 / g2 -
        D11_D22 * D23_D31 * d1d2 / g2 + D11_D23 * D31_D32 * d1d3 / g2 -
        D11_D23 * D31_D33 * d1d2 / g2 + D11_D31 * D22_2 * d1d3 / g2 + D12_2 * D21_2 * D33_2 -
        D12_2 * D21_2 * d3_2 / g2 - 2 * D12_2 * D21_D23 * D31_D33 + D12_2 * D21_D23 * d1d3 / g2 +
        D12_2 * D21_D31 * d2d3 / g2 + D12_2 * D23_2 * D31_2 - D12_2 * D23_D31 * d1d2 / g2 +
        D12_D13 * D21_2 * d2d3 / g2 + 2 * D12_D13 * D21_D22 * D31_D33 -
        D12_D13 * D21_D22 * d1d3 / g2 + 2 * D12_D13 * D21_D23 * D31_D32 -
        D12_D13 * D21_D31 * d2_2 / g2 - D12_D13 * D21_D31 * d3_2 / g2 +
        D12_D13 * D21_D33 * d1d3 / g2 - 2 * D12_D13 * D22_D23 * D31_2 +
        D12_D13 * D22_D31 * d1d2 / g2 + D12_D13 * D31_2 * d2d3 / g2 -
        D12_D13 * D31_D33 * d1d2 / g2 - D12_D21 * D22_D31 * d1d3 / g2 -
        D12_D21 * D23_D32 * d1_2 / g2 - D12_D21 * D23_D32 * d3_2 / g2 +
        D12_D21 * D23_D33 * d2d3 / g2 + D12_D21 * D31_D33 * d1d3 / g2 -
        D12_D21 * D33_2 * d1_2 / g2 - D12_D21 * D33_2 * d2_2 / g2 + D12_D22 * D23_D31 * d1_2 / g2 +
        D12_D22 * D23_D31 * d3_2 / g2 - D12_D22 * D23_D33 * d1d3 / g2 -
        D12_D22 * D31_D33 * d2d3 / g2 + D12_D22 * D33_2 * d1d2 / g2 - D12_D23 * D31_2 * d1d3 / g2 +
        D12_D23 * D31_D33 * d1_2 / g2 + D12_D23 * D31_D33 * d2_2 / g2 -
        D12_D31 * D23_2 * d2d3 / g2 + D12_D32 * D21_2 * d1d3 / g2 + D12_D32 * D23_2 * d1d3 / g2 +
        D13_2 * D21_2 * D32_2 - 2 * D13_2 * D21_D22 * D31_D32 + D13_2 * D21_D31 * d2d3 / g2 -
        D13_2 * D21_D32 * d1d3 / g2 + D13_2 * D22_2 * D31_2 - D13_2 * D31_2 * d2_2 / g2 +
        D13_2 * D31_D32 * d1d2 / g2 + D13_D21 * D22_D31 * d1d2 / g2 +
        D13_D21 * D22_D32 * d1_2 / g2 + D13_D21 * D22_D32 * d3_2 / g2 -
        D13_D21 * D22_D33 * d2d3 / g2 - D13_D21 * D31_D33 * d1d2 / g2 -
        D13_D21 * D32_2 * d2d3 / g2 + D13_D22 * D23_D31 * d2d3 / g2 -
        D13_D22 * D23_D32 * d1d3 / g2 + D13_D22 * D31_D32 * d2d3 / g2 +
        D13_D23 * D31_2 * d1d2 / g2 - D13_D23 * D31_D32 * d1_2 / g2 -
        D13_D23 * D31_D32 * d2_2 / g2 + D13_D23 * D32_2 * d1d2 / g2 - D13_D31 * D22_2 * d1_2 / g2 -
        D13_D31 * D22_2 * d3_2 / g2 - D13_D32 * D21_2 * d1d2 / g2 + D13_D33 * D22_2 * d1d3 / g2 +
        D21_D22 * D33_2 * d1d2 / g2 + D21_D23 * D32_2 * d1d3 / g2 + D22_2 * D31_D33 * d1d3 / g2 -
        D22_2 * D33_2 * d1_2 / g2 - D22_D23 * D31_D32 * d1d3 / g2 - D22_D23 * D31_D33 * d1d2 / g2 +
        D23_2 * D31_D32 * d1d2 / g2 - D23_2 * D32_2 * d1_2 / g2 +
        D32_D33 * (-2 * D11_2 * D22_D23 + D11_2 * d2d3 / g2 + 2 * D11_D12 * D21_D23 +
                   2 * D11_D13 * D21_D22 - D11_D13 * d1d2 / g2 - D11_D21 * d1d3 / g2 -
                   2 * D12_D13 * D21_2 + D12_D21 * d2d3 / g2 - D12_D23 * d1d2 / g2 +
                   D13_D21 * d1_2 / g2 + D13_D21 * d2_2 / g2 - D13_D22 * d1d2 / g2 -
                   D21_D22 * d1d3 / g2 - D21_D23 * d1d2 / g2 + 2 * D22_D23 * d1_2 / g2);
    return coeff;
}
}  // namespace tassel_core
