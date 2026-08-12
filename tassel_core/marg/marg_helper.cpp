#include <Eigen/Householder>

#include <cmath>
#include <limits>
#include <stdexcept>
#include "factor/marginalization_prior_factor.h"
#include "marg/marg_helper.h"
#include "marg/state_layout.h"
#include "tassel_utils/macros.h"

#include <sophus/so3.hpp>

namespace tassel_core {

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

Eigen::VectorXd MargHelper::evaluatePriorResidual(
    const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double delay_time) {
    const int num_kept = static_cast<int>(prior.linearization_poses.size());
    TASSEL_ASSERT(static_cast<int>(poses.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(speed_bias.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(prior.linearization_speed_bias.size()) == num_kept);
    const PriorStateLayout layout(num_kept, static_cast<int>(prior.H.cols()));
    TASSEL_ASSERT(layout.hasDelay());

    MarginalizationPriorFactor factor(prior);
    std::vector<const double*> parameters;
    parameters.reserve(num_kept * 2 + 1);
    for (int i = 0; i < num_kept; ++i) {
        parameters.push_back(poses[i].data());
        if (layout.hasSpeedBias(i)) {
            parameters.push_back(speed_bias[i].data());
        }
    }
    parameters.push_back(&delay_time);

    Eigen::VectorXd residual(factor.num_residuals());
    TASSEL_ASSERT(factor.Evaluate(parameters.data(), residual.data(), nullptr));
    return residual;
}

void MargHelper::recenterPrior(
    MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double delay_time) {
    const int n = static_cast<int>(prior.linearization_poses.size());
    if (static_cast<int>(poses.size()) != n || static_cast<int>(speed_bias.size()) != n ||
        static_cast<int>(prior.linearization_speed_bias.size()) != n) {
        throw std::logic_error("Prior recenter state count does not match its linearization data");
    }
    const PriorStateLayout layout(n, static_cast<int>(prior.H.cols()));
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(prior.H.cols());
    std::vector<Eigen::Matrix3d> rotation_maps(n, Eigen::Matrix3d::Identity());
    for (int i = 0; i < n; ++i) {
        const int col = layout.poseColumn(i);
        const Eigen::Vector3d old_position(
            prior.linearization_poses[i][0], prior.linearization_poses[i][1],
            prior.linearization_poses[i][2]);
        const Eigen::Vector3d new_position(poses[i][0], poses[i][1], poses[i][2]);
        const Eigen::Vector3d old_phi(
            prior.linearization_poses[i][3], prior.linearization_poses[i][4],
            prior.linearization_poses[i][5]);
        const Eigen::Vector3d new_phi(poses[i][3], poses[i][4], poses[i][5]);
        const Eigen::Vector3d rotation_delta =
            (Sophus::SO3d::exp(old_phi).inverse() * Sophus::SO3d::exp(new_phi)).log();
        if (!old_position.allFinite() || !new_position.allFinite() || !rotation_delta.allFinite() ||
            rotation_delta.norm() >= kPi - 1e-6) {
            throw std::logic_error("Prior recenter encountered an invalid SO(3) linearization");
        }
        delta.segment<3>(col) = new_position - old_position;
        delta.segment<3>(col + 3) = rotation_delta;
        rotation_maps[i] = Sophus::SO3d::leftJacobianInverse(-rotation_delta);

        if (layout.hasSpeedBias(i)) {
            const int sb_col = layout.speedBiasColumn(i);
            for (int d = 0; d < kSpeedBiasSize; ++d) {
                delta(sb_col + d) = speed_bias[i][d] - prior.linearization_speed_bias[i][d];
            }
        }
    }
    if (layout.hasDelay()) {
        delta(layout.delayColumn()) = delay_time - prior.linearization_delay_time;
    }
    prior.b += prior.H * delta;
    for (int i = 0; i < n; ++i) {
        const int col = layout.poseColumn(i);
        prior.H.middleCols(col + 3, 3) *= rotation_maps[i];
    }
    prior.linearization_poses = poses;
    prior.linearization_speed_bias = speed_bias;
    prior.linearization_delay_time = delay_time;
    if (!prior.H.allFinite() || !prior.b.allFinite()) {
        throw std::logic_error("Recentered marginalization prior is not finite");
    }
}

void MargHelper::transformPriorGauge(
    MargLinData& prior, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    if (!rotation.allFinite() || !translation.allFinite() ||
        !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
        std::abs(rotation.determinant() - 1.0) > 1e-8) {
        throw std::logic_error("Prior gauge transform is not a valid rigid transform");
    }
    const PriorStateLayout layout(
        static_cast<int>(prior.linearization_poses.size()), static_cast<int>(prior.H.cols()));
    const Eigen::Matrix3d inverse_rotation = rotation.transpose();
    const int n = static_cast<int>(prior.linearization_poses.size());
    for (int i = 0; i < n; ++i) {
        const int col = layout.poseColumn(i);
        prior.H.middleCols(col, 3) *= inverse_rotation;
        if (layout.hasSpeedBias(i)) {
            prior.H.middleCols(layout.speedBiasColumn(i), 3) *= inverse_rotation;
        }

        const Eigen::Vector3d position(
            prior.linearization_poses[i][0], prior.linearization_poses[i][1],
            prior.linearization_poses[i][2]);
        const Eigen::Vector3d phi(
            prior.linearization_poses[i][3], prior.linearization_poses[i][4],
            prior.linearization_poses[i][5]);
        const Eigen::Vector3d transformed_position = rotation * position + translation;
        const Eigen::Vector3d transformed_phi =
            (Sophus::SO3d(rotation) * Sophus::SO3d::exp(phi)).log();
        const Eigen::Vector3d velocity(
            prior.linearization_speed_bias[i][0], prior.linearization_speed_bias[i][1],
            prior.linearization_speed_bias[i][2]);
        const Eigen::Vector3d transformed_velocity = rotation * velocity;
        for (int d = 0; d < 3; ++d) {
            prior.linearization_poses[i][d] = transformed_position[d];
            prior.linearization_poses[i][3 + d] = transformed_phi[d];
            prior.linearization_speed_bias[i][d] = transformed_velocity[d];
        }
    }
    if (!prior.H.allFinite()) {
        throw std::logic_error("Gauge-transformed marginalization prior is not finite");
    }
}

void MargHelper::marginalizeSquareRootSystem(
    size_t marginalized_size, size_t retained_size, Eigen::MatrixXd& jacobian,
    Eigen::VectorXd& residual, Eigen::MatrixXd& prior_jacobian, Eigen::VectorXd& prior_residual) {
    TASSEL_ASSERT(Eigen::Index(marginalized_size + retained_size) == jacobian.cols());
    TASSEL_ASSERT(jacobian.rows() == residual.rows());

    if (jacobian.rows() == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        return;
    }

    // 丢弃缺秩行，避免将无效约束写入先验。
    Eigen::Index marginalized_rank = 0;
    Eigen::Index total_rank = 0;
    const Eigen::Index rows = jacobian.rows();
    const Eigen::Index cols = jacobian.cols();

    Eigen::VectorXd temp_vec(cols + 1);
    double* temp_data = temp_vec.data();
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon());
    for (Eigen::Index i = 0; i < cols && total_rank < rows; ++i) {
        Eigen::Index remainingRows = rows - total_rank;
        Eigen::Index remainingCols = cols - i - 1;

        double beta;
        double hCoeff;
        jacobian.col(i).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
        if (std::abs(beta) > rank_threshold) {
            jacobian.coeffRef(total_rank, i) = beta;

            jacobian.bottomRightCorner(remainingRows, remainingCols)
                .applyHouseholderOnTheLeft(
                    jacobian.col(i).tail(remainingRows - 1), hCoeff, temp_data + i + 1);
            residual.tail(remainingRows)
                .applyHouseholderOnTheLeft(
                    jacobian.col(i).tail(remainingRows - 1), hCoeff, temp_data + cols);
            total_rank++;
        } else {
            jacobian.coeffRef(total_rank, i) = 0;
        }

        jacobian.col(i).tail(remainingRows - 1).setZero();

        if (i == Eigen::Index(marginalized_size) - 1) {
            marginalized_rank = total_rank;
        }
    }

    const Eigen::Index retained_rank = total_rank - marginalized_rank;

    if (retained_rank == 0) {
        prior_jacobian.resize(0, static_cast<Eigen::Index>(retained_size));
        prior_residual.resize(0);
        jacobian.resize(0, 0);
        residual.resize(0);
        return;
    }

    prior_jacobian =
        jacobian.block(marginalized_rank, marginalized_size, retained_rank, retained_size);
    prior_residual = residual.segment(marginalized_rank, retained_rank);

    jacobian.resize(0, 0);
    residual.resize(0);
}

Eigen::MatrixXd MargHelper::reorderForMarginalization(
    const Eigen::MatrixXd& jacobian, RetainedHostAction action) {
    TASSEL_ASSERT(jacobian.cols() >= 2 * kFullStateSize + 1);
    TASSEL_ASSERT((jacobian.cols() - 1) % kFullStateSize == 0);
    const int window_state_count = (static_cast<int>(jacobian.cols()) - 1) / kFullStateSize;
    const std::vector<int> order = marginalizationColumnOrder(window_state_count, action);
    Eigen::MatrixXd reordered(jacobian.rows(), jacobian.cols());
    for (int destination = 0; destination < static_cast<int>(order.size()); ++destination) {
        reordered.col(destination) = jacobian.col(order[static_cast<size_t>(destination)]);
    }
    return reordered;
}

}  // namespace tassel_core
