#include <Eigen/Householder>

#include <cmath>
#include <limits>
#include <stdexcept>
#include "factor/marginalization_prior_factor.h"
#include "marg/marg_helper.h"
#include "tassel_utils/macros.h"

#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {

namespace {

std::vector<int> marginalizationColumnOrder(int window_state_count, RetainedHostAction action) {
    if (window_state_count < 2) {
        throw std::invalid_argument("Marginalization requires at least two window states");
    }
    constexpr int pose_size = MargHelper::kPoseSize;
    constexpr int speed_bias_size = MargHelper::kSpeedBiasSize;
    constexpr int state_size = MargHelper::kFullStateSize;
    const int total_columns = window_state_count * state_size + 1;
    std::vector<int> order;
    order.reserve(static_cast<size_t>(total_columns));
    const auto append = [&order](int first, int count) {
        for (int i = 0; i < count; ++i) {
            order.push_back(first + i);
        }
    };

    switch (action) {
        case RetainedHostAction::InitializeRetainedSlot:
        case RetainedHostAction::ReplaceRetainedSlot:
            append(0, state_size);
            append(state_size + pose_size, speed_bias_size);
            append(state_size, pose_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
        case RetainedHostAction::MarginalizeOldestFrame:
            append(state_size, state_size);
            append(0, state_size);
            append(2 * state_size, total_columns - 2 * state_size);
            break;
    }
    if (static_cast<int>(order.size()) != total_columns) {
        throw std::logic_error("Marginalization column mapping is incomplete");
    }
    return order;
}

}  // namespace

Eigen::VectorXd MargHelper::evaluatePriorResidual(
    const MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double time_delay) {
    const int num_kept = static_cast<int>(prior.linearization_poses.size());
    TASSEL_ASSERT(static_cast<int>(poses.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(speed_bias.size()) == num_kept);
    TASSEL_ASSERT(static_cast<int>(prior.linearization_speed_bias.size()) == num_kept);
    prior.validate();

    MarginalizationPriorFactor factor(prior);
    std::vector<const double*> parameters;
    parameters.reserve(num_kept * 2 + 1);
    for (int i = 0; i < num_kept; ++i) {
        parameters.push_back(poses[i].data());
        if (i > 0) {
            parameters.push_back(speed_bias[i].data());
        }
    }
    parameters.push_back(&time_delay);

    Eigen::VectorXd residual(factor.num_residuals());
    TASSEL_ASSERT(factor.Evaluate(parameters.data(), residual.data(), nullptr));
    return residual;
}

void MargHelper::recenterPrior(
    MargLinData& prior, const std::vector<std::array<double, 6>>& poses,
    const std::vector<std::array<double, 9>>& speed_bias, double time_delay) {
    const int n = static_cast<int>(prior.linearization_poses.size());
    if (static_cast<int>(poses.size()) != n || static_cast<int>(speed_bias.size()) != n ||
        static_cast<int>(prior.linearization_speed_bias.size()) != n) {
        throw std::logic_error("Prior recenter state count does not match its linearization data");
    }
    prior.validate();
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(prior.H.cols());
    std::vector<Eigen::Matrix3d> rotation_maps(n, Eigen::Matrix3d::Identity());
    for (int i = 0; i < n; ++i) {
        const int col = prior.poseColumn(i);
        const Eigen::Vector3d old_position(
            prior.linearization_poses[i][0], prior.linearization_poses[i][1],
            prior.linearization_poses[i][2]);
        const Eigen::Vector3d new_position(poses[i][0], poses[i][1], poses[i][2]);
        const Eigen::Vector3d old_phi(
            prior.linearization_poses[i][3], prior.linearization_poses[i][4],
            prior.linearization_poses[i][5]);
        const Eigen::Vector3d new_phi(poses[i][3], poses[i][4], poses[i][5]);
        const Eigen::Vector3d rotation_delta =
            rightTangentDelta(
                (Eigen::Matrix<double, 6, 1>() << old_position, old_phi).finished(),
                (Eigen::Matrix<double, 6, 1>() << new_position, new_phi).finished())
                .tail<3>();
        if (!old_position.allFinite() || !new_position.allFinite()) {
            throw std::logic_error("Prior recenter encountered an invalid state");
        }
        delta.segment<3>(col) = new_position - old_position;
        delta.segment<3>(col + 3) = rotation_delta;
        rotation_maps[i] = rightTangentTransport(old_phi, new_phi);

        if (i > 0) {
            const int sb_col = prior.speedBiasColumn(i);
            for (int d = 0; d < kSpeedBiasSize; ++d) {
                delta(sb_col + d) = speed_bias[i][d] - prior.linearization_speed_bias[i][d];
            }
        }
    }
    delta(prior.delayColumn()) = time_delay - prior.linearization_delay_time;
    prior.b += prior.H * delta;
    for (int i = 0; i < n; ++i) {
        const int col = prior.poseColumn(i);
        prior.H.middleCols(col + 3, 3) *= rotation_maps[i];
    }
    prior.linearization_poses = poses;
    prior.linearization_speed_bias = speed_bias;
    prior.linearization_delay_time = time_delay;
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
    prior.validate();
    const Eigen::Matrix3d inverse_rotation = rotation.transpose();
    for (int i = 0; i < prior.stateCount(); ++i) {
        prior.H.middleCols(prior.poseColumn(i), 3) *= inverse_rotation;
        if (i > 0) {
            prior.H.middleCols(prior.speedBiasColumn(i), 3) *= inverse_rotation;
        }

        const Eigen::Vector3d position(
            prior.linearization_poses[i][0], prior.linearization_poses[i][1],
            prior.linearization_poses[i][2]);
        const Eigen::Vector3d phi(
            prior.linearization_poses[i][3], prior.linearization_poses[i][4],
            prior.linearization_poses[i][5]);
        const Eigen::Vector3d velocity(
            prior.linearization_speed_bias[i][0], prior.linearization_speed_bias[i][1],
            prior.linearization_speed_bias[i][2]);
        const Eigen::Vector3d transformed_position = rotation * position + translation;
        const Eigen::Vector3d transformed_phi =
            (Sophus::SO3d(rotation) * Sophus::SO3d::exp(phi)).log();
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
    // Householder beta 与输入同尺度，秩阈值必须随整个平方根系统缩放。
    const double system_scale = jacobian.cwiseAbs().maxCoeff();
    const double rank_threshold = std::sqrt(std::numeric_limits<double>::epsilon()) * system_scale;
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
