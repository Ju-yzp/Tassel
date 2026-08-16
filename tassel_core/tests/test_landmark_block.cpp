#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>

#include "marg/landmark_block.h"

namespace tassel_core {
namespace {

TEST(LandmarkBlockTest, MarginalizeLandmarkMatchesHouseholderQr) {
    LandmarkBlock block(6, nullptr);
    block.allocate(3, 3, 6);
    auto& storage = block.get_mutable_storage();
    for (int row = 0; row < storage.rows(); ++row) {
        for (int col = 0; col < storage.cols(); ++col) {
            storage(row, col) = 0.25 * (row + 1) - 0.1 * (col + 2) + 0.01 * row * col;
        }
    }
    storage.col(block.get_landmark_index()) << 2.0, -1.0, 3.0, 4.0, -2.0, 1.0;
    const Eigen::MatrixXd original = storage;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(original.col(block.get_landmark_index()));
    const Eigen::MatrixXd expected = qr.householderQ().adjoint() * original;

    block.marginalizeLandmark();

    EXPECT_TRUE(storage.isApprox(expected, 1e-12));
    EXPECT_EQ(block.get_kept_rows(), storage.rows() - 1);
    EXPECT_LE(storage.col(block.get_landmark_index()).tail(storage.rows() - 1).norm(), 1e-12);
}

TEST(LandmarkBlockTest, ZeroLandmarkJacobianKeepsEveryConstraint) {
    LandmarkBlock block(6, nullptr);
    block.allocate(2, 1, 6);
    auto& storage = block.get_mutable_storage();
    storage.setRandom();
    storage.col(block.get_landmark_index()).setZero();
    const Eigen::MatrixXd original = storage;

    block.marginalizeLandmark();

    EXPECT_EQ(block.get_kept_rows(), block.get_num_rows());
    EXPECT_TRUE(storage.isApprox(original));
}

}  // namespace
}  // namespace tassel_core
