#ifndef TASSEL_UTILS_TRIANGULATION_H_
#define TASSEL_UTILS_TRIANGULATION_H_

#include <Eigen/Core>
#include <Eigen/SVD>
#include <vector>

namespace tassel_utils {

// 两视图 DLT 线性三角化, 返回齐次坐标 (X, Y, Z, W)
inline Eigen::Vector4d triangulateTwoView(
    const Eigen::Matrix<double, 3, 4>& P0, const Eigen::Vector2d& uv0,
    const Eigen::Matrix<double, 3, 4>& P1, const Eigen::Vector2d& uv1, double* cond = nullptr) {
    Eigen::Matrix4d A;
    A.row(0) = uv0.x() * P0.row(2) - P0.row(0);
    A.row(1) = uv0.y() * P0.row(2) - P0.row(1);
    A.row(2) = uv1.x() * P1.row(2) - P1.row(0);
    A.row(3) = uv1.y() * P1.row(2) - P1.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    if (cond) {
        Eigen::Vector4d s = svd.singularValues();
        *cond = s(0) / s(3);
    }
    return svd.matrixV().col(3);
}

// 多视图 DLT 线性三角化
inline Eigen::Vector4d triangulateMultiView(
    const std::vector<Eigen::Matrix<double, 3, 4>>& poses, const std::vector<Eigen::Vector2d>& uvs,
    double* cond = nullptr) {
    int n = static_cast<int>(uvs.size());
    Eigen::MatrixXd A(2 * n, 4);
    for (int i = 0; i < n; ++i) {
        A.row(2 * i) = uvs[i].x() * poses[i].row(2) - poses[i].row(0);
        A.row(2 * i + 1) = uvs[i].y() * poses[i].row(2) - poses[i].row(1);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    if (cond) {
        Eigen::VectorXd s = svd.singularValues();
        *cond = s(0) / s(3);
    }
    return svd.matrixV().col(3);
}

// 齐次坐标转 3D 点
inline Eigen::Vector3d dehomogenize(const Eigen::Vector4d& h) { return h.head<3>() / h(3); }

}  // namespace tassel_utils

#endif  // TASSEL_UTILS_TRIANGULATION_H_
