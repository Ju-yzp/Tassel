#ifndef TASSEL_LOOP_GEOMETRIC_VERIFIER_H_
#define TASSEL_LOOP_GEOMETRIC_VERIFIER_H_

#include <Eigen/Core>

#include <vector>

namespace tassel_loop {

struct GeometricVerification {
    bool accepted = false;
    Eigen::Matrix3d candidate_R_current = Eigen::Matrix3d::Identity();
    Eigen::Vector3d candidate_t_current = Eigen::Vector3d::UnitX();
    int match_count = 0;
    int inlier_count = 0;
};

class GeometricVerifier {
public:
    GeometricVerifier(
        int min_inliers = 30, double min_inlier_ratio = 0.3, double ransac_threshold = 0.002);

    GeometricVerification verify(
        const std::vector<Eigen::Vector2d>& candidate_points,
        const std::vector<Eigen::Vector2d>& current_points) const;

private:
    int min_inliers_;
    double min_inlier_ratio_;
    double ransac_threshold_;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_GEOMETRIC_VERIFIER_H_
