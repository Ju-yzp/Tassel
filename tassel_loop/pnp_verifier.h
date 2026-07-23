#ifndef TASSEL_LOOP_PNP_VERIFIER_H_
#define TASSEL_LOOP_PNP_VERIFIER_H_

#include <Eigen/Core>

#include <vector>

namespace tassel_loop {

struct PnpVerification {
    bool accepted = false;
    Eigen::Matrix3d host_R_current = Eigen::Matrix3d::Identity();
    Eigen::Vector3d host_t_current = Eigen::Vector3d::Zero();
    int match_count = 0;
    int inlier_count = 0;
    double inlier_ratio = 0.0;
    double mean_reprojection_error = 0.0;
    double translation_variance = 0.0;
    double rotation_variance = 0.0;
};

class PnpVerifier {
public:
    PnpVerifier(
        int min_inliers = 20, double min_inlier_ratio = 0.35, double inlier_threshold = 0.006,
        int max_iterations = 1000, double confidence = 0.999, int variance_quantile_divisor = 4,
        double max_translation_variance = 0.0);

    PnpVerification verify(
        const std::vector<Eigen::Vector3d>& host_points,
        const std::vector<Eigen::Vector2d>& current_points) const;

private:
    int min_inliers_;
    double min_inlier_ratio_;
    double inlier_threshold_;
    int max_iterations_;
    double confidence_;
    int variance_quantile_divisor_;
    double max_translation_variance_;
};

}  // namespace tassel_loop

#endif  // TASSEL_LOOP_PNP_VERIFIER_H_
