#ifndef TASSEL_CORE_INITIAL_INITIAL_SFM_H_
#define TASSEL_CORE_INITIAL_INITIAL_SFM_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <unordered_set>
#include <vector>

#include <opencv2/core/types.hpp>

namespace tassel_core {

class FeatureManager;
struct State;

struct SFMFeature {
    bool triangulated = false;
    int id = -1;
    std::vector<std::pair<int, Eigen::Vector2d>> observations;
    std::array<double, 3> position{};
};

class InitialSFM {
public:
    InitialSFM(
        int min_points = 10, int min_inliers = 8, double epipolar_threshold = 0.004,
        double pnp_threshold = 0.03, int ba_iterations = 30)
        : min_points_(min_points),
          min_inliers_(min_inliers),
          epipolar_threshold_(epipolar_threshold),
          pnp_threshold_(pnp_threshold),
          ba_iterations_(ba_iterations) {}

    bool construct(
        const State& state, const FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
        std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out,
        int first_frame_index = 0);

private:
    struct PoseCandidate {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        int score = 0;
        double prior_error = 0.0;
    };

    std::vector<std::pair<int, int>> scoreBaselineFrames(
        int first_frame_index, int host_id, const std::vector<Eigen::Quaterniond>& camera_rotations,
        const FeatureManager& feature_manager);

    bool computeEssential(
        int seed_id, int other_id, const std::vector<SFMFeature>& features,
        std::vector<PoseCandidate>& candidates, std::vector<cv::Point2f>& pts_seed,
        std::vector<cv::Point2f>& pts_other, std::unordered_set<int>& inlier_feature_ids);

    bool resolvePose(
        const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
        const std::vector<cv::Point2f>& pts_other, const Eigen::Matrix3d& rotation_prior,
        PoseCandidate& selected);

    bool reconstructScene(
        int frame_num, int seed_id, int other_id, const Eigen::Vector3d& relative_T,
        std::vector<Eigen::Quaterniond>& q_cam_rel, std::vector<Eigen::Vector3d>& t_arr,
        const std::unordered_set<int>& initial_feature_ids, std::vector<SFMFeature> features);

    void alignToReference(
        int frame_num, std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps);

    bool solveFramePose(
        Eigen::Matrix3d& rotation, Eigen::Vector3d& translation, int frame_id,
        const std::vector<SFMFeature>& features);

    void triangulateFeatures(
        const std::vector<bool>& solved, const std::vector<Eigen::Matrix<double, 3, 4>>& poses,
        std::vector<SFMFeature>& features,
        const std::unordered_set<int>* allowed_feature_ids = nullptr);

    void scoreByCheirality(
        const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
        const std::vector<cv::Point2f>& pts_other, std::vector<PoseCandidate>& scored);

    int min_points_, min_inliers_;
    double epipolar_threshold_;
    double pnp_threshold_;
    int ba_iterations_;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_INITIAL_INITIAL_SFM_H_
