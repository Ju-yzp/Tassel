#ifndef TASSEL_CORE_INITIAL_INITIAL_SFM_H_
#define TASSEL_CORE_INITIAL_INITIAL_SFM_H_

#include <Eigen/Core>
#include <unordered_set>
#include <vector>

#include <opencv2/core/types.hpp>

#include "frond_end/feature_manager.h"
#include "state/state.h"

namespace tassel_core {

struct SFMFeature {
    bool state;
    int id;
    std::vector<std::pair<int, Eigen::Vector2d>> observation;
    double position[3];
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
        State& cur_state, FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
        std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out,
        int first_frame_index = 0);

private:
    struct PoseCandidate {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        int score = 0;
        double prior_error = 0.0;
    };

    int selectSeedFrame(int frame_num, const std::vector<SFMFeature>& sfm_f);

    std::vector<std::pair<int, int>> findParallaxFrames(
        int seed_id, int frame_num, const std::vector<SFMFeature>& sfm_f);

    bool computeEssential(
        int seed_id, int other_id, const std::vector<SFMFeature>& sfm_f,
        std::vector<PoseCandidate>& candidates, std::vector<cv::Point2f>& pts_seed,
        std::vector<cv::Point2f>& pts_other, std::unordered_set<int>& inlier_feature_ids);

    void decomposeEssentialMat(
        const Eigen::Matrix3d& essential, std::vector<PoseCandidate>& candidates);

    bool estimateTranslationDirection(
        int seed_id, int other_id, const Eigen::Matrix3d& rotation_other_seed,
        const std::vector<SFMFeature>& sfm_f, std::vector<PoseCandidate>& candidates,
        std::vector<cv::Point2f>& pts_seed, std::vector<cv::Point2f>& pts_other,
        std::unordered_set<int>& inlier_feature_ids);

    bool resolvePose(
        const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
        const std::vector<cv::Point2f>& pts_other, const Eigen::Matrix3d& rotation_prior,
        PoseCandidate& selected);

    bool reconstructScene(
        int frame_num, int seed_id, int other_id, const Eigen::Vector3d& relative_T,
        std::vector<Eigen::Quaterniond>& q_cam_rel, std::vector<Eigen::Vector3d>& t_arr,
        const std::unordered_set<int>& initial_feature_ids, std::vector<SFMFeature>& sfm_f);

    void alignToReference(
        int frame_num, std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps);

    bool registerFramePnP(
        Eigen::Matrix3d& R_initial, Eigen::Vector3d& P_initial, int frame_idx,
        std::vector<SFMFeature>& sfm_f);

    void triangulateTwoFrames(
        int frame0, Eigen::Matrix<double, 3, 4>& Pose0, int frame1,
        Eigen::Matrix<double, 3, 4>& Pose1, std::vector<SFMFeature>& sfm_f,
        const std::unordered_set<int>* allowed_feature_ids = nullptr);

    void scoreByCheirality(
        const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
        const std::vector<cv::Point2f>& pts_other, std::vector<PoseCandidate>& scored);

    int min_points_, min_inliers_;
    double epipolar_threshold_;
    double pnp_threshold_;
    int ba_iterations_;
    int feature_num_ = 0;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_INITIAL_INITIAL_SFM_H_
