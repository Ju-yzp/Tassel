#ifndef TASSEL_CORE_INITIAL_INITIAL_SFM_H_
#define TASSEL_CORE_INITIAL_INITIAL_SFM_H_

#include <Eigen/Core>
#include <map>
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
        double min_depth = 0.1, double max_depth = 3.0, int min_seed_pts = 10,
        int min_e_inliers = 8, double e_ransac_threshold = 0.004, int min_pnp_pts = 10,
        double pnp_reproj_threshold = 0.03, double max_bad_pnp_ratio = 0.3,
        int ba_max_iterations = 30, int ba_num_threads = 5)
        : min_depth_(min_depth),
          max_depth_(max_depth),
          min_seed_pts_(min_seed_pts),
          min_e_inliers_(min_e_inliers),
          e_ransac_threshold_(e_ransac_threshold),
          min_pnp_pts_(min_pnp_pts),
          pnp_reproj_threshold_(pnp_reproj_threshold),
          max_bad_pnp_ratio_(max_bad_pnp_ratio),
          ba_max_iterations_(ba_max_iterations),
          ba_num_threads_(ba_num_threads) {}

    bool construct(
        State& cur_state, FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1,
        std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out);

private:
    struct Observation {
        bool is_stereo = false;
        Eigen::Vector3d uv_l, uv_r;
        double depth = std::numeric_limits<double>::quiet_NaN();
        int feature_id = -1;
    };

    struct PoseCandidate {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        int score = 0;
    };

    void collectStereoDepths(
        int frame_num, FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
        const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1,
        std::vector<std::vector<Observation>>& all_frames);

    int selectSeedFrame(int frame_num, const std::vector<std::vector<Observation>>& all_frames);

    std::vector<std::pair<int, int>> findParallaxFrames(
        int seed_id, int frame_num, const std::vector<std::vector<Observation>>& all_frames,
        const std::vector<SFMFeature>& sfm_f);

    bool computeEssential(
        int seed_id, int other_id, const std::vector<SFMFeature>& sfm_f,
        std::vector<PoseCandidate>& candidates, std::vector<cv::Point2f>& pts_seed,
        std::vector<cv::Point2f>& pts_other);

    bool resolvePose(
        const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
        const std::vector<cv::Point2f>& pts_other, PoseCandidate& selected);

    bool runBA(
        int frame_num, int seed_id, int other_id, const Eigen::Vector3d& relative_T,
        std::vector<Eigen::Quaterniond>& q_cam_rel, std::vector<Eigen::Vector3d>& t_arr,
        std::vector<SFMFeature>& sfm_f, std::map<int, Eigen::Vector3d>& tracked_pts);

    void alignToReference(
        int frame_num, std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps);

    bool registerFramePnP(
        Eigen::Matrix3d& R_initial, Eigen::Vector3d& P_initial, int frame_idx,
        std::vector<SFMFeature>& sfm_f);

    void triangulateTwoFrames(
        int frame0, Eigen::Matrix<double, 3, 4>& Pose0, int frame1,
        Eigen::Matrix<double, 3, 4>& Pose1, std::vector<SFMFeature>& sfm_f);

    double triangulateStereo(
        const Eigen::Vector3d& uv_l, const Eigen::Vector3d& uv_r, const Eigen::Matrix3d& R_lr,
        const Eigen::Vector3d& P_lr);

    void decomposeEssentialMat(const Eigen::Matrix3d& E, std::vector<PoseCandidate>& candidates);

    void scoreByCheirality(
        const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
        const std::vector<cv::Point2f>& pts_other, std::vector<PoseCandidate>& scored);

    double min_depth_, max_depth_;
    int min_seed_pts_, min_e_inliers_;
    double e_ransac_threshold_;
    int min_pnp_pts_;
    double pnp_reproj_threshold_, max_bad_pnp_ratio_;
    int ba_max_iterations_, ba_num_threads_;
    int feature_num_ = 0;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_INITIAL_INITIAL_SFM_H_
