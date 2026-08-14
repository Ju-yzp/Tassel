#include "initial/initial_sfm.h"

#include <spdlog/spdlog.h>
#include <algorithm>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>

#include "frond_end/feature_manager.h"
#include "state/state.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
namespace {

struct SfmReprojectionFactor {
public:
    explicit SfmReprojectionFactor(const Eigen::Vector2d& observation)
        : observation_(observation) {}

    template <typename T>
    bool operator()(
        const T* const rotation, const T* const translation, const T* const point,
        T* residuals) const {
        T point_camera[3];
        ceres::QuaternionRotatePoint(rotation, point, point_camera);
        point_camera[0] += translation[0];
        point_camera[1] += translation[1];
        point_camera[2] += translation[2];
        residuals[0] = point_camera[0] / point_camera[2] - T(observation_.x());
        residuals[1] = point_camera[1] / point_camera[2] - T(observation_.y());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& observation) {
        return new ceres::AutoDiffCostFunction<SfmReprojectionFactor, 2, 4, 3, 3>(
            new SfmReprojectionFactor(observation));
    }

private:
    Eigen::Vector2d observation_;
};

}  // namespace

std::vector<std::pair<int, int>> InitialSFM::scoreBaselineFrames(
    int first_frame_index, int host_id, const std::vector<Eigen::Quaterniond>& camera_rotations,
    const FeatureManager& feature_manager) {
    // 返回 {目标帧局部索引, 旋转补偿后有效视差点数}。
    std::vector<int> scores(camera_rotations.size(), 0);
    const int host_frame_index = first_frame_index + host_id;
    const Eigen::Matrix3d host_rotation = camera_rotations[host_id].toRotationMatrix();
    for (const auto& [_, feature] : feature_manager.features()) {
        const int host_observation_index = host_frame_index - feature.host_frame_index;
        if (host_observation_index < 0 ||
            host_observation_index >= static_cast<int>(feature.observations.size())) {
            continue;
        }

        const Eigen::Vector3d host_ray =
            feature.observations[host_observation_index].uv.normalized();
        for (size_t observation_index = 0; observation_index < feature.observations.size();
             ++observation_index) {
            const int frame_id =
                feature.observationFrameIndex(observation_index) - first_frame_index;
            if (frame_id == host_id || frame_id < 0 ||
                frame_id >= static_cast<int>(camera_rotations.size())) {
                continue;
            }
            const Eigen::Vector3d target_ray =
                feature.observations[observation_index].uv.normalized();
            const Eigen::Matrix3d host_to_target =
                camera_rotations[frame_id].inverse().toRotationMatrix() * host_rotation;
            const double parallax = (host_to_target * host_ray).cross(target_ray).norm();
            if (parallax > epipolar_threshold_) {
                ++scores[frame_id];
            }
        }
    }

    std::vector<std::pair<int, int>> candidates;
    for (int frame_id = 0; frame_id < static_cast<int>(scores.size()); ++frame_id) {
        if (scores[frame_id] >= min_points_) {
            candidates.emplace_back(frame_id, scores[frame_id]);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });
    if (candidates.size() > 2) {
        candidates.resize(2);
    }
    if (candidates.empty()) {
        const int best_score = scores.empty() ? 0 : *std::max_element(scores.begin(), scores.end());
        spdlog::warn(
            "SFM baseline failed: host={}, rotated_parallax_points={}, required={}", host_id,
            best_score, min_points_);
    } else {
        spdlog::info(
            "SFM host frame {}: candidates={}, best_score={}", host_id, candidates.size(),
            candidates.front().second);
    }
    return candidates;
}

bool InitialSFM::computeEssential(
    int seed_id, int other_id, const std::vector<SFMFeature>& features,
    std::vector<PoseCandidate>& candidates, std::vector<cv::Point2f>& pts_seed,
    std::vector<cv::Point2f>& pts_other, std::unordered_set<int>& inlier_feature_ids) {
    std::vector<int> feature_ids;
    for (const auto& feature : features) {
        bool in_seed = false;
        bool in_other = false;
        Eigen::Vector2d uv_seed = Eigen::Vector2d::Zero();
        Eigen::Vector2d uv_other = Eigen::Vector2d::Zero();
        for (const auto& [frame_id, uv] : feature.observations) {
            if (frame_id == seed_id) {
                in_seed = true;
                uv_seed = uv;
            }
            if (frame_id == other_id) {
                in_other = true;
                uv_other = uv;
            }
        }
        if (in_seed && in_other) {
            pts_seed.emplace_back(uv_seed.x(), uv_seed.y());
            pts_other.emplace_back(uv_other.x(), uv_other.y());
            feature_ids.push_back(feature.id);
        }
    }

    const int required_correspondences = std::max(5, min_inliers_);
    if (static_cast<int>(pts_seed.size()) < required_correspondences) {
        spdlog::warn(
            "SFM essential failed: seed={}, other={}, correspondences={}, required={}", seed_id,
            other_id, pts_seed.size(), required_correspondences);
        return false;
    }

    const cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat inlier_mask;
    const cv::Mat essential = cv::findEssentialMat(
        pts_seed, pts_other, camera_matrix, cv::RANSAC, 0.99, epipolar_threshold_, inlier_mask);
    if (essential.empty() || inlier_mask.empty()) {
        spdlog::warn(
            "SFM essential failed: seed={}, other={}, reason=no_model, correspondences={}", seed_id,
            other_id, pts_seed.size());
        return false;
    }
    const int inlier_count = cv::countNonZero(inlier_mask);
    if (inlier_count < min_inliers_) {
        spdlog::warn(
            "SFM essential failed: seed={}, other={}, inliers={}/{}, required={}, threshold={:.6f}",
            seed_id, other_id, inlier_count, pts_seed.size(), min_inliers_, epipolar_threshold_);
        return false;
    }
    if (inlier_mask.total() != pts_seed.size() || !inlier_mask.isContinuous()) {
        throw std::logic_error("Essential inlier mask does not match correspondences");
    }

    std::vector<cv::Point2f> inlier_seed;
    std::vector<cv::Point2f> inlier_other;
    inlier_seed.reserve(inlier_count);
    inlier_other.reserve(inlier_count);
    inlier_feature_ids.clear();
    inlier_feature_ids.reserve(inlier_count);
    const uchar* mask = inlier_mask.ptr<uchar>();
    for (size_t i = 0; i < pts_seed.size(); ++i) {
        if (mask[i] == 0) {
            continue;
        }
        inlier_seed.push_back(pts_seed[i]);
        inlier_other.push_back(pts_other[i]);
        inlier_feature_ids.insert(feature_ids[i]);
    }
    pts_seed = std::move(inlier_seed);
    pts_other = std::move(inlier_other);

    if (essential.rows != 3 || essential.cols != 3) {
        throw std::logic_error("Essential matrix must be 3x3");
    }
    cv::Mat rotation1_cv;
    cv::Mat rotation2_cv;
    cv::Mat translation_cv;
    cv::decomposeEssentialMat(essential, rotation1_cv, rotation2_cv, translation_cv);
    Eigen::Matrix3d rotation1;
    Eigen::Matrix3d rotation2;
    Eigen::Vector3d translation;
    cv::cv2eigen(rotation1_cv, rotation1);
    cv::cv2eigen(rotation2_cv, rotation2);
    cv::cv2eigen(translation_cv, translation);
    translation.normalize();
    candidates = {
        {rotation1, translation},
        {rotation1, -translation},
        {rotation2, translation},
        {rotation2, -translation}};
    spdlog::info(
        "SFM essential: seed={}, other={}, inliers={}/{}, threshold={:.6f}", seed_id, other_id,
        inlier_count, feature_ids.size(), epipolar_threshold_);
    return true;
}

bool InitialSFM::resolvePose(
    const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
    const std::vector<cv::Point2f>& pts_other, const Eigen::Matrix3d& rotation_prior,
    PoseCandidate& selected) {
    std::vector<PoseCandidate> scored;
    scoreByCheirality(candidates, pts_seed, pts_other, scored);

    constexpr int kMinCheiralityPoints = 5;
    if (scored.empty()) {
        spdlog::warn(
            "SFM pose failed: cheirality_points=0, required={}, correspondences={}",
            kMinCheiralityPoints, pts_seed.size());
        return false;
    }
    for (auto& candidate : scored) {
        candidate.prior_error = Eigen::AngleAxisd(rotation_prior.transpose() * candidate.R).angle();
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        return lhs.prior_error < rhs.prior_error;
    });
    if (scored.front().score < kMinCheiralityPoints) {
        spdlog::warn(
            "SFM pose failed: cheirality_points={}, required={}, correspondences={}",
            scored.front().score, kMinCheiralityPoints, pts_seed.size());
        return false;
    }
    selected = scored.front();
    spdlog::debug(
        "SFM pose selected: cheirality={}/{}, gyro_rotation_error={:.3f}deg", selected.score,
        pts_seed.size(), selected.prior_error * 180.0 / 3.141592653589793);
    return true;
}

bool InitialSFM::reconstructScene(
    int frame_num, int seed_id, int other_id, const Eigen::Vector3d& relative_T,
    std::vector<Eigen::Quaterniond>& q_cam_rel, std::vector<Eigen::Vector3d>& t_arr,
    const std::unordered_set<int>& initial_feature_ids, std::vector<SFMFeature> features) {
    t_arr[seed_id].setZero();
    t_arr[other_id] = relative_T;

    std::vector<Eigen::Matrix3d> camera_rotations(frame_num);
    std::vector<Eigen::Vector3d> camera_translations(frame_num);
    std::vector<std::array<double, 4>> rotation_params(frame_num);
    std::vector<std::array<double, 3>> translation_params(frame_num);
    std::vector<Eigen::Matrix<double, 3, 4>> poses(frame_num);

    // q_cam_rel/t_arr 为相机到种子世界系位姿，三角化使用世界到相机投影矩阵。
    const auto initialize_pose = [&](int frame_id) {
        camera_rotations[frame_id] = q_cam_rel[frame_id].inverse().toRotationMatrix();
        camera_translations[frame_id] = -(camera_rotations[frame_id] * t_arr[frame_id]);
        poses[frame_id].leftCols<3>() = camera_rotations[frame_id];
        poses[frame_id].rightCols<1>() = camera_translations[frame_id];
    };
    initialize_pose(seed_id);
    initialize_pose(other_id);

    std::vector<bool> solved(frame_num, false);
    solved[seed_id] = true;
    solved[other_id] = true;
    triangulateFeatures(solved, poses, features, &initial_feature_ids);

    for (int solved_count = 2; solved_count < frame_num; ++solved_count) {
        int best_frame = -1;
        int best_point_count = 0;
        for (int frame_id = 0; frame_id < frame_num; ++frame_id) {
            if (solved[frame_id]) {
                continue;
            }
            int point_count = 0;
            for (const auto& feature : features) {
                if (!feature.triangulated) {
                    continue;
                }
                for (const auto& [observed_frame, _] : feature.observations) {
                    if (observed_frame == frame_id) {
                        ++point_count;
                        break;
                    }
                }
            }
            if (point_count > best_point_count) {
                best_point_count = point_count;
                best_frame = frame_id;
            }
        }
        if (best_frame < 0 || best_point_count < min_points_) {
            spdlog::warn(
                "SFM failed: no unsolved frame has enough 3D correspondences, "
                "best_frame={}, points={}, required={}",
                best_frame, best_point_count, min_points_);
            return false;
        }

        Eigen::Matrix3d frame_rotation;
        Eigen::Vector3d frame_translation;
        if (!solveFramePose(frame_rotation, frame_translation, best_frame, features)) {
            return false;
        }
        camera_rotations[best_frame] = frame_rotation;
        camera_translations[best_frame] = frame_translation;
        poses[best_frame].leftCols<3>() = frame_rotation;
        poses[best_frame].rightCols<1>() = frame_translation;
        solved[best_frame] = true;

        triangulateFeatures(solved, poses, features);
    }

    {
        ceres::Problem problem;
        ceres::Manifold* quat_manifold = new ceres::QuaternionManifold();
        for (int i = 0; i < frame_num; i++) {
            const Eigen::Quaterniond rotation(camera_rotations[i]);
            rotation_params[i] = {rotation.w(), rotation.x(), rotation.y(), rotation.z()};
            translation_params[i] = {
                camera_translations[i].x(), camera_translations[i].y(), camera_translations[i].z()};
            problem.AddParameterBlock(rotation_params[i].data(), 4, quat_manifold);
            problem.AddParameterBlock(translation_params[i].data(), 3);
            if (i == seed_id) {
                problem.SetParameterBlockConstant(rotation_params[i].data());
            }
            // seed 平移固定世界原点，基线帧平移固定为视觉估计的单位尺度。
            if (i == seed_id || i == other_id) {
                problem.SetParameterBlockConstant(translation_params[i].data());
            }
        }

        int observation_count = 0;
        for (auto& feature : features) {
            if (!feature.triangulated) {
                continue;
            }
            problem.AddParameterBlock(feature.position.data(), 3);
            for (const auto& [frame_index, observation] : feature.observations) {
                problem.AddResidualBlock(
                    SfmReprojectionFactor::Create(observation),
                    new ceres::HuberLoss(epipolar_threshold_), rotation_params[frame_index].data(),
                    translation_params[frame_index].data(), feature.position.data());
                ++observation_count;
            }
        }

        if (observation_count == 0) {
            spdlog::warn("SFM BA failed: no valid observations");
            return false;
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = ba_iterations_;
        // 初始化 BA 规模较小，固定单线程避免并行调度成本。
        options.num_threads = 1;
        options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        spdlog::info(
            "SFM BA: {} observations, {}, iters={}, cost={:.4e}", observation_count,
            summary.termination_type == ceres::CONVERGENCE ? "CONV" : "NOCONV",
            summary.iterations.size(), summary.final_cost);
        if (!summary.IsSolutionUsable()) {
            spdlog::warn(
                "SFM BA failed: termination={}, message={}",
                static_cast<int>(summary.termination_type), summary.message);
            return false;
        }

        for (int i = 0; i < frame_num; i++) {
            const Eigen::Quaterniond world_to_camera =
                Eigen::Quaterniond(
                    rotation_params[i][0], rotation_params[i][1], rotation_params[i][2],
                    rotation_params[i][3])
                    .normalized();
            const Eigen::Vector3d translation(
                translation_params[i][0], translation_params[i][1], translation_params[i][2]);
            q_cam_rel[i] = world_to_camera.inverse();
            t_arr[i] = -(q_cam_rel[i] * translation);
        }
    }

    return true;
}

void InitialSFM::alignToReference(
    int frame_num, std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps) {
    Eigen::Matrix3d R_seed_c0 = Rs[0].transpose();
    Eigen::Vector3d Ps0 = Ps[0];
    for (int i = 0; i < frame_num; i++) {
        Rs[i] = R_seed_c0 * Rs[i];
        Ps[i] = R_seed_c0 * (Ps[i] - Ps0);
    }
}

bool InitialSFM::solveFramePose(
    Eigen::Matrix3d& rotation, Eigen::Vector3d& translation, int i,
    const std::vector<SFMFeature>& features) {
    std::vector<Eigen::Vector2d> observations;
    std::vector<Eigen::Vector3d> world_points;
    for (const auto& feature : features) {
        if (!feature.triangulated) {
            continue;
        }
        for (const auto& [frame_id, observation] : feature.observations) {
            if (frame_id == i) {
                observations.push_back(observation);
                world_points.emplace_back(
                    feature.position[0], feature.position[1], feature.position[2]);
                break;
            }
        }
    }
    const int n_pts = static_cast<int>(observations.size());
    if (n_pts < min_points_) {
        spdlog::warn("SFM PnP failed: frame={}, points={}, required={}", i, n_pts, min_points_);
        return false;
    }

    std::vector<cv::Point3f> pnp_world_points;
    std::vector<cv::Point2f> pnp_observations;
    pnp_world_points.reserve(n_pts);
    pnp_observations.reserve(n_pts);
    for (int k = 0; k < n_pts; ++k) {
        pnp_world_points.emplace_back(
            static_cast<float>(world_points[k].x()), static_cast<float>(world_points[k].y()),
            static_cast<float>(world_points[k].z()));
        pnp_observations.emplace_back(
            static_cast<float>(observations[k].x()), static_cast<float>(observations[k].y()));
    }
    cv::Mat rotation_vector;
    cv::Mat translation_vector = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<int> pnp_inliers;
    if (!cv::solvePnPRansac(
            pnp_world_points, pnp_observations, cv::Mat::eye(3, 3, CV_64F), cv::noArray(),
            rotation_vector, translation_vector, false, 200, pnp_threshold_, 0.99, pnp_inliers,
            cv::SOLVEPNP_EPNP) ||
        static_cast<int>(pnp_inliers.size()) < min_points_) {
        spdlog::warn(
            "SFM PnP failed: frame={}, reason=insufficient_ransac_inliers, inliers={}/{}, "
            "required={}, threshold={:.6f}",
            i, pnp_inliers.size(), n_pts, min_points_, pnp_threshold_);
        return false;
    }

    std::vector<cv::Point3f> inlier_world_points;
    std::vector<cv::Point2f> inlier_observations;
    inlier_world_points.reserve(pnp_inliers.size());
    inlier_observations.reserve(pnp_inliers.size());
    for (const int inlier : pnp_inliers) {
        inlier_world_points.push_back(pnp_world_points[inlier]);
        inlier_observations.push_back(pnp_observations[inlier]);
    }
    if (!cv::solvePnP(
            inlier_world_points, inlier_observations, cv::Mat::eye(3, 3, CV_64F), cv::noArray(),
            rotation_vector, translation_vector, true, cv::SOLVEPNP_ITERATIVE)) {
        spdlog::warn("SFM PnP failed: frame={}, reason=no_solution, points={}", i, n_pts);
        return false;
    }
    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);
    cv::cv2eigen(rotation_matrix, rotation);
    for (int d = 0; d < 3; ++d) {
        translation[d] = translation_vector.at<double>(d);
    }
    if (!rotation.allFinite() || !translation.allFinite()) {
        spdlog::warn("SFM PnP failed: frame={}, reason=non_finite_solution", i);
        return false;
    }

    int pnp_bad = 0;
    double pnp_inlier_error = 0.0;
    for (int k = 0; k < n_pts; ++k) {
        const Eigen::Vector3d point_camera = rotation * world_points[k] + translation;
        if (!point_camera.allFinite() || point_camera.z() <= 1e-12) {
            ++pnp_bad;
            continue;
        }
        const double error = (point_camera.head<2>() / point_camera.z() - observations[k]).norm();
        if (error > pnp_threshold_) {
            ++pnp_bad;
        } else {
            pnp_inlier_error += error;
        }
    }
    const int pnp_inlier_count = n_pts - pnp_bad;
    if (pnp_inlier_count < min_points_) {
        spdlog::warn(
            "SFM PnP failed: frame={}, inliers={}/{}, required={}, mean_inlier_error={:.6f}, "
            "threshold={:.6f}",
            i, pnp_inlier_count, n_pts, min_points_,
            pnp_inlier_count > 0 ? pnp_inlier_error / pnp_inlier_count
                                 : std::numeric_limits<double>::infinity(),
            pnp_threshold_);
        return false;
    }
    return true;
}

void InitialSFM::triangulateFeatures(
    const std::vector<bool>& solved, const std::vector<Eigen::Matrix<double, 3, 4>>& poses,
    std::vector<SFMFeature>& features, const std::unordered_set<int>* allowed_feature_ids) {
    TASSEL_ASSERT(solved.size() == poses.size());
    for (auto& feature : features) {
        if (feature.triangulated ||
            (allowed_feature_ids != nullptr && !allowed_feature_ids->contains(feature.id))) {
            continue;
        }

        std::vector<Eigen::Matrix<double, 3, 4>> feature_poses;
        std::vector<Eigen::Vector2d> observations;
        for (const auto& [frame_id, uv] : feature.observations) {
            if (frame_id < 0 || frame_id >= static_cast<int>(solved.size()) || !solved[frame_id]) {
                continue;
            }
            feature_poses.push_back(poses[frame_id]);
            observations.push_back(uv);
        }
        if (observations.size() < 2) {
            continue;
        }

        double condition = std::numeric_limits<double>::infinity();
        const Eigen::Vector4d homogeneous =
            tassel_utils::triangulateMultiView(feature_poses, observations, &condition);
        if (!std::isfinite(condition) || condition >= 1e6 || !std::isfinite(homogeneous.w()) ||
            std::abs(homogeneous.w()) <= 1e-12) {
            continue;
        }
        const Eigen::Vector3d point = tassel_utils::dehomogenize(homogeneous);
        if (!point.allFinite()) {
            continue;
        }
        bool has_positive_depth = true;
        for (const auto& pose : feature_poses) {
            const Eigen::Vector3d point_camera = pose.leftCols<3>() * point + pose.rightCols<1>();
            if (!point_camera.allFinite() || point_camera.z() <= 0.1) {
                has_positive_depth = false;
                break;
            }
        }
        if (!has_positive_depth) {
            continue;
        }

        feature.triangulated = true;
        feature.position[0] = point.x();
        feature.position[1] = point.y();
        feature.position[2] = point.z();
    }
}

void InitialSFM::scoreByCheirality(
    const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
    const std::vector<cv::Point2f>& pts_other, std::vector<PoseCandidate>& scored) {
    int npts = static_cast<int>(pts_seed.size());
    scored = candidates;
    if (npts < 5) {
        return;
    }

    Eigen::Matrix<double, 3, 4> P0 = Eigen::Matrix<double, 3, 4>::Identity();

    for (size_t c = 0; c < candidates.size(); ++c) {
        const auto& cand = candidates[c];
        Eigen::Matrix<double, 3, 4> P1;
        P1.block<3, 3>(0, 0) = cand.R;
        P1.block<3, 1>(0, 3) = cand.t;

        int valid = 0;
        for (int i = 0; i < npts; i++) {
            Eigen::Vector2d uv0(pts_seed[i].x, pts_seed[i].y);
            Eigen::Vector2d uv1(pts_other[i].x, pts_other[i].y);
            Eigen::Vector4d pt = tassel_utils::triangulateTwoView(P0, uv0, P1, uv1);
            if (std::abs(pt(3)) < 1e-10) {
                continue;
            }
            Eigen::Vector3d X = tassel_utils::dehomogenize(pt);
            if (X.z() <= 0.1) {
                continue;
            }
            Eigen::Vector3d X2 = cand.R * X + cand.t;
            if (X2.z() <= 0.1) {
                continue;
            }
            ++valid;
        }
        scored[c].score = valid;
    }
}

bool InitialSFM::construct(
    const State& state, const FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
    std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out,
    int first_frame_index) {
    if (first_frame_index < 0 || first_frame_index > state.latest_active_frame_index) {
        spdlog::error(
            "SFM input failed: first_frame={}, latest_frame={}", first_frame_index,
            state.latest_active_frame_index);
        return false;
    }
    int frame_num = state.latest_active_frame_index - first_frame_index + 1;
    if (frame_num < 2) {
        spdlog::error("SFM input failed: frames={}, required=2", frame_num);
        return false;
    }

    std::vector<Eigen::Quaterniond> q_cam_i0(frame_num);
    for (int i = 0; i < frame_num; i++) {
        q_cam_i0[i] =
            Eigen::Quaterniond(state.frames[first_frame_index + i].rot_w_i * ric).normalized();
    }

    // SFMFeature 的帧索引从 first_frame_index 映射到窗口局部索引。
    const int seed_id = frame_num / 2;
    const auto other_candidates =
        scoreBaselineFrames(first_frame_index, seed_id, q_cam_i0, feature_manager);
    if (other_candidates.empty()) {
        return false;
    }
    auto features = feature_manager.collectSFMFeatures(state, first_frame_index);

    for (const auto& candidate : other_candidates) {
        const int other_id = candidate.first;
        std::vector<PoseCandidate> candidates;
        std::vector<cv::Point2f> pts_seed, pts_other;
        std::unordered_set<int> inlier_feature_ids;
        if (!computeEssential(
                seed_id, other_id, features, candidates, pts_seed, pts_other, inlier_feature_ids)) {
            continue;
        }

        const Eigen::Matrix3d rotation_prior =
            (q_cam_i0[other_id].inverse() * q_cam_i0[seed_id]).toRotationMatrix();
        PoseCandidate selected;
        if (!resolvePose(candidates, pts_seed, pts_other, rotation_prior, selected)) {
            continue;
        }

        std::vector<Eigen::Quaterniond> q_cam_rel(frame_num);
        q_cam_rel[seed_id] = Eigen::Quaterniond::Identity();
        // selected.R 为 R_other_seed，q_cam_rel 保存相机到种子世界系的旋转。
        q_cam_rel[other_id] = Eigen::Quaterniond(selected.R.transpose());

        Eigen::Vector3d T_dir = (-selected.R.transpose() * selected.t).normalized();

        std::vector<Eigen::Vector3d> t_arr(frame_num, Eigen::Vector3d::Zero());
        // SFM 中间点按候选复制并由 reconstructScene 独占，函数返回后立即释放。
        if (!reconstructScene(
                frame_num, seed_id, other_id, T_dir, q_cam_rel, t_arr, inlier_feature_ids,
                features)) {
            spdlog::warn(
                "SFM candidate failed: seed={}, other={}, stage=reconstruction", seed_id, other_id);
            continue;
        }

        Rs_out.resize(frame_num);
        Ps_out.resize(frame_num);
        for (int i = 0; i < frame_num; i++) {
            Rs_out[i] = q_cam_rel[i].toRotationMatrix();
            Ps_out[i] = t_arr[i];
        }

        alignToReference(frame_num, Rs_out, Ps_out);

        spdlog::info("SFM: {} frames, seed={}, other={}", frame_num, seed_id, other_id);
        return true;
    }

    spdlog::warn("SFM failed: all candidates were rejected");
    return false;
}

}  // namespace tassel_core
