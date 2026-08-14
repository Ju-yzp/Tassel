#include "initial/initial_sfm.h"

#include <spdlog/spdlog.h>
#include <algorithm>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>

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

int InitialSFM::selectSeedFrame(int frame_num, const std::vector<SFMFeature>& sfm_f) {
    std::vector<int> feature_count(frame_num, 0);
    std::vector<int> connectivity_per_frame(frame_num, 0);
    for (const auto& feature : sfm_f) {
        for (const auto& [frame_id, _] : feature.observation) {
            if (frame_id < 0 || frame_id >= frame_num) {
                continue;
            }
            ++feature_count[frame_id];
            for (const auto& [other_id, __] : feature.observation) {
                const int distance = std::abs(frame_id - other_id);
                if (distance > 0 && distance <= 3) {
                    connectivity_per_frame[frame_id] += 4 - distance;
                }
            }
        }
    }

    int seed_id = -1;
    int best_score = -1;
    for (int i = 0; i < frame_num; ++i) {
        if (feature_count[i] < min_points_) {
            continue;
        }
        if (connectivity_per_frame[i] > best_score) {
            best_score = connectivity_per_frame[i];
            seed_id = i;
        }
    }

    if (seed_id < 0) {
        const int max_features = *std::max_element(feature_count.begin(), feature_count.end());
        spdlog::warn(
            "SFM seed failed: frames={}, max_features={}, required={}", frame_num, max_features,
            min_points_);
        return -1;
    }

    spdlog::info(
        "SFM mono seed frame {}: features={}, connectivity={}", seed_id, feature_count[seed_id],
        best_score);
    return seed_id;
}

std::vector<std::pair<int, int>> InitialSFM::findParallaxFrames(
    int seed_id, int frame_num, const std::vector<SFMFeature>& sfm_f) {
    struct FrameCandidate {
        int frame_id;
        int common_count;
        int frame_distance;
        double median_parallax;
    };

    std::vector<FrameCandidate> connected_candidates;
    int max_common = 0;
    for (int i = 0; i < frame_num; ++i) {
        if (i == seed_id) {
            continue;
        }
        int common = 0;
        std::vector<double> parallaxes;
        for (const auto& f : sfm_f) {
            bool in_seed = false, in_other = false;
            Eigen::Vector2d uv_seed, uv_other;
            for (const auto& [fid, uv] : f.observation) {
                if (fid == seed_id) {
                    in_seed = true;
                    uv_seed = uv;
                }
                if (fid == i) {
                    in_other = true;
                    uv_other = uv;
                }
            }
            if (!in_seed || !in_other) {
                continue;
            }
            ++common;
            parallaxes.push_back((uv_seed - uv_other).norm());
        }
        max_common = std::max(max_common, common);
        if (common < min_points_) {
            continue;
        }
        const size_t mid = parallaxes.size() / 2;
        std::nth_element(parallaxes.begin(), parallaxes.begin() + mid, parallaxes.end());
        connected_candidates.push_back({i, common, std::abs(i - seed_id), parallaxes[mid]});
    }

    std::vector<FrameCandidate> parallax_candidates;
    for (const auto& candidate : connected_candidates) {
        if (candidate.median_parallax > epipolar_threshold_) {
            parallax_candidates.push_back(candidate);
        }
    }

    if (parallax_candidates.empty()) {
        if (connected_candidates.empty()) {
            spdlog::warn(
                "SFM baseline failed: seed={}, max_common={}, required={}", seed_id, max_common,
                min_points_);
        } else {
            const auto best = std::max_element(
                connected_candidates.begin(), connected_candidates.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.median_parallax < rhs.median_parallax;
                });
            spdlog::warn(
                "SFM baseline failed: seed={}, max_median_parallax={:.6f}, required>{:.6f}",
                seed_id, best->median_parallax, epipolar_threshold_);
        }
        return {};
    }

    std::sort(
        parallax_candidates.begin(), parallax_candidates.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.median_parallax != rhs.median_parallax) {
                return lhs.median_parallax > rhs.median_parallax;
            }
            return lhs.common_count > rhs.common_count;
        });

    std::vector<std::pair<int, int>> other_candidates;
    const size_t candidate_count = std::min<size_t>(2, parallax_candidates.size());
    other_candidates.reserve(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
        other_candidates.emplace_back(
            parallax_candidates[i].frame_id, parallax_candidates[i].common_count);
    }
    return other_candidates;
}

bool InitialSFM::computeEssential(
    int seed_id, int other_id, const std::vector<SFMFeature>& sfm_f,
    std::vector<PoseCandidate>& candidates, std::vector<cv::Point2f>& pts_seed,
    std::vector<cv::Point2f>& pts_other, std::unordered_set<int>& inlier_feature_ids) {
    std::vector<int> feature_ids;
    for (const auto& feature : sfm_f) {
        bool in_seed = false;
        bool in_other = false;
        Eigen::Vector2d uv_seed = Eigen::Vector2d::Zero();
        Eigen::Vector2d uv_other = Eigen::Vector2d::Zero();
        for (const auto& [frame_id, uv] : feature.observation) {
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

    Eigen::Matrix3d essential_eigen;
    cv::cv2eigen(essential, essential_eigen);
    decomposeEssentialMat(essential_eigen, candidates);
    spdlog::info(
        "SFM essential: seed={}, other={}, inliers={}/{}, threshold={:.6f}", seed_id, other_id,
        inlier_count, feature_ids.size(), epipolar_threshold_);
    return true;
}

bool InitialSFM::estimateTranslationDirection(
    int seed_id, int other_id, const Eigen::Matrix3d& rotation_other_seed,
    const std::vector<SFMFeature>& sfm_f, std::vector<PoseCandidate>& candidates,
    std::vector<cv::Point2f>& pts_seed, std::vector<cv::Point2f>& pts_other,
    std::unordered_set<int>& inlier_feature_ids) {
    std::vector<int> feature_ids;
    for (const auto& f : sfm_f) {
        bool in_seed = false, in_other = false;
        Eigen::Vector2d uv_seed, uv_other;
        for (const auto& [fid, uv] : f.observation) {
            if (fid == seed_id) {
                in_seed = true;
                uv_seed = uv;
            }
            if (fid == other_id) {
                in_other = true;
                uv_other = uv;
            }
        }
        if (in_seed && in_other) {
            pts_seed.emplace_back(uv_seed.x(), uv_seed.y());
            pts_other.emplace_back(uv_other.x(), uv_other.y());
            feature_ids.push_back(f.id);
        }
    }
    const int required_correspondences = std::max(5, min_inliers_);
    if (static_cast<int>(pts_seed.size()) < required_correspondences) {
        spdlog::warn(
            "SFM translation failed: seed={}, other={}, correspondences={}, required={}", seed_id,
            other_id, pts_seed.size(), required_correspondences);
        return false;
    }

    if (feature_ids.size() != pts_seed.size()) {
        throw std::logic_error("Translation correspondences do not match their feature IDs");
    }

    const int correspondence_count = static_cast<int>(pts_seed.size());
    Eigen::MatrixXd constraints(correspondence_count, 3);
    int valid_constraint_count = 0;
    for (int i = 0; i < correspondence_count; ++i) {
        const Eigen::Vector3d ray_seed =
            Eigen::Vector3d(pts_seed[i].x, pts_seed[i].y, 1.0).normalized();
        const Eigen::Vector3d ray_other =
            Eigen::Vector3d(pts_other[i].x, pts_other[i].y, 1.0).normalized();
        Eigen::Vector3d constraint = (rotation_other_seed * ray_seed).cross(ray_other);
        const double norm = constraint.norm();
        if (!std::isfinite(norm) || norm < 1e-12) {
            constraint.setZero();
        } else {
            constraint /= norm;
            ++valid_constraint_count;
        }
        constraints.row(i) = constraint.transpose();
    }

    Eigen::VectorXd weights = Eigen::VectorXd::Ones(correspondence_count);
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    for (int iteration = 0; iteration < 8; ++iteration) {
        Eigen::MatrixXd weighted_constraints = constraints;
        for (int i = 0; i < correspondence_count; ++i) {
            weighted_constraints.row(i) *= std::sqrt(weights[i]);
        }
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted_constraints, Eigen::ComputeFullV);
        translation = svd.matrixV().col(2).normalized();
        if (!translation.allFinite()) {
            spdlog::warn(
                "SFM translation failed: seed={}, other={}, non-finite IRLS solution at "
                "iteration={}, valid_constraints={}/{}",
                seed_id, other_id, iteration, valid_constraint_count, correspondence_count);
            return false;
        }
        for (int i = 0; i < correspondence_count; ++i) {
            const double scaled_residual =
                std::abs(constraints.row(i).dot(translation)) / epipolar_threshold_;
            // Cauchy IRLS 固定迭代次数和阈值，不依赖随机采样顺序。
            weights[i] = 1.0 / (1.0 + scaled_residual * scaled_residual);
        }
    }

    std::vector<double> residuals;
    residuals.reserve(correspondence_count);
    for (int i = 0; i < correspondence_count; ++i) {
        residuals.push_back(std::abs(constraints.row(i).dot(translation)));
    }
    const auto percentile = [&residuals](double fraction) {
        const size_t index = static_cast<size_t>(fraction * (residuals.size() - 1));
        std::nth_element(residuals.begin(), residuals.begin() + index, residuals.end());
        return residuals[index];
    };
    const double median_residual = percentile(0.50);
    const double p90_residual = percentile(0.90);
    const double max_residual = *std::max_element(residuals.begin(), residuals.end());
    std::vector<int> inlier_indices;
    inlier_indices.reserve(correspondence_count);
    for (int i = 0; i < correspondence_count; ++i) {
        if (std::abs(constraints.row(i).dot(translation)) <= epipolar_threshold_) {
            inlier_indices.push_back(i);
        }
    }
    if (static_cast<int>(inlier_indices.size()) < min_inliers_) {
        spdlog::warn(
            "SFM translation failed: seed={}, other={}, inliers={}/{}, required={}, "
            "threshold={:.6f}, residual_median={:.6f}, residual_p90={:.6f}, "
            "residual_max={:.6f}, rotation_angle={:.3f}deg",
            seed_id, other_id, inlier_indices.size(), correspondence_count, min_inliers_,
            epipolar_threshold_, median_residual, p90_residual, max_residual,
            Eigen::AngleAxisd(rotation_other_seed).angle() * 180.0 / M_PI);
        return false;
    }

    Eigen::MatrixXd inlier_constraints(inlier_indices.size(), 3);
    for (size_t i = 0; i < inlier_indices.size(); ++i) {
        inlier_constraints.row(i) = constraints.row(inlier_indices[i]);
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> inlier_svd(inlier_constraints, Eigen::ComputeFullV);
    translation = inlier_svd.matrixV().col(2).normalized();
    if (!translation.allFinite()) {
        spdlog::warn(
            "SFM translation failed: seed={}, other={}, non-finite inlier solution", seed_id,
            other_id);
        return false;
    }

    std::vector<cv::Point2f> inlier_seed;
    std::vector<cv::Point2f> inlier_other;
    inlier_seed.reserve(inlier_indices.size());
    inlier_other.reserve(inlier_indices.size());
    inlier_feature_ids.clear();
    inlier_feature_ids.reserve(inlier_indices.size());
    // 约束行、两组匹配点和 feature ID 严格同序，初始三角化只能使用最终内点。
    for (const int index : inlier_indices) {
        inlier_seed.push_back(pts_seed[index]);
        inlier_other.push_back(pts_other[index]);
        inlier_feature_ids.insert(feature_ids[index]);
    }
    pts_seed = std::move(inlier_seed);
    pts_other = std::move(inlier_other);

    candidates.clear();
    candidates.push_back({rotation_other_seed, translation});
    candidates.push_back({rotation_other_seed, -translation});
    return true;
}

bool InitialSFM::resolvePose(
    const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
    const std::vector<cv::Point2f>& pts_other, const Eigen::Matrix3d& rotation_prior,
    PoseCandidate& selected) {
    std::vector<PoseCandidate> scored;
    scoreByCheirality(candidates, pts_seed, pts_other, scored);

    constexpr int kMinCheiralityPoints = 5;
    if (scored.empty() || scored.front().score < kMinCheiralityPoints) {
        const int best_score = scored.empty() ? 0 : scored.front().score;
        spdlog::warn(
            "SFM pose failed: cheirality_points={}, required={}, correspondences={}", best_score,
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
    selected = scored.front();
    spdlog::debug(
        "SFM pose selected: cheirality={}/{}, gyro_rotation_error={:.3f}deg", selected.score,
        pts_seed.size(), selected.prior_error * 180.0 / 3.141592653589793);
    return true;
}

bool InitialSFM::reconstructScene(
    int frame_num, int seed_id, int other_id, const Eigen::Vector3d& relative_T,
    std::vector<Eigen::Quaterniond>& q_cam_rel, std::vector<Eigen::Vector3d>& t_arr,
    const std::unordered_set<int>& initial_feature_ids, std::vector<SFMFeature>& sfm_f) {
    feature_num_ = static_cast<int>(sfm_f.size());
    int l = seed_id, last = other_id;

    t_arr[l].setZero();
    t_arr[last] = relative_T;

    std::vector<Eigen::Matrix3d> c_Rotation(frame_num);
    std::vector<Eigen::Vector3d> c_Translation(frame_num);
    std::vector<Eigen::Quaterniond> c_Quat(frame_num);
    std::vector<std::array<double, 4>> c_rotation(frame_num);
    std::vector<std::array<double, 3>> c_translation(frame_num);
    std::vector<Eigen::Matrix<double, 3, 4>> Pose(frame_num);

    c_Quat[l] = q_cam_rel[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * t_arr[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[last] = q_cam_rel[last].inverse();
    c_Rotation[last] = c_Quat[last].toRotationMatrix();
    c_Translation[last] = -1 * (c_Rotation[last] * t_arr[last]);
    Pose[last].block<3, 3>(0, 0) = c_Rotation[last];
    Pose[last].block<3, 1>(0, 3) = c_Translation[last];

    triangulateTwoFrames(l, Pose[l], last, Pose[last], sfm_f, &initial_feature_ids);

    for (int j = 0; j < feature_num_; j++) {
        if (!sfm_f[j].state) {
            continue;
        }
        Eigen::Vector3d X(sfm_f[j].position);
        if ((X - t_arr[l]).dot(q_cam_rel[l].toRotationMatrix().col(2)) <= 0.1 ||
            (X - t_arr[last]).dot(q_cam_rel[last].toRotationMatrix().col(2)) <= 0.1) {
            sfm_f[j].state = false;
        }
    }

    std::vector<bool> solved(frame_num, false);
    solved[l] = true;
    solved[last] = true;

    for (int solved_count = 2; solved_count < frame_num; solved_count++) {
        int best_i = -1, best_ref = -1;
        int best_dist = std::numeric_limits<int>::max();
        for (int i = 0; i < frame_num; i++) {
            if (solved[i]) {
                continue;
            }
            for (int r = 0; r < frame_num; r++) {
                if (!solved[r]) {
                    continue;
                }
                int d = std::abs(i - r);
                if (d < best_dist) {
                    best_dist = d;
                    best_i = i;
                    best_ref = r;
                }
            }
        }
        if (best_i < 0) {
            break;
        }

        Eigen::Matrix3d R_initial = q_cam_rel[best_i].inverse().toRotationMatrix();
        Eigen::Vector3d P_initial = c_Translation[best_ref];
        if (!registerFramePnP(R_initial, P_initial, best_i, sfm_f)) {
            return false;
        }
        c_Rotation[best_i] = R_initial;
        c_Translation[best_i] = P_initial;
        c_Quat[best_i] = c_Rotation[best_i];
        Pose[best_i].block<3, 3>(0, 0) = c_Rotation[best_i];
        Pose[best_i].block<3, 1>(0, 3) = c_Translation[best_i];
        triangulateTwoFrames(l, Pose[l], best_i, Pose[best_i], sfm_f);
        triangulateTwoFrames(best_i, Pose[best_i], last, Pose[last], sfm_f);
        solved[best_i] = true;
    }

    for (int i = 0; i < frame_num; i++) {
        if (i == l || i == last) {
            continue;
        }
        q_cam_rel[i] = c_Quat[i].inverse();
        t_arr[i] = -1 * (c_Quat[i] * c_Translation[i]);
    }

    {
        ceres::Problem problem;
        ceres::Manifold* quat_manifold = new ceres::QuaternionManifold();
        for (int i = 0; i < frame_num; i++) {
            c_translation[i][0] = c_Translation[i].x();
            c_translation[i][1] = c_Translation[i].y();
            c_translation[i][2] = c_Translation[i].z();
            c_rotation[i][0] = c_Quat[i].w();
            c_rotation[i][1] = c_Quat[i].x();
            c_rotation[i][2] = c_Quat[i].y();
            c_rotation[i][3] = c_Quat[i].z();
            problem.AddParameterBlock(c_rotation[i].data(), 4, quat_manifold);
            problem.AddParameterBlock(c_translation[i].data(), 3);
            if (i == l) {
                problem.SetParameterBlockConstant(c_rotation[i].data());
            }
            // seed 平移固定世界原点，基线帧平移固定为视觉估计的单位尺度。
            if (i == l || i == last) {
                problem.SetParameterBlockConstant(c_translation[i].data());
            }
        }

        int observation_count = 0;
        for (int i = 0; i < feature_num_; i++) {
            if (!sfm_f[i].state) {
                continue;
            }
            problem.AddParameterBlock(sfm_f[i].position, 3);
            for (const auto& [frame_index, observation] : sfm_f[i].observation) {
                problem.AddResidualBlock(
                    SfmReprojectionFactor::Create(observation),
                    new ceres::HuberLoss(epipolar_threshold_), c_rotation[frame_index].data(),
                    c_translation[frame_index].data(), sfm_f[i].position);
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
            c_Quat[i] = Eigen::Quaterniond(
                            c_rotation[i][0], c_rotation[i][1], c_rotation[i][2], c_rotation[i][3])
                            .normalized();
            c_Rotation[i] = c_Quat[i].toRotationMatrix();
            c_Translation[i] =
                Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]);
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
            q_cam_rel[i] = c_Quat[i].inverse();
            t_arr[i] = -1 * (q_cam_rel[i] * c_Translation[i]);
        }

        for (int i = 0; i < feature_num_; i++) {
            if (!sfm_f[i].state) {
                continue;
            }
            const Eigen::Vector3d point_3d(sfm_f[i].position);
            bool ok = true;
            for (const auto& [frame_index, observation] : sfm_f[i].observation) {
                (void)observation;
                const Eigen::Vector3d point_camera =
                    c_Rotation[frame_index] * point_3d + c_Translation[frame_index];
                if (!point_camera.allFinite() || point_camera.z() < 0.1) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                sfm_f[i].state = false;
                continue;
            }
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

bool InitialSFM::registerFramePnP(
    Eigen::Matrix3d& R_initial, Eigen::Vector3d& P_initial, int i, std::vector<SFMFeature>& sfm_f) {
    std::vector<Eigen::Vector2d> observations;
    std::vector<Eigen::Vector3d> world_points;
    for (int j = 0; j < feature_num_; j++) {
        if (sfm_f[j].state != true) {
            continue;
        }
        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++) {
            if (sfm_f[j].observation[k].first == i) {
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
                observations.push_back(img_pts);
                world_points.emplace_back(
                    sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
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
    cv::cv2eigen(rotation_matrix, R_initial);
    for (int d = 0; d < 3; ++d) {
        P_initial[d] = translation_vector.at<double>(d);
    }
    if (!R_initial.allFinite() || !P_initial.allFinite()) {
        spdlog::warn("SFM PnP failed: frame={}, reason=non_finite_solution", i);
        return false;
    }

    int pnp_bad = 0;
    double pnp_inlier_error = 0.0;
    for (int k = 0; k < n_pts; ++k) {
        const Eigen::Vector3d point_camera = R_initial * world_points[k] + P_initial;
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

void InitialSFM::triangulateTwoFrames(
    int frame0, Eigen::Matrix<double, 3, 4>& Pose0, int frame1, Eigen::Matrix<double, 3, 4>& Pose1,
    std::vector<SFMFeature>& sfm_f, const std::unordered_set<int>* allowed_feature_ids) {
    TASSEL_ASSERT(frame0 != frame1);
    for (int j = 0; j < feature_num_; j++) {
        if (sfm_f[j].state == true) {
            continue;
        }
        // 空过滤器用于已注册帧扩展地图点；初始基线只使用确定性对极筛选内点。
        if (allowed_feature_ids != nullptr && !allowed_feature_ids->contains(sfm_f[j].id)) {
            continue;
        }
        bool has_0 = false, has_1 = false;
        Eigen::Vector2d point0;
        Eigen::Vector2d point1;
        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++) {
            if (sfm_f[j].observation[k].first == frame0) {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1) {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }
        if (has_0 && has_1) {
            Eigen::Vector3d point_3d;
            point_3d = tassel_utils::dehomogenize(
                tassel_utils::triangulateTwoView(Pose0, point0, Pose1, point1));
            if (!point_3d.allFinite()) {
                continue;
            }
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}

void InitialSFM::decomposeEssentialMat(
    const Eigen::Matrix3d& essential, std::vector<PoseCandidate>& candidates) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d v = svd.matrixV();
    if (u.determinant() < 0.0) {
        u.col(2) *= -1.0;
    }
    if (v.determinant() < 0.0) {
        v.col(2) *= -1.0;
    }

    Eigen::Matrix3d w;
    w << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d r1 = u * w * v.transpose();
    Eigen::Matrix3d r2 = u * w.transpose() * v.transpose();
    if (r1.determinant() < 0.0) {
        r1 = -r1;
    }
    if (r2.determinant() < 0.0) {
        r2 = -r2;
    }
    const Eigen::Vector3d t = u.col(2).normalized();
    candidates = {{r1, t}, {r1, -t}, {r2, t}, {r2, -t}};
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

    std::sort(scored.begin(), scored.end(), [](const PoseCandidate& a, const PoseCandidate& b) {
        return a.score > b.score;
    });
}

bool InitialSFM::construct(
    State& cur_state, FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
    std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out,
    int first_frame_index) {
    if (first_frame_index < 0 || first_frame_index > cur_state.latest_active_frame_index) {
        spdlog::error(
            "SFM input failed: first_frame={}, latest_frame={}", first_frame_index,
            cur_state.latest_active_frame_index);
        return false;
    }
    int frame_num = cur_state.latest_active_frame_index - first_frame_index + 1;
    if (frame_num < 2) {
        spdlog::error("SFM input failed: frames={}, required=2", frame_num);
        return false;
    }

    auto sfm_f = feature_manager.collectSFMFeatures(cur_state, first_frame_index);

    int seed_id = selectSeedFrame(frame_num, sfm_f);
    if (seed_id < 0) {
        return false;
    }

    auto other_candidates = findParallaxFrames(seed_id, frame_num, sfm_f);
    if (other_candidates.empty() || other_candidates[0].second < min_points_) {
        return false;
    }

    std::vector<Eigen::Quaterniond> q_cam_i0(frame_num);
    for (int i = 0; i < frame_num; i++) {
        q_cam_i0[i] =
            Eigen::Quaterniond(cur_state.frames[first_frame_index + i].rot_w_i * ric).normalized();
    }

    for (const auto& [other_id, common] : other_candidates) {
        std::vector<PoseCandidate> candidates;
        std::vector<cv::Point2f> pts_seed, pts_other;
        std::unordered_set<int> inlier_feature_ids;
        if (!computeEssential(
                seed_id, other_id, sfm_f, candidates, pts_seed, pts_other, inlier_feature_ids)) {
            continue;
        }

        const Eigen::Matrix3d rotation_prior =
            (q_cam_i0[other_id].inverse() * q_cam_i0[seed_id]).toRotationMatrix();
        PoseCandidate selected;
        if (!resolvePose(candidates, pts_seed, pts_other, rotation_prior, selected)) {
            continue;
        }

        Eigen::Matrix3d R_sel = selected.R;
        Eigen::Vector3d t_sel = selected.t;

        Eigen::Quaterniond q_cam_seed = q_cam_i0[seed_id];
        std::vector<Eigen::Quaterniond> q_cam_rel(frame_num);
        for (int i = 0; i < frame_num; i++) {
            q_cam_rel[i] = q_cam_seed.inverse() * q_cam_i0[i];
        }

        Eigen::Vector3d T_dir = (-R_sel.transpose() * t_sel).normalized();

        std::vector<Eigen::Vector3d> t_arr(frame_num, Eigen::Vector3d::Zero());
        // reconstructScene 会原地写入三角化状态；失败候选不得污染后续候选。
        auto candidate_features = sfm_f;
        if (!reconstructScene(
                frame_num, seed_id, other_id, T_dir, q_cam_rel, t_arr, inlier_feature_ids,
                candidate_features)) {
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

        const int point_count = std::count_if(
            candidate_features.begin(), candidate_features.end(),
            [](const SFMFeature& feature) { return feature.state; });
        spdlog::info(
            "SFM: {} frames, {} pts, seed={}, other={}", frame_num, point_count, seed_id, other_id);
        return true;
    }

    spdlog::warn("SFM failed: all candidates were rejected");
    return false;
}

}  // namespace tassel_core
