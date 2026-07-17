#include "initial/initial_sfm.h"

#include <spdlog/spdlog.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

#include "tassel_utils/macros.h"
#include "tassel_utils/rotation.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {
namespace {

struct EpipolarSampsonFactor {
    EpipolarSampsonFactor(const Eigen::Vector2d& uv_i, const Eigen::Vector2d& uv_j)
        : xi_(uv_i.x(), uv_i.y(), 1.0), xj_(uv_j.x(), uv_j.y(), 1.0) {}

    template <typename T>
    bool operator()(
        const T* const q_i, const T* const t_i, const T* const q_j, const T* const t_j,
        T* residual) const {
        T xi[3] = {T(xi_.x()), T(xi_.y()), T(1.0)};
        T xj[3] = {T(xj_.x()), T(xj_.y()), T(1.0)};

        T q_i_inv[4] = {q_i[0], -q_i[1], -q_i[2], -q_i[3]};

        T x_world_dir[3];
        ceres::QuaternionRotatePoint(q_i_inv, xi, x_world_dir);
        T rji_xi[3];
        ceres::QuaternionRotatePoint(q_j, x_world_dir, rji_xi);

        T ti_world[3];
        ceres::QuaternionRotatePoint(q_i_inv, t_i, ti_world);
        T rji_ti[3];
        ceres::QuaternionRotatePoint(q_j, ti_world, rji_ti);
        T tji[3] = {t_j[0] - rji_ti[0], t_j[1] - rji_ti[1], t_j[2] - rji_ti[2]};

        T ex[3] = {
            tji[1] * rji_xi[2] - tji[2] * rji_xi[1], tji[2] * rji_xi[0] - tji[0] * rji_xi[2],
            tji[0] * rji_xi[1] - tji[1] * rji_xi[0]};
        T algebraic = xj[0] * ex[0] + xj[1] * ex[1] + xj[2] * ex[2];

        T xj_cross_t[3] = {
            xj[1] * tji[2] - xj[2] * tji[1], xj[2] * tji[0] - xj[0] * tji[2],
            xj[0] * tji[1] - xj[1] * tji[0]};
        T q_j_inv[4] = {q_j[0], -q_j[1], -q_j[2], -q_j[3]};
        T et_xj_world[3];
        ceres::QuaternionRotatePoint(q_j_inv, xj_cross_t, et_xj_world);
        T et_xj[3];
        ceres::QuaternionRotatePoint(q_i, et_xj_world, et_xj);

        T denom = ceres::sqrt(
            ex[0] * ex[0] + ex[1] * ex[1] + et_xj[0] * et_xj[0] + et_xj[1] * et_xj[1] + T(1e-12));
        residual[0] = algebraic / denom;
        return true;
    }

    Eigen::Vector3d xi_;
    Eigen::Vector3d xj_;
};

}  // namespace

int InitialSFM::selectSeedFrame(int frame_num, const std::vector<SFMFeature>& sfm_f) {
    std::vector<int> feature_count(frame_num, 0);
    std::vector<int> connectivity_per_frame(frame_num, 0);
    for (const auto& feature : sfm_f) {
        for (const auto& [frame_id, _] : feature.observation) {
            if (frame_id < 0 || frame_id >= frame_num) continue;
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
        if (feature_count[i] < min_seed_pts_) continue;
        if (connectivity_per_frame[i] > best_score) {
            best_score = connectivity_per_frame[i];
            seed_id = i;
        }
    }

    if (seed_id < 0) {
        spdlog::warn("SFM: no frame has enough tracked features for a seed");
        return -1;
    }

    spdlog::info(
        "SFM mono seed frame {}: features={}, connectivity={}", seed_id, feature_count[seed_id],
        best_score);
    return seed_id;
}

std::vector<std::pair<int, int>> InitialSFM::findParallaxFrames(
    int seed_id, int frame_num, const std::vector<SFMFeature>& sfm_f) {
    std::vector<std::pair<int, int>> other_candidates;
    for (int i = 0; i < frame_num; ++i) {
        if (i == seed_id) continue;
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
            if (!in_seed || !in_other) continue;
            ++common;
            parallaxes.push_back((uv_seed - uv_other).norm());
        }
        if (common < min_seed_pts_) continue;
        const size_t mid = parallaxes.size() / 2;
        std::nth_element(parallaxes.begin(), parallaxes.begin() + mid, parallaxes.end());
        if (parallaxes[mid] <= e_ransac_threshold_) continue;
        other_candidates.emplace_back(i, common);
    }
    std::sort(other_candidates.begin(), other_candidates.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    if (other_candidates.size() > 2) other_candidates.resize(2);
    if (other_candidates.empty() || other_candidates[0].second < min_seed_pts_) {
        spdlog::warn(
            "SFM: no suitable parallax frame (best common={})",
            other_candidates.empty() ? 0 : other_candidates[0].second);
    }
    return other_candidates;
}

bool InitialSFM::computeEssential(
    int seed_id, int other_id, const std::vector<SFMFeature>& sfm_f,
    std::vector<PoseCandidate>& candidates, std::vector<cv::Point2f>& pts_seed,
    std::vector<cv::Point2f>& pts_other) {
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
        }
    }
    if (static_cast<int>(pts_seed.size()) < std::max(5, min_e_inliers_)) return false;

    cv::Mat K_cv = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        pts_seed, pts_other, K_cv, cv::RANSAC, 0.99, e_ransac_threshold_, inlier_mask);
    if (cv::countNonZero(inlier_mask) < min_e_inliers_) return false;

    Eigen::Matrix3d E_eigen;
    cv::cv2eigen(E, E_eigen);
    decomposeEssentialMat(E_eigen, candidates);
    return true;
}

bool InitialSFM::resolvePose(
    const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
    const std::vector<cv::Point2f>& pts_other, PoseCandidate& selected) {
    std::vector<PoseCandidate> scored;
    scoreByCheirality(candidates, pts_seed, pts_other, scored);

    if (scored.empty() || scored.front().score < 5) return false;
    selected = scored.front();
    return true;
}

bool InitialSFM::runBA(
    int frame_num, int seed_id, int other_id, const Eigen::Vector3d& relative_T,
    std::vector<Eigen::Quaterniond>& q_cam_rel, std::vector<Eigen::Vector3d>& t_arr,
    std::vector<SFMFeature>& sfm_f, std::map<int, Eigen::Vector3d>& tracked_pts) {
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

    triangulateTwoFrames(l, Pose[l], last, Pose[last], sfm_f);

    for (int j = 0; j < feature_num_; j++) {
        if (!sfm_f[j].state) continue;
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
            if (solved[i]) continue;
            for (int r = 0; r < frame_num; r++) {
                if (!solved[r]) continue;
                int d = std::abs(i - r);
                if (d < best_dist) {
                    best_dist = d;
                    best_i = i;
                    best_ref = r;
                }
            }
        }
        if (best_i < 0) break;

        Eigen::Matrix3d R_initial = c_Rotation[best_ref];
        Eigen::Vector3d P_initial = c_Translation[best_ref];
        if (!registerFramePnP(R_initial, P_initial, best_i, sfm_f)) return false;
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
        if (i == l || i == last) continue;
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
            if (i == l || i == last) {
                problem.SetParameterBlockConstant(c_rotation[i].data());
                problem.SetParameterBlockConstant(c_translation[i].data());
            }
        }

        int n_epipolar_edges = 0;
        for (int i = 0; i < feature_num_; i++) {
            const int nobs = static_cast<int>(sfm_f[i].observation.size());
            if (nobs < 2) continue;
            for (int j = 0; j < nobs; j++) {
                int frame_j = sfm_f[i].observation[j].first;
                for (int k = j + 1; k < nobs; k++) {
                    int frame_k = sfm_f[i].observation[k].first;
                    if (frame_j == frame_k) continue;
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<EpipolarSampsonFactor, 1, 4, 3, 4, 3>(
                            new EpipolarSampsonFactor(
                                sfm_f[i].observation[j].second, sfm_f[i].observation[k].second));
                    ceres::LossFunction* loss = new ceres::HuberLoss(e_ransac_threshold_);
                    problem.AddResidualBlock(
                        cost_function, loss, c_rotation[frame_j].data(),
                        c_translation[frame_j].data(), c_rotation[frame_k].data(),
                        c_translation[frame_k].data());
                    n_epipolar_edges++;
                }
            }
        }

        if (n_epipolar_edges == 0) {
            spdlog::warn("SFM epipolar optimization: no valid edges");
            return false;
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = ba_max_iterations_;
        options.num_threads = ba_num_threads_;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        spdlog::debug(
            "SFM epipolar: {} edges, {}, iters={}, cost={:.4e}", n_epipolar_edges,
            summary.termination_type == ceres::CONVERGENCE ? "CONV" : "NOCONV",
            summary.iterations.size(), summary.final_cost);
        if (!summary.IsSolutionUsable()) return false;

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

        for (int i = 0; i < feature_num_; i++) sfm_f[i].state = false;
        for (int i = 0; i < feature_num_; i++) {
            int nobs = static_cast<int>(sfm_f[i].observation.size());
            if (nobs < 3) continue;
            std::vector<Eigen::Matrix<double, 3, 4>> obs_poses;
            std::vector<Eigen::Vector2d> obs_uvs;
            for (int j = 0; j < nobs; j++) {
                int frame_idx = sfm_f[i].observation[j].first;
                obs_poses.push_back(Pose[frame_idx]);
                obs_uvs.push_back(sfm_f[i].observation[j].second);
            }
            Eigen::Vector3d point_3d =
                tassel_utils::dehomogenize(tassel_utils::triangulateMultiView(obs_poses, obs_uvs));
            if (!point_3d.allFinite()) continue;
            bool ok = true;
            for (int j = 0; j < nobs; j++) {
                int frame_idx = sfm_f[i].observation[j].first;
                if ((point_3d - t_arr[frame_idx])
                        .dot(q_cam_rel[frame_idx].toRotationMatrix().col(2)) < 0.1) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            sfm_f[i].state = true;
            sfm_f[i].position[0] = point_3d(0);
            sfm_f[i].position[1] = point_3d(1);
            sfm_f[i].position[2] = point_3d(2);
        }
    }

    for (int i = 0; i < static_cast<int>(sfm_f.size()); i++) {
        if (sfm_f[i].state)
            tracked_pts[sfm_f[i].id] =
                Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
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
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < feature_num_; j++) {
        if (sfm_f[j].state != true) continue;
        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++) {
            if (sfm_f[j].observation[k].first == i) {
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
                pts_2_vector.emplace_back(img_pts(0), img_pts(1));
                pts_3_vector.emplace_back(
                    sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                break;
            }
        }
    }
    int n_pts = static_cast<int>(pts_2_vector.size());
    if (n_pts < min_pnp_pts_) {
        spdlog::warn("SFM PnP frame {}: only {} pts", i, n_pts);
        return false;
    }

    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true)) {
        spdlog::warn("SFM PnP frame {}: solvePnP failed", i);
        return false;
    }

    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp, T_pnp;
    cv::cv2eigen(r, R_pnp);
    cv::cv2eigen(t, T_pnp);

    int bad = 0;
    double sum_err = 0;
    for (int k = 0; k < n_pts; ++k) {
        Eigen::Vector3d p3e(pts_3_vector[k].x, pts_3_vector[k].y, pts_3_vector[k].z);
        Eigen::Vector3d p3_proj = R_pnp * p3e + T_pnp;
        if (!p3_proj.allFinite() || p3_proj.z() <= 1e-12) {
            ++bad;
            sum_err += pnp_reproj_threshold_ * 2.0;
            continue;
        }
        double u = p3_proj(0) / p3_proj(2);
        double v = p3_proj(1) / p3_proj(2);
        double err = std::sqrt(
            (u - pts_2_vector[k].x) * (u - pts_2_vector[k].x) +
            (v - pts_2_vector[k].y) * (v - pts_2_vector[k].y));
        sum_err += err;
        if (err > pnp_reproj_threshold_) ++bad;
    }
    if (sum_err / n_pts > pnp_reproj_threshold_ || bad > n_pts * max_bad_pnp_ratio_) {
        spdlog::warn("SFM PnP frame {}: high reproj err mean={:.4f}", i, sum_err / n_pts);
        return false;
    }

    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

void InitialSFM::triangulateTwoFrames(
    int frame0, Eigen::Matrix<double, 3, 4>& Pose0, int frame1, Eigen::Matrix<double, 3, 4>& Pose1,
    std::vector<SFMFeature>& sfm_f) {
    TASSEL_ASSERT(frame0 != frame1);
    for (int j = 0; j < feature_num_; j++) {
        if (sfm_f[j].state == true) continue;
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
            if (!point_3d.allFinite()) continue;
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}

void InitialSFM::decomposeEssentialMat(
    const Eigen::Matrix3d& E, std::vector<PoseCandidate>& candidates) {
    candidates.clear();
    candidates.reserve(4);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() < 0) U.col(2) *= -1;
    if (V.determinant() < 0) V.col(2) *= -1;

    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Eigen::Matrix3d R1 = U * W * V.transpose();
    if (R1.determinant() < 0) R1 = -R1;

    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
    if (R2.determinant() < 0) R2 = -R2;

    Eigen::Vector3d t = U.col(2).normalized();

    candidates.push_back({R1, t});
    candidates.push_back({R1, -t});
    candidates.push_back({R2, t});
    candidates.push_back({R2, -t});
}

void InitialSFM::scoreByCheirality(
    const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
    const std::vector<cv::Point2f>& pts_other, std::vector<PoseCandidate>& scored) {
    int npts = static_cast<int>(pts_seed.size());
    scored = candidates;
    if (npts < 5) return;

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
            if (std::abs(pt(3)) < 1e-10) continue;
            Eigen::Vector3d X = tassel_utils::dehomogenize(pt);
            if (X.z() <= 0.1) continue;
            Eigen::Vector3d X2 = cand.R * X + cand.t;
            if (X2.z() <= 0.1) continue;
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
    std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out) {
    int frame_num = cur_state.cur_frame_count + 1;
    if (frame_num < 2) return false;

    auto sfm_f = feature_manager.collectSFMFeatures(cur_state);

    int seed_id = selectSeedFrame(frame_num, sfm_f);
    if (seed_id < 0) return false;

    auto other_candidates = findParallaxFrames(seed_id, frame_num, sfm_f);
    if (other_candidates.empty() || other_candidates[0].second < min_seed_pts_) return false;

    std::vector<Eigen::Quaterniond> q_cam_i0(frame_num);
    for (int i = 0; i < frame_num; i++) {
        q_cam_i0[i] = Eigen::Quaterniond(cur_state.Rs[i] * ric).normalized();
    }

    for (const auto& [other_id, common] : other_candidates) {
        std::vector<PoseCandidate> candidates;
        std::vector<cv::Point2f> pts_seed, pts_other;
        if (!computeEssential(seed_id, other_id, sfm_f, candidates, pts_seed, pts_other)) continue;

        PoseCandidate selected;
        if (!resolvePose(candidates, pts_seed, pts_other, selected)) continue;

        Eigen::Matrix3d R_sel = selected.R;
        Eigen::Vector3d t_sel = selected.t;

        Eigen::Quaterniond q_cam_seed = q_cam_i0[seed_id];
        std::vector<Eigen::Quaterniond> q_cam_rel(frame_num);
        for (int i = 0; i < frame_num; i++) q_cam_rel[i] = q_cam_seed.inverse() * q_cam_i0[i];

        q_cam_rel[other_id] = Eigen::Quaterniond(tassel_utils::normalizeRot(R_sel.transpose()));
        Eigen::Vector3d T_dir = (-R_sel.transpose() * t_sel).normalized();

        std::vector<Eigen::Vector3d> t_arr(frame_num, Eigen::Vector3d::Zero());
        std::map<int, Eigen::Vector3d> tracked_pts;
        if (!runBA(frame_num, seed_id, other_id, T_dir, q_cam_rel, t_arr, sfm_f, tracked_pts))
            continue;

        Rs_out.resize(frame_num);
        Ps_out.resize(frame_num);
        for (int i = 0; i < frame_num; i++) {
            Rs_out[i] = q_cam_rel[i].toRotationMatrix();
            Ps_out[i] = t_arr[i];
        }

        alignToReference(frame_num, Rs_out, Ps_out);

        spdlog::info(
            "SFM: {} frames, {} pts, seed={}, other={}", frame_num, tracked_pts.size(), seed_id,
            other_id);
        return true;
    }

    spdlog::warn("SFM: all candidates failed");
    return false;
}

}  // namespace tassel_core
