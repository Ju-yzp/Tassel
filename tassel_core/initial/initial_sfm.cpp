#include "initial/initial_sfm.h"

#include <cassert>

#include <spdlog/spdlog.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

#include <sophus/so3.hpp>

#include "factor/reprojection_factor.h"
#include "tassel_utils/rotation.h"
#include "tassel_utils/triangulation.h"

namespace tassel_core {

void InitialSFM::collectStereoDepths(
    int frame_num, const State& cur_state, FeatureManager& feature_manager,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1,
    const Eigen::Vector3d& tic1, std::vector<std::vector<Observation>>& all_frames) {
    for (auto& [id, feature] : feature_manager.features()) {
        int start_id = static_cast<int>(feature.start_frame_id);
        for (size_t k = 0; k < feature.observations.size(); ++k) {
            int abs_id = start_id + static_cast<int>(k);
            if (abs_id >= frame_num) continue;
            Observation obs;
            obs.feature_id = id;
            auto& obs_per = feature.observations[k];
            obs.is_stereo = obs_per.is_stereo;
            obs.uv_l = obs_per.uv;
            obs.uv_r = obs_per.uv_r;
            all_frames[abs_id].push_back(obs);
        }
    }

    R_lr_ = ric1.transpose() * ric;
    P_lr_ = ric1.transpose() * (tic - tic1);
    for (auto& frame_obs : all_frames) {
        for (auto& obs : frame_obs) {
            if (obs.is_stereo) obs.depth = triangulateStereo(obs.uv_l, obs.uv_r, R_lr_, P_lr_);
        }
    }
}

int InitialSFM::selectSeedFrame(
    int frame_num, const std::vector<std::vector<Observation>>& all_frames,
    const std::vector<SFMFeature>& /*sfm_f*/) {
    std::vector<int> valid_per_frame(frame_num, 0);
    for (int i = 0; i < frame_num; ++i) {
        for (const auto& obs : all_frames[i])
            if (!std::isnan(obs.depth)) ++valid_per_frame[i];
    }

    int seed_id = -1;
    int best_score = 0;
    for (int i = 0; i < frame_num; ++i) {
        if (valid_per_frame[i] < min_seed_pts_) continue;
        int score = 0;
        std::set<int> seed_fids;
        for (const auto& obs : all_frames[i])
            if (!std::isnan(obs.depth)) seed_fids.insert(obs.feature_id);
        for (int n = 1; n <= 3; ++n) {
            int weight = 4 - n;
            for (int delta : {-n, n}) {
                int nb = i + delta;
                if (nb < 0 || nb >= frame_num) continue;
                for (const auto& obs : all_frames[nb])
                    if (seed_fids.count(obs.feature_id)) score += weight;
            }
        }
        if (score > best_score) {
            best_score = score;
            seed_id = i;
        }
    }
    if (seed_id < 0) {
        for (int i = 0; i < frame_num; ++i)
            if (valid_per_frame[i] > valid_per_frame[seed_id]) seed_id = i;
    }
    spdlog::info(
        "SFM seed frame {}: {} depths, connectivity={}", seed_id, valid_per_frame[seed_id],
        best_score);
    if (valid_per_frame[seed_id] < min_seed_pts_) {
        spdlog::warn("SFM: insufficient stereo depths ({})", valid_per_frame[seed_id]);
        return -1;
    }
    return seed_id;
}

std::vector<std::pair<int, int>> InitialSFM::findParallaxFrames(
    int seed_id, int frame_num, const std::vector<std::vector<Observation>>& all_frames,
    const std::vector<SFMFeature>& sfm_f) {
    std::vector<std::pair<int, int>> other_candidates;
    for (int i = 0; i < frame_num; ++i) {
        if (i == seed_id) continue;
        int common = 0;
        for (const auto& f : sfm_f) {
            bool in_seed = false, in_other = false;
            for (const auto& [fid, uv] : f.observation) {
                if (fid == seed_id) in_seed = true;
                if (fid == i) in_other = true;
            }
            if (!in_seed || !in_other) continue;
            for (const auto& obs : all_frames[seed_id])
                if (obs.feature_id == f.id && !std::isnan(obs.depth)) {
                    ++common;
                    break;
                }
        }
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
    const std::vector<std::vector<Observation>>&, std::vector<PoseCandidate>& candidates,
    std::vector<cv::Point2f>& pts_seed, std::vector<cv::Point2f>& pts_other) {
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
    if (static_cast<int>(pts_seed.size()) < 10) return false;

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
    const std::vector<cv::Point2f>& pts_other, const std::vector<double>& seed_depths,
    PoseCandidate& selected) {
    int pnp_pick = selectByPnP(candidates, pts_seed, pts_other, seed_depths);

    std::vector<PoseCandidate> scored;
    scoreByCheirality(candidates, pts_seed, pts_other, scored);

    std::vector<int> priority;
    if (pnp_pick >= 0) {
        for (int i = 0; i < static_cast<int>(scored.size()); ++i)
            if (tassel_utils::rotDiff(scored[i].R, candidates[pnp_pick].R).norm() < 1e-6 &&
                scored[i].t.dot(candidates[pnp_pick].t) > 0.5) {
                priority.push_back(i);
                break;
            }
    }
    for (int i = 0; i < std::min(2, static_cast<int>(scored.size())); ++i) {
        bool dup = false;
        for (int p : priority) dup |= (p == i);
        if (!dup) priority.push_back(i);
    }

    for (int pi = 0; pi < static_cast<int>(priority.size()); ++pi) {
        int ci = priority[pi];
        if (scored[ci].score < 5) continue;
        selected = scored[ci];
        return true;
    }
    return false;
}

bool InitialSFM::checkCheirality(
    int seed_id, int other_id, const std::vector<Eigen::Quaterniond>& q_arr_rel,
    const Eigen::Vector3d& T_dir, const std::vector<SFMFeature>& sfm_f) {
    const Eigen::Quaterniond& q_other = q_arr_rel[other_id];
    std::vector<Eigen::Matrix<double, 3, 4>> P(2);
    P[0].block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    P[0].block<3, 1>(0, 3) = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rw2c_1 = q_other.inverse().toRotationMatrix();
    P[1].block<3, 3>(0, 0) = Rw2c_1;
    P[1].block<3, 1>(0, 3) = -Rw2c_1 * T_dir.normalized();

    int n_valid = 0;
    for (const auto& f : sfm_f) {
        bool h0 = false, h1 = false;
        Eigen::Vector2d uv0, uv1;
        for (const auto& [fid, uv] : f.observation) {
            if (fid == seed_id) {
                h0 = true;
                uv0 = uv;
            }
            if (fid == other_id) {
                h1 = true;
                uv1 = uv;
            }
        }
        if (!h0 || !h1) continue;
        Eigen::Vector4d pt = tassel_utils::triangulateTwoView(P[0], uv0, P[1], uv1);
        if (std::abs(pt(3)) < 1e-10) continue;
        Eigen::Vector3d X = tassel_utils::dehomogenize(pt);
        if (X.z() > 0.1 && (X - T_dir.normalized()).dot(q_other.toRotationMatrix().col(2)) > 0.1)
            n_valid++;
    }
    if (n_valid < 5) return false;
    return true;
}

bool InitialSFM::runBA(
    int frame_num, int seed_id, int other_id, const Eigen::Matrix3d& /*relative_R*/,
    const Eigen::Vector3d& relative_T, std::vector<Eigen::Quaterniond>& q_arr_rel,
    std::vector<Eigen::Vector3d>& t_arr, std::vector<SFMFeature>& sfm_f,
    std::map<int, Eigen::Vector3d>& tracked_pts) {
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

    c_Quat[l] = q_arr_rel[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * t_arr[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[last] = q_arr_rel[last].inverse();
    c_Rotation[last] = c_Quat[last].toRotationMatrix();
    c_Translation[last] = -1 * (c_Rotation[last] * t_arr[last]);
    Pose[last].block<3, 3>(0, 0) = c_Rotation[last];
    Pose[last].block<3, 1>(0, 3) = c_Translation[last];

    triangulateTwoFrames(l, Pose[l], last, Pose[last], sfm_f);

    for (int j = 0; j < feature_num_; j++) {
        if (!sfm_f[j].state) continue;
        Eigen::Vector3d X(sfm_f[j].position);
        if ((X - t_arr[l]).dot(q_arr_rel[l].toRotationMatrix().col(2)) <= 0.1 ||
            (X - t_arr[last]).dot(q_arr_rel[last].toRotationMatrix().col(2)) <= 0.1) {
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
        q_arr_rel[i] = c_Quat[i].inverse();
        t_arr[i] = -1 * (c_Quat[i] * c_Translation[i]);
    }

    for (int j = 0; j < feature_num_; j++) sfm_f[j].state = false;
    for (int j = 0; j < feature_num_; j++) {
        int nobs = static_cast<int>(sfm_f[j].observation.size());
        if (nobs < 3) continue;
        std::vector<Eigen::Matrix<double, 3, 4>> obs_poses;
        std::vector<Eigen::Vector2d> obs_uvs;
        for (int k = 0; k < nobs; k++) {
            int fid = sfm_f[j].observation[k].first;
            obs_poses.push_back(Pose[fid]);
            obs_uvs.push_back(sfm_f[j].observation[k].second);
        }
        Eigen::Vector3d point_3d =
            tassel_utils::dehomogenize(tassel_utils::triangulateMultiView(obs_poses, obs_uvs));
        bool ok = true;
        for (int k = 0; k < nobs; k++) {
            int fid = sfm_f[j].observation[k].first;
            if ((point_3d - t_arr[fid]).dot(q_arr_rel[fid].toRotationMatrix().col(2)) < 0.1) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        sfm_f[j].state = true;
        sfm_f[j].position[0] = point_3d(0);
        sfm_f[j].position[1] = point_3d(1);
        sfm_f[j].position[2] = point_3d(2);
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

        int n_ba_pts = 0;
        for (int i = 0; i < feature_num_; i++) {
            if (sfm_f[i].state != true) continue;
            if (static_cast<int>(sfm_f[i].observation.size()) < 3) continue;
            bool ok = true;
            Eigen::Vector3d X(sfm_f[i].position);
            for (int j = 0; j < static_cast<int>(sfm_f[i].observation.size()); j++) {
                int frame_idx = sfm_f[i].observation[j].first;
                if ((X - t_arr[frame_idx]).dot(q_arr_rel[frame_idx].toRotationMatrix().col(2)) <
                    0.1) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            for (int j = 0; j < static_cast<int>(sfm_f[i].observation.size()); j++) {
                int frame_idx = sfm_f[i].observation[j].first;
                ceres::CostFunction* cost_function = new ReprojectionFactor(
                    sfm_f[i].observation[j].second.x(), sfm_f[i].observation[j].second.y());
                problem.AddResidualBlock(
                    cost_function, nullptr, c_rotation[frame_idx].data(),
                    c_translation[frame_idx].data(), sfm_f[i].position);
            }
            n_ba_pts++;
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = ba_max_iterations_;
        options.num_threads = ba_num_threads_;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        spdlog::debug(
            "SFM BA: {} pts, {}, iters={}, cost={:.4e}", n_ba_pts,
            summary.termination_type == ceres::CONVERGENCE ? "CONV" : "NOCONV",
            summary.iterations.size(), summary.final_cost);

        for (int i = 0; i < frame_num; i++) {
            q_arr_rel[i].w() = c_rotation[i][0];
            q_arr_rel[i].x() = c_rotation[i][1];
            q_arr_rel[i].y() = c_rotation[i][2];
            q_arr_rel[i].z() = c_rotation[i][3];
            q_arr_rel[i] = q_arr_rel[i].inverse();
        }
        for (int i = 0; i < frame_num; i++) {
            t_arr[i] = -1 * (q_arr_rel[i] *
                             Eigen::Vector3d(
                                 c_translation[i][0], c_translation[i][1], c_translation[i][2]));
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
    assert(frame0 != frame1);
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
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}

double InitialSFM::triangulateStereo(
    const Eigen::Vector3d& uv_l, const Eigen::Vector3d& uv_r, const Eigen::Matrix3d& R_lr,
    const Eigen::Vector3d& P_lr) {
    Eigen::Matrix<double, 3, 4> pose0 = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> pose1;
    pose1.block<3, 3>(0, 0) = R_lr;
    pose1.block<3, 1>(0, 3) = P_lr;

    double cond;
    Eigen::Vector4d h =
        tassel_utils::triangulateTwoView(pose0, uv_l.head<2>(), pose1, uv_r.head<2>(), &cond);
    if (cond < 1e6) {
        Eigen::Vector3d p = tassel_utils::dehomogenize(h);
        if (p.z() > min_depth_ && p.z() < max_depth_) return p.z();
    }
    return std::numeric_limits<double>::quiet_NaN();
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

int InitialSFM::selectByPnP(
    const std::vector<PoseCandidate>& candidates, const std::vector<cv::Point2f>& pts_seed,
    const std::vector<cv::Point2f>& pts_other, const std::vector<double>& seed_depths) {
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    pts_3d.reserve(pts_seed.size());
    pts_2d.reserve(pts_seed.size());

    for (size_t i = 0; i < pts_seed.size(); ++i) {
        if (std::isnan(seed_depths[i])) continue;
        double d = seed_depths[i];
        pts_3d.emplace_back(pts_seed[i].x * d, pts_seed[i].y * d, d);
        pts_2d.push_back(pts_other[i]);
    }

    if (static_cast<int>(pts_3d.size()) < 7) return -1;

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat rvec, t_pnp_cv, inliers;
    if (!cv::solvePnPRansac(
            pts_3d, pts_2d, K, cv::Mat(), rvec, t_pnp_cv, false, 100, 0.01, 0.99, inliers))
        return -1;
    if (inliers.rows < std::max(7, static_cast<int>(pts_3d.size()) / 2)) return -1;

    cv::Mat R_pnp_cv;
    cv::Rodrigues(rvec, R_pnp_cv);
    Eigen::Matrix3d R_pnp;
    Eigen::Vector3d t_pnp_eig;
    cv::cv2eigen(R_pnp_cv, R_pnp);
    cv::cv2eigen(t_pnp_cv, t_pnp_eig);
    Eigen::Vector3d t_pnp_dir = t_pnp_eig.normalized();

    int best_idx = -1;
    double best_err = std::numeric_limits<double>::max();
    for (int i = 0; i < 4; ++i) {
        double r_err = tassel_utils::rotDiff(candidates[i].R, R_pnp).norm();
        double t_dot = candidates[i].t.dot(t_pnp_dir);
        if (t_dot < 0.5) continue;
        double err = r_err + (1.0 - t_dot);
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return best_idx;
}

bool InitialSFM::construct(
    State& cur_state, FeatureManager& feature_manager, const Eigen::Matrix3d& ric,
    const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1,
    std::vector<Eigen::Matrix3d>& Rs_out, std::vector<Eigen::Vector3d>& Ps_out) {
    int frame_num = cur_state.cur_frame_count + 1;
    if (frame_num < 2) return false;

    auto sfm_f = feature_manager.collectSFMFeatures(frame_num);

    std::vector<std::vector<Observation>> all_frames(frame_num);
    collectStereoDepths(frame_num, cur_state, feature_manager, ric, tic, ric1, tic1, all_frames);
    int seed_id = selectSeedFrame(frame_num, all_frames, sfm_f);
    if (seed_id < 0) return false;

    auto other_candidates = findParallaxFrames(seed_id, frame_num, all_frames, sfm_f);
    if (other_candidates.empty() || other_candidates[0].second < 10) return false;

    std::vector<Eigen::Quaterniond> q_imu_w(frame_num);
    for (int i = 0; i < frame_num; i++) {
        q_imu_w[i] = Eigen::Quaterniond(cur_state.Rs[i] * ric).normalized();
    }

    for (const auto& [other_id, common] : other_candidates) {
        std::vector<PoseCandidate> candidates;
        std::vector<cv::Point2f> pts_seed, pts_other;
        if (!computeEssential(
                seed_id, other_id, sfm_f, all_frames, candidates, pts_seed, pts_other))
            continue;

        std::vector<double> seed_depths(pts_seed.size(), std::numeric_limits<double>::quiet_NaN());
        for (size_t k = 0; k < pts_seed.size(); ++k) {
            for (const auto& f : sfm_f) {
                bool in_seed = false, in_other = false;
                Eigen::Vector2d uv_seed;
                for (const auto& [fid, uv] : f.observation) {
                    if (fid == seed_id) {
                        in_seed = true;
                        uv_seed = uv;
                    }
                    if (fid == other_id) in_other = true;
                }
                if (in_seed && in_other && std::abs(uv_seed.x() - pts_seed[k].x) < 1e-6 &&
                    std::abs(uv_seed.y() - pts_seed[k].y) < 1e-6) {
                    for (const auto& obs : all_frames[seed_id])
                        if (obs.feature_id == f.id) {
                            seed_depths[k] = obs.depth;
                            break;
                        }
                    break;
                }
            }
        }

        PoseCandidate selected;
        if (!resolvePose(candidates, pts_seed, pts_other, seed_depths, selected)) continue;

        Eigen::Matrix3d R_sel = selected.R;
        Eigen::Vector3d t_sel = selected.t;

        Eigen::Quaterniond q_seed_w = q_imu_w[seed_id];
        std::vector<Eigen::Quaterniond> q_arr_rel(frame_num);
        for (int i = 0; i < frame_num; i++) q_arr_rel[i] = q_seed_w.inverse() * q_imu_w[i];

        q_arr_rel[other_id] = Eigen::Quaterniond(tassel_utils::normalizeRot(R_sel.transpose()));
        Eigen::Vector3d T_dir = (-R_sel.transpose() * t_sel).normalized();

        if (!checkCheirality(seed_id, other_id, q_arr_rel, T_dir, sfm_f)) continue;

        std::vector<Eigen::Vector3d> t_arr(frame_num, Eigen::Vector3d::Zero());
        std::map<int, Eigen::Vector3d> tracked_pts;
        if (!runBA(
                frame_num, seed_id, other_id, Eigen::Matrix3d::Identity(), T_dir, q_arr_rel, t_arr,
                sfm_f, tracked_pts))
            continue;

        Rs_out.resize(frame_num);
        Ps_out.resize(frame_num);
        for (int i = 0; i < frame_num; i++) {
            Rs_out[i] = q_arr_rel[i].toRotationMatrix();
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
