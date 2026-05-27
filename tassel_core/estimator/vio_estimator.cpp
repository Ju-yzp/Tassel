#include "vio_estimator.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <fstream>
#include <set>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>

#include "factor/integrator_base.h"
#include "factor/landmark_block.h"
#include "factor/marg_helper.h"
#include "factor/marg_linearized_data.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/se3_right_manifold.h"
#include "factor/visual_factor.h"

namespace tassel_core {

VioEstimator::VioEstimator(
    const EstimatorOption& option, std::shared_ptr<State> state, std::shared_ptr<FeatureManager> fm,
    const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric1,
    const Eigen::Vector3d& tic1)
    : option_(option),
      state_(std::move(state)),
      feature_manager_(std::move(fm)),
      ric_(ric),
      tic_(tic),
      ric1_(ric1),
      tic1_(tic1),
      imu_initialized_(false),
      init_ts_(-1) {
    cv::setNumThreads(option_.num_threads);
    noise_ = initNoise();
    imu_manager_ = std::make_shared<IntegratorManager>();
    for (int i = 0; i < state_->max_frame_count; ++i) {
        imu_manager_->addIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
    }
    delay_time_estimator_ =
        std::make_unique<DelayTimeEstimator>(imu_manager_, feature_manager_, ric_, tic_);
    state_->cur_frame_count = 0;
    state_->ric = ric_;
    state_->tic = tic_;
}

void VioEstimator::processMeasurement(
    double ts, const std::unordered_map<int, FeaturePerFrame>& feature_frame,
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    if (td_estimated_) {
        return;
    }

    if (!imu_initialized_) {
        if (init_ts_ < 0) {
            init_ts_ = ts;
        }
        imu_init_buf_.insert(imu_init_buf_.end(), imu_measurements.begin(), imu_measurements.end());
        if (ts - init_ts_ >= option_.init_time_span) {
            initializeImu(imu_init_buf_);
        }
        return;
    }

    int& frame_count = state_->cur_frame_count;
    bool is_keyframe = feature_manager_->checkKeyFrameByParallax(frame_count, feature_frame);

    if (frame_count > 0) {
        Eigen::Matrix3d& R = state_->Rs[frame_count];
        Eigen::Vector3d& P = state_->Ps[frame_count];
        Eigen::Vector3d& V = state_->Vs[frame_count];

        auto* integrator = imu_manager_->getIntegrator(frame_count - 1);
        Eigen::Vector3d Ba = integrator->ba_linearized;
        Eigen::Vector3d Bg = integrator->bg_linearized;
        for (const auto& imu : imu_measurements) {
            double dt = imu.timestamp - last_ts_;
            Eigen::Vector3d acc_0 = R * (last_imu_acc_ - Ba) - tassel_utils::G;
            Eigen::Vector3d gyr = 0.5 * (last_imu_gyro_ + imu.gyro) - Bg;
            R = R * Sophus::SO3d::exp(gyr * dt).matrix();
            Eigen::Vector3d acc_1 = R * (imu.acc - Ba) - tassel_utils::G;
            Eigen::Vector3d acc = 0.5 * (acc_0 + acc_1);
            P += V * dt + 0.5 * acc * dt * dt;
            V += acc * dt;
            if (pose_callback_) pose_callback_(imu.timestamp, Sophus::SE3d(R, P));
            if (path_callback_) path_callback_(imu.timestamp, Sophus::SE3d(R, P));
            integrator->update(imu);
            last_ts_ = imu.timestamp;
            last_imu_gyro_ = imu.gyro;
            last_imu_acc_ = imu.acc;
        }

        Eigen::Quaterniond q(R);
        q.normalize();
        state_->Rs[frame_count] = q.matrix();
    }
    if (is_keyframe) {
        feature_manager_->triangulate(*state_, ric_, tic_, ric1_, tic1_);
        cam_timestamps_.push_back(ts);
        if (frame_count == state_->max_frame_count - 1) {
            if (!td_estimated_) {
                double td =
                    delay_time_estimator_->estimate(*state_, cam_timestamps_, state_->delay_time);
                state_->delay_time = td;
                td_estimated_ = true;
                spdlog::info("time delay estimated: {:.6f} s", td);
                dumpToFile("/tmp/vio_output");
            }
            slideWindow();
        } else {
            ++frame_count;
            state_->Rs[frame_count] = state_->Rs[frame_count - 1];
            state_->Ps[frame_count] = state_->Ps[frame_count - 1];
            state_->Vs[frame_count] = state_->Vs[frame_count - 1];
            state_->Bas[frame_count] = state_->Bas[frame_count - 1];
            state_->Bgs[frame_count] = state_->Bgs[frame_count - 1];

            auto* prev_integrator = imu_manager_->getIntegrator(frame_count - 1);
            imu_manager_->getIntegrator(frame_count)
                ->reset(prev_integrator->ba_linearized, prev_integrator->bg_linearized, noise_);
        }
    } else {
        feature_manager_->removeNewest(frame_count);
    }

    if (state_->cur_frame_count > 0) {
        int latest_idx = state_->cur_frame_count - 1;

        Sophus::SE3d latest_pose(state_->Rs[latest_idx], state_->Ps[latest_idx]);

        if (mono_cloud_callback_) {
            auto pts = feature_manager_->getMonoPointCloud(*state_, ric_, tic_);
            mono_cloud_callback_(ts, pts);
        }
        if (stereo_cloud_callback_) {
            auto pts = feature_manager_->getStereoPointCloud(*state_, ric_, tic_);
            stereo_cloud_callback_(ts, pts);
        }
    }
}

void VioEstimator::optimize() {
    state_->stateToParams();
    auto features = feature_manager_->collectOptimizedFeatures();

    ceres::Problem problem;

    // inverse depth for each feature
    std::vector<double> inv_depth_params(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double d = features[k]->estimated_depth;
        inv_depth_params[k] = (d > 0 && d < option_.max_depth) ? (1.0 / d) : 1.0;
        problem.AddParameterBlock(&inv_depth_params[k], 1);
        // problem.SetParameterLowerBound(&inv_depth_params[k], 0, 1.0 / option_.max_depth);
        // problem.SetParameterUpperBound(&inv_depth_params[k], 0, 1.0 / option_.min_depth);
    }

    for (int i = 0; i < state_->max_frame_count; ++i) {
        auto se3_manifold = new SE3RightManifold();
        problem.AddParameterBlock(state_->param_poses[i].data(), 6, se3_manifold);
    }

    problem.SetParameterBlockConstant(state_->param_poses[0].data());

    ceres::LossFunction* loss = new ceres::HuberLoss(0.005);
    std::set<int> involved_indices;
    for (size_t k = 0; k < features.size(); ++k) {
        Feature* f = features[k];
        int host_id = f->start_frame_id;
        involved_indices.insert(host_id);
        for (size_t obs_idx = 1; obs_idx < f->observations.size(); ++obs_idx) {
            int target_id = host_id + static_cast<int>(obs_idx);
            involved_indices.insert(target_id);
            auto* cost = new VisualFactor(
                f->observations[0].uv, f->observations[obs_idx].uv, Eigen::Matrix3d::Identity(),
                Eigen::Vector3d::Zero(), option_.min_depth);
            problem.AddResidualBlock(
                cost, loss, state_->param_poses[host_id].data(),
                state_->param_poses[target_id].data(), &inv_depth_params[k]);
        }
    }
    if (static_cast<int>(involved_indices.size()) == state_->max_frame_count) {
        spdlog::info(
            "Optimized features cover all {} frames in the sliding window",
            state_->max_frame_count);
    }

    // 添加边缘化先验
    if (marg_lin_data_ && marg_lin_data_->H.rows() > 0) {
        auto* prior =
            new MarginalizationPrior(marg_lin_data_->H, marg_lin_data_->b, marg_poses_linearized_);
        int num_kept = static_cast<int>(marg_poses_linearized_.size());
        std::vector<double*> param_blocks;
        for (int i = 0; i < num_kept; ++i) {
            param_blocks.push_back(state_->param_poses[i].data());
        }
        problem.AddResidualBlock(prior, nullptr, param_blocks);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.max_num_iterations = option_.num_iterations;
    opts.num_threads = option_.num_threads;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    // spdlog::info("Ceres: {}", summary.FullReport());

    state_->paramsToState();

    std::vector<double> depths;
    depths.reserve(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        double inv_d = inv_depth_params[k];
        double d = (inv_d > 1e-6) ? (1.0 / inv_d) : INVALID_DEPTH;
        features[k]->estimated_depth = d;
        if (d > 0) depths.push_back(d);
    }

    if (!depths.empty()) {
        std::sort(depths.begin(), depths.end());
        double sum = 0;
        for (double d : depths) sum += d;
        double mean = sum / depths.size();
        double median = depths[depths.size() / 2];
        spdlog::info(
            "depth stats | count: {} min: {:.2f} max: {:.2f} mean: {:.2f} median: {:.2f} "
            "p10: {:.2f} p90: {:.2f}",
            depths.size(), depths.front(), depths.back(), mean, median, depths[depths.size() / 10],
            depths[depths.size() * 9 / 10]);
    }
}

void VioEstimator::marginalize() {
    int num_frames = state_->max_frame_count;

    // 收集当前窗口所有帧的线性化点
    std::vector<std::array<double, 6>> poses_linearized;
    for (int i = 0; i < num_frames; ++i) {
        poses_linearized.push_back(state_->param_poses[i]);
    }

    // 收集需要边缘化的特征
    auto marg_features = feature_manager_->collectMarginalizationFeatures();

    int marg_size = 6;
    int keep_size = (num_frames - 1) * 6;

    ceres::LossFunction* loss = new ceres::HuberLoss(0.005);
    std::vector<LandmarkBlock> blocks;
    int total_new_rows = 0;
    for (const auto& feature : marg_features) {
        int num_obs = static_cast<int>(feature.observations.size()) - 1;
        LandmarkBlock lb(option_.min_depth, loss);
        lb.allocate(num_frames, num_obs);
        lb.linearize(
            feature, poses_linearized, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        lb.performQR();
        total_new_rows += lb.keptRows();
        blocks.push_back(std::move(lb));
    }

    int prev_rows = (marg_lin_data_ && marg_lin_data_->H.rows() > 0) ? marg_lin_data_->H.rows() : 0;
    int all_rows = total_new_rows + prev_rows;

    Eigen::MatrixXd Q2Jp(all_rows, marg_size + keep_size);
    Eigen::VectorXd Q2r(all_rows);
    Q2Jp.setZero();
    Q2r.setZero();

    int offset = 0;
    for (const auto& lb : blocks) {
        lb.get_dense_Q2Jp_Q2r(Q2Jp, Q2r, offset);
        offset += lb.keptRows();
    }

    if (prev_rows > 0) {
        Q2Jp.block(total_new_rows, 0, prev_rows, marg_lin_data_->H.cols()) = marg_lin_data_->H;
        Q2r.segment(total_new_rows, prev_rows) = marg_lin_data_->b;
    }

    Eigen::MatrixXd H_new;
    Eigen::VectorXd b_new;
    MargHelper::marginalizeSqrtToSqrt(marg_size, keep_size, Q2Jp, Q2r, H_new, b_new);

    if (!marg_lin_data_) {
        marg_lin_data_ = std::make_unique<MargLinData>();
    }
    marg_lin_data_->H = H_new;
    marg_lin_data_->b = b_new;

    marg_poses_linearized_.clear();
    for (int i = 1; i < num_frames; ++i) {
        marg_poses_linearized_.push_back(poses_linearized[i]);
    }

    feature_manager_->removeMarginalizedFeatures();
    delete loss;
}

void VioEstimator::initializeImu(
    const std::vector<tassel_utils::IMUMeasurement>& imu_measurements) {
    Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d avg_gyro = Eigen::Vector3d::Zero();
    for (const auto& m : imu_measurements) {
        avg_acc += m.acc;
        avg_gyro += m.gyro;
    }
    avg_acc /= static_cast<double>(imu_measurements.size());
    avg_gyro /= static_cast<double>(imu_measurements.size());

    Eigen::Vector3d g_imu = avg_acc.normalized();
    Eigen::Vector3d g_world = tassel_utils::G.normalized();

    Eigen::Matrix3d R_w_i = Eigen::Quaterniond::FromTwoVectors(g_imu, g_world).toRotationMatrix();
    Eigen::Matrix3d R_w_c0 = R_w_i;
    Eigen::Quaterniond q = Eigen::Quaterniond(R_w_c0);
    q.normalize();
    state_->Rs[0] = q.matrix();
    state_->Ps[0] = Eigen::Vector3d::Zero();
    state_->Vs[0] = Eigen::Vector3d::Zero();
    state_->Bas[0] = avg_acc - R_w_i.transpose() * tassel_utils::G;
    state_->Bgs[0] = avg_gyro;
    imu_manager_->getIntegrator(0)->ba_linearized = state_->Bas[0];
    imu_manager_->getIntegrator(0)->bg_linearized = state_->Bgs[0];
    cam_timestamps_.push_back(init_ts_);
    last_ts_ = imu_measurements.back().timestamp;
    last_imu_acc_ = imu_measurements.back().acc;
    last_imu_gyro_ = imu_measurements.back().gyro;
    imu_initialized_ = true;
    spdlog::info("IMU gravity initialized");
}

void VioEstimator::slideWindow() {
    const int n = state_->max_frame_count;
    for (int i = 0; i < n - 1; ++i) {
        state_->Rs[i] = state_->Rs[i + 1];
        state_->Ps[i] = state_->Ps[i + 1];
        state_->Vs[i] = state_->Vs[i + 1];
        state_->Bas[i] = state_->Bas[i + 1];
        state_->Bgs[i] = state_->Bgs[i + 1];
    }
    Eigen::Quaterniond q(state_->Rs[n - 2]);
    q.normalize();
    state_->Rs[n - 2] = q.matrix();
    cam_timestamps_.erase(cam_timestamps_.begin());
    imu_manager_->removeOldest();
    imu_manager_->addIntegrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise_);
}

void VioEstimator::dumpToFile(const std::string& path) const {
    // 状态: 每帧位姿/速度/偏置
    {
        std::ofstream ofs(path + "_state.txt");
        if (!ofs) {
            spdlog::error("dumpToFile: cannot open {}", path + "_state.txt");
            return;
        }
        ofs << "# ts R(9) P(3) V(3) Ba(3) Bg(3)\n";
        int n = std::min(static_cast<int>(cam_timestamps_.size()), state_->max_frame_count);
        for (int i = 0; i < n; ++i) {
            ofs << std::fixed << cam_timestamps_[i];
            const auto& R = state_->Rs[i];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    ofs << ' ' << R(r, c);
                }
            }
            const auto& P = state_->Ps[i];
            ofs << ' ' << P.x() << ' ' << P.y() << ' ' << P.z();
            const auto& V = state_->Vs[i];
            ofs << ' ' << V.x() << ' ' << V.y() << ' ' << V.z();
            const auto& Ba = state_->Bas[i];
            ofs << ' ' << Ba.x() << ' ' << Ba.y() << ' ' << Ba.z();
            const auto& Bg = state_->Bgs[i];
            ofs << ' ' << Bg.x() << ' ' << Bg.y() << ' ' << Bg.z();
            ofs << '\n';
        }
        spdlog::info("dumped {} frames to {}", n, path + "_state.txt");
    }

    // 特征点: 每帧观测
    {
        std::ofstream ofs(path + "_features.txt");
        if (!ofs) {
            spdlog::error("dumpToFile: cannot open {}", path + "_features.txt");
            return;
        }
        ofs << "# feat_id start_frame depth tri_src num_obs\n";
        ofs << "#   frame_id u v [u_r v_r is_stereo]\n";
        const auto& features = feature_manager_->testFeatures();
        for (const auto& [feat_id, f] : features) {
            ofs << feat_id << ' ' << f.start_frame_id << ' ' << f.estimated_depth << ' '
                << static_cast<int>(f.tri_source) << ' ' << f.observations.size() << '\n';
            for (size_t k = 0; k < f.observations.size(); ++k) {
                int frame_id = static_cast<int>(f.start_frame_id) + static_cast<int>(k);
                const auto& obs = f.observations[k];
                ofs << "  " << frame_id << ' ' << obs.uv.x() << ' ' << obs.uv.y() << ' '
                    << obs.uv_r.x() << ' ' << obs.uv_r.y() << ' ' << static_cast<int>(obs.is_stereo)
                    << '\n';
            }
        }
        spdlog::info("dumped {} features to {}", features.size(), path + "_features.txt");
    }

    // IMU 数据
    {
        std::ofstream ofs(path + "_imu.txt");
        if (!ofs) {
            spdlog::error("dumpToFile: cannot open {}", path + "_imu.txt");
            return;
        }
        ofs << "# ts acc_x acc_y acc_z gyro_x gyro_y gyro_z\n";
        size_t total = 0;
        for (int i = 0; i < imu_manager_->numIntegrators(); ++i) {
            const auto& buf = imu_manager_->getIntegrator(i)->buffer;
            for (const auto& m : buf) {
                ofs << std::fixed << m.timestamp << ' ' << m.acc.x() << ' ' << m.acc.y() << ' '
                    << m.acc.z() << ' ' << m.gyro.x() << ' ' << m.gyro.y() << ' ' << m.gyro.z()
                    << '\n';
                ++total;
            }
        }
        spdlog::info("dumped {} IMU measurements to {}", total, path + "_imu.txt");
    }
}

Eigen::Matrix<double, 18, 18> VioEstimator::initNoise() const {
    Eigen::Matrix<double, 18, 18> noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (option_.acc_n * option_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (option_.gyr_n * option_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (option_.acc_n * option_.acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (option_.gyr_n * option_.gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (option_.acc_w * option_.acc_w) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (option_.gyr_w * option_.gyr_w) * Eigen::Matrix3d::Identity();
    return noise;
}

}  // namespace tassel_core
