#include "estimator/window_optimizer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include <ceres/loss_function.h>
#include <spdlog/spdlog.h>

#include "factor/imu_factor.h"
#include "factor/marginalization_prior_factor.h"
#include "factor/reprojection_factor.h"
#include "factor/visual_frame_cache.h"
#include "marg/marg_lin_data.h"
#include "parameters/parameters.h"
#include "state/state.h"
#include "tassel_utils/macros.h"
#include "tassel_utils/se3_right_manifold.h"

namespace tassel_core {
namespace {

constexpr int kRetainedFrameIndex = 0;

ceres::TrustRegionStrategyType ceresTrustRegionStrategy(
    tassel_tools::TrustRegionStrategy strategy) {
    switch (strategy) {
        case tassel_tools::TrustRegionStrategy::LevenbergMarquardt:
            return ceres::LEVENBERG_MARQUARDT;
        case tassel_tools::TrustRegionStrategy::Dogleg:
            return ceres::DOGLEG;
    }
    throw std::logic_error("Unknown trust-region strategy");
}

template <typename Range>
bool allFinite(const Range& values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

std::vector<Eigen::Vector3d> captureAccelBiasMeans(const State& state) {
    std::vector<Eigen::Vector3d> means;
    means.reserve(static_cast<size_t>(state.latest_active_frame_index));
    for (int i = 1; i <= state.latest_active_frame_index; ++i) {
        // speed_bias 的隐含局部布局为 [v(3), ba(3), bg(3)]。
        means.emplace_back(
            Eigen::Map<const Eigen::Vector3d>(state.frames[i].param_speed_bias.data() + 3));
    }
    return means;
}

void restoreAccelBiasMeans(State& state, const std::vector<Eigen::Vector3d>& means) {
    if (means.size() != static_cast<size_t>(state.latest_active_frame_index)) {
        throw std::logic_error("Acceleration-bias snapshot does not match the active window");
    }
    for (int i = 1; i <= state.latest_active_frame_index; ++i) {
        Eigen::Map<Eigen::Vector3d>(state.frames[i].param_speed_bias.data() + 3) =
            means[static_cast<size_t>(i - 1)];
    }
}

#if defined(CERES_HAS_SCHUR_LAYOUT_CALLBACK)
ceres::Solver::Options::SchurLayoutCallback makeSchurLayoutCallback() {
    return [](int chunk_index, const std::vector<int>& default_block_order,
              const std::vector<int>& block_sizes, std::vector<int>* block_order) {
        (void)chunk_index;
        if (block_order == nullptr) {
            throw std::invalid_argument("Schur layout callback received a null output");
        }
        if (default_block_order.size() != block_sizes.size()) {
            throw std::invalid_argument("Schur layout callback received mismatched layout data");
        }

        struct RankedBlock {
            int priority;
            int order;
            int block_id;
        };

        const auto blockPriority = [](int size) {
            if (size == 6) {
                return 0;
            }
            if (size == 9) {
                return 1;
            }
            if (size == 1) {
                return 2;
            }
            return 3;
        };
        std::vector<RankedBlock> ranked;
        ranked.reserve(default_block_order.size());
        for (int i = 0; i < static_cast<int>(default_block_order.size()); ++i) {
            ranked.push_back(
                {blockPriority(block_sizes[static_cast<size_t>(i)]), i,
                 default_block_order[static_cast<size_t>(i)]});
        }
        std::stable_sort(
            ranked.begin(), ranked.end(), [](const RankedBlock& a, const RankedBlock& b) {
                if (a.priority != b.priority) {
                    return a.priority < b.priority;
                }
                return a.order < b.order;
            });

        block_order->clear();
        block_order->reserve(ranked.size());
        for (const RankedBlock& item : ranked) {
            block_order->push_back(item.block_id);
        }
    };
}
#endif

std::unique_ptr<VisualFrameCache> createVisualFrameCache(
    State& state, const tassel_tools::Parameters& params) {
    return std::make_unique<VisualFrameCache>(state, params.ric);
}

void addParameterBlocks(ceres::Problem& problem, State& state, const MargLinData* prior) {
    for (int i = 0; i < state.max_frame_count; ++i) {
        problem.AddParameterBlock(state.frames[i].param_pose.data(), 6, new SE3RightManifold());
        if (i != kRetainedFrameIndex || prior == nullptr) {
            problem.AddParameterBlock(state.frames[i].param_speed_bias.data(), 9);
        }
    }
    if (prior == nullptr) {
        // 初始化期间 frame0 为空，不参与任何物理因子。
        problem.SetParameterBlockConstant(
            state.frames[kRetainedFrameIndex].param_speed_bias.data());
        problem.SetParameterBlockConstant(state.frames[kRetainedFrameIndex].param_pose.data());
    }
    problem.AddParameterBlock(&state.param_time_delay, 1);
}

void configureDelayParameter(
    ceres::Problem& problem, State& state, const tassel_tools::Parameters& params) {
    int observable_frames = 0;
    for (int i = 0; i <= state.latest_active_frame_index; ++i) {
        const FrameState& frame = state.frames[i];
        const bool angular_motion =
            (frame.imu_gyro - frame.gyro_bias).norm() > params.delay_obs_gyro_threshold;
        const bool linear_motion = frame.vel_w.norm() > params.delay_obs_speed_threshold;
        if (angular_motion || linear_motion) {
            ++observable_frames;
        }
    }
    if (observable_frames < params.delay_obs_min_frames) {
        problem.SetParameterBlockConstant(&state.param_time_delay);
    }
}

ceres::ResidualBlockId addPriorFactor(
    ceres::Problem& problem, State& state, const MargLinData* prior) {
    if (prior == nullptr) {
        return nullptr;
    }
    auto* factor = new MarginalizationPriorFactor(*prior);
    std::vector<double*> blocks;
    blocks.push_back(state.frames[0].param_pose.data());
    for (int i = 1; i < prior->stateCount(); ++i) {
        blocks.push_back(state.frames[i].param_pose.data());
        blocks.push_back(state.frames[i].param_speed_bias.data());
    }
    blocks.push_back(&state.param_time_delay);
    return problem.AddResidualBlock(factor, nullptr, blocks);
}

void addVisualFactors(
    ceres::Problem& problem, VisualFrameCache& cache, State& state,
    const tassel_tools::Parameters& params, const std::vector<std::pair<int, Feature>>& features,
    std::vector<double>& inverse_depths, std::vector<int>& factors_per_frame) {
    const double huber_delta = params.reproj_huber_thres * params.visual_factor_weight;
    ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);
    inverse_depths.resize(features.size());
    // inverse depth 与 feature 索引一一对应，求解后按同一索引生成深度结果。
    for (size_t k = 0; k < features.size(); ++k) {
        const Feature& feature = features[k].second;
        TASSEL_ASSERT(std::isfinite(feature.estimated_depth) && feature.estimated_depth > 1e-12);
        inverse_depths[k] = 1.0 / feature.estimated_depth;
        problem.AddParameterBlock(&inverse_depths[k], 1);
    }

    for (size_t k = 0; k < features.size(); ++k) {
        const Feature& feature = features[k].second;
        const int host_index = feature.host_frame_index;
        if (host_index < 0 || host_index > state.latest_active_frame_index) {
            throw std::logic_error("Feature host index is outside the active window");
        }
        for (size_t observation_index = 0; observation_index < feature.observations.size();
             ++observation_index) {
            const int frame_index = feature.observationFrameIndex(observation_index);
            if (frame_index <= state.latest_active_frame_index) {
                ++factors_per_frame[frame_index];
            }
        }
        for (size_t observation_index = 1; observation_index < feature.observations.size();
             ++observation_index) {
            const int target_index = feature.observationFrameIndex(observation_index);
            if (target_index > state.latest_active_frame_index) {
                throw std::logic_error("Feature target index is outside the active window");
            }
            const FeaturePerFrame& host = feature.observations.front();
            const FeaturePerFrame& target = feature.observations[observation_index];
            if (host.sync_delay != state.frames[host_index].image_sync_delay ||
                target.sync_delay != state.frames[target_index].image_sync_delay) {
                throw std::logic_error("Feature sync delay does not match its frame state");
            }
            const Eigen::Vector2d target_pixel(target.pt.x, target.pt.y);
            auto* cost = new ReprojectionFactor(
                host.uv, target_pixel, params.ric, params.tic, state.frames[host_index].imu_gyro,
                state.frames[target_index].imu_gyro, state.frames[host_index].imu_acc,
                state.frames[target_index].imu_acc,
                state.frames[host_index].param_speed_bias.data(),
                state.frames[target_index].param_speed_bias.data(),
                state.frames[host_index].param_speed_bias.data() + 6,
                state.frames[target_index].param_speed_bias.data() + 6,
                state.frames[host_index].param_speed_bias.data() + 3,
                state.frames[target_index].param_speed_bias.data() + 3, state.visual_sqrt_info,
                state.camera, host.sync_delay, target.sync_delay, &state, host_index, target_index);
            problem.AddResidualBlock(
                cost, loss, state.frames[host_index].param_pose.data(),
                state.frames[target_index].param_pose.data(), &state.param_time_delay,
                &inverse_depths[k]);
        }
    }
}

ceres::Solver::Options createSolverOptions(
    const tassel_tools::Parameters& params, State& state, const MargLinData* prior,
    std::vector<double>& inverse_depths) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
#if defined(CERES_HAS_SCHUR_STRUCTURE_HINTS)
    options.schur_structure_row_block_size = 2;
    options.schur_structure_e_block_size = 1;
    options.schur_structure_f_block_size = -1;
#endif
    options.trust_region_strategy_type = ceresTrustRegionStrategy(params.trust_region_strategy);
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (double& inverse_depth : inverse_depths) {
        ordering->AddElementToGroup(&inverse_depth, 0);
    }
    int group = 1;
    for (int i = 0; i < state.max_frame_count; ++i) {
        ordering->AddElementToGroup(state.frames[i].param_pose.data(), group++);
        if (i != kRetainedFrameIndex || prior == nullptr) {
            ordering->AddElementToGroup(state.frames[i].param_speed_bias.data(), group++);
        }
    }
    ordering->AddElementToGroup(&state.param_time_delay, group);
    options.linear_solver_ordering = std::move(ordering);
#if defined(CERES_HAS_SCHUR_LAYOUT_CALLBACK)
    options.schur_layout_callback = makeSchurLayoutCallback();
#endif
    options.max_num_iterations = params.num_iterations;
    if (params.max_solver_time > 0.0) {
        options.max_solver_time_in_seconds = params.max_solver_time;
    }
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    return options;
}

}  // namespace

WindowOptimizer::WindowOptimizer(
    const tassel_tools::Parameters& params, std::shared_ptr<State> state)
    : params_(params), state_(std::move(state)) {
    if (!state_) {
        throw std::invalid_argument("WindowOptimizer requires a state");
    }
}

WindowOptimizationResult WindowOptimizer::solve(
    const std::vector<std::pair<int, Feature>>& features,
    std::vector<MidPointIntegrator>& preintegrators, const MargLinData* prior,
    bool hold_accel_bias) {
    return solveImpl(features, preintegrators, prior, hold_accel_bias);
}

WindowOptimizationResult WindowOptimizer::solve(
    const std::vector<std::pair<int, Feature>>& features,
    std::vector<EulerIntegrator>& preintegrators, const MargLinData* prior, bool hold_accel_bias) {
    return solveImpl(features, preintegrators, prior, hold_accel_bias);
}

template <typename Integrator>
WindowOptimizationResult WindowOptimizer::solveImpl(
    const std::vector<std::pair<int, Feature>>& features, std::vector<Integrator>& preintegrators,
    const MargLinData* prior, bool hold_accel_bias) {
    const std::vector<Eigen::Vector3d> accel_bias_means =
        hold_accel_bias ? captureAccelBiasMeans(*state_) : std::vector<Eigen::Vector3d>{};

    auto visual_cache = createVisualFrameCache(*state_, params_);
    ceres::Problem::Options problem_options;
    problem_options.context = ceres_context_.get();
    problem_options.evaluation_callback = visual_cache.get();
    ceres::Problem problem(problem_options);
    addParameterBlocks(problem, *state_, prior);
    configureDelayParameter(problem, *state_, params_);
    addPriorFactor(problem, *state_, prior);

    std::vector<double> inverse_depths;
    WindowOptimizationResult result;
    result.visual_factors_per_frame.assign(state_->latest_active_frame_index + 1, 0);
    addVisualFactors(
        problem, *visual_cache, *state_, params_, features, inverse_depths,
        result.visual_factors_per_frame);

    const int imu_start = prior == nullptr ? 0 : 1;
    for (int i = imu_start; i < state_->latest_active_frame_index; ++i) {
        if (preintegrators[i].buffer.size() < 2) {
            continue;
        }
        auto integrator = std::shared_ptr<Integrator>(&preintegrators[i], [](Integrator*) {});
        problem.AddResidualBlock(
            new IMUFactor<Integrator>(std::move(integrator)), nullptr,
            state_->frames[i].param_pose.data(), state_->frames[i].param_speed_bias.data(),
            state_->frames[i + 1].param_pose.data(), state_->frames[i + 1].param_speed_bias.data());
    }

    ceres::Solver::Summary summary;
    ceres::Solver::Options options = createSolverOptions(params_, *state_, prior, inverse_depths);
    ceres::Solve(options, &problem, &summary);
    bool finite_solution = std::isfinite(state_->param_time_delay);
    for (int i = 0; i <= state_->latest_active_frame_index && finite_solution; ++i) {
        finite_solution = allFinite(state_->frames[i].param_pose) &&
                          allFinite(state_->frames[i].param_speed_bias);
    }
    finite_solution = finite_solution && allFinite(inverse_depths);
    if (!summary.IsSolutionUsable() || !finite_solution) {
        spdlog::error("Optimization rejected: {}", summary.BriefReport());
        throw std::runtime_error("VIO optimization failed");
    }
    if (hold_accel_bias) {
        // 近静止时保留联合求解得到的其他状态，仅近似保持 Ba 均值；不维护 Schmidt 协方差。
        restoreAccelBiasMeans(*state_, accel_bias_means);
    }
    result.feature_depths.reserve(features.size());
    for (size_t k = 0; k < features.size(); ++k) {
        const double inverse_depth = inverse_depths[k];
        const double depth = inverse_depth > 1e-6 ? 1.0 / inverse_depth : Feature::InvalidDepth;
        result.feature_depths.emplace_back(features[k].first, depth);
    }
    return result;
}

}  // namespace tassel_core
