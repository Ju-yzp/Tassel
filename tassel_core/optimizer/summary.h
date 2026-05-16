#ifndef TASSEL_CORE_SUMMARY_H_
#define TASSEL_CORE_SUMMARY_H_

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <vector>

namespace tassel_core {

enum class LogType { VERBOSE = 0, CONCISE = 1 };

struct PerIterationSummary {
    Eigen::VectorXd step;
    double cost;
    double lambda;
    bool rejected;
};

struct Summary {
    int actual_iterations;
    int num_rejected;
    double init_cost;
    double final_cost;
    double spend_time;
    bool converged;
    std::vector<PerIterationSummary> summaries;

    Summary() { reset(); }

    void log(LogType type = LogType::CONCISE) {
        spdlog::info(
            "actual iterations: {}  num_rejected: {}  init_cost: {:.8f}  final_cost: {:.8f}  "
            "spend_time: {:.8f}",
            actual_iterations, num_rejected, init_cost, final_cost, spend_time);

        if (type == LogType::VERBOSE) {
            for (size_t i = 0; i < summaries.size(); ++i) {
                const auto& s = summaries[i];
                spdlog::info(
                    "Iteration:{:>3} | {} | step: {:.6f} | cost: {:.8f} | lambda: {:.6e}", i,
                    s.rejected ? "rejected" : "accepted", s.step.norm(), s.cost, s.lambda);
            }
        }
    }

    void reset() {
        actual_iterations = 0;
        num_rejected = 0;
        init_cost = 0;
        final_cost = 0;
        spend_time = 0;
        converged = false;
        summaries.clear();
    }
};

}  // namespace tassel_core
#endif /* TASSEL_CORE_SUMMARY_H_ */
