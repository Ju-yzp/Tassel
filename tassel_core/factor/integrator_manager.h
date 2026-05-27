#ifndef TASSEL_CORE_FACTOR_INTEGRATOR_MANAGER_H_
#define TASSEL_CORE_FACTOR_INTEGRATOR_MANAGER_H_

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "integrator_base.h"

namespace tassel_core {

class IntegratorManager {
public:
    IntegratorManager() = default;

    Eigen::Vector3d getAcceleration(double ts) const {
        auto m = getIMUAt(ts);
        const auto* integrator = findIntegrator(ts);
        if (integrator) {
            m.acc -= integrator->ba_linearized;
        }
        return m.acc;
    }

    Eigen::Vector3d getGyro(double ts) const {
        auto m = getIMUAt(ts);
        const auto* integrator = findIntegrator(ts);
        if (integrator) {
            m.gyro -= integrator->bg_linearized;
        }
        return m.gyro;
    }

    tassel_utils::IMUMeasurement getIMUAt(double ts) const {
        const auto* integrator = findIntegrator(ts);
        if (integrator) {
            return interpolate(integrator->buffer, ts);
        }
        // ts 超出所有积分器范围，返回最近边界的值
        return nearestEdge(ts);
    }

    MidPointIntegrator* getIntegrator(int i) { return integrators_[i].get(); }

    int numIntegrators() const { return static_cast<int>(integrators_.size()); }

    void addIntegrator(
        Eigen::Vector3d ba_lin, Eigen::Vector3d bg_lin, Eigen::Matrix<double, 18, 18> init_noise) {
        integrators_.push_back(std::make_unique<MidPointIntegrator>(ba_lin, bg_lin, init_noise));
    }

    // 从所有积分器的 buffer 计算实际 IMU 采样间隔
    double computeImuDt() const {
        size_t total = 0;
        double span = 0.0;
        for (const auto& ig : integrators_) {
            const auto& buf = ig->buffer;
            if (buf.size() >= 2) {
                total += buf.size();
                span += buf.back().timestamp - buf.front().timestamp;
            }
        }
        return (total >= 2) ? span / (total - integrators_.size()) : 2.5e-3;
    }

    void removeOldest() {
        if (!integrators_.empty()) {
            integrators_.erase(integrators_.begin());
        }
    }

    void clear() { integrators_.clear(); }

private:
    tassel_utils::IMUMeasurement nearestEdge(double ts) const {
        const tassel_utils::IMUMeasurement* best = nullptr;
        double best_dist = std::numeric_limits<double>::max();
        for (const auto& ig : integrators_) {
            const auto& buf = ig->buffer;
            if (buf.empty()) {
                continue;
            }
            double df = std::abs(buf.front().timestamp - ts);
            if (df < best_dist) {
                best_dist = df;
                best = &buf.front();
            }
            double db = std::abs(buf.back().timestamp - ts);
            if (db < best_dist) {
                best_dist = db;
                best = &buf.back();
            }
        }
        if (best) {
            return *best;
        }
        return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), ts};
    }

    const MidPointIntegrator* findIntegrator(double ts) const {
        if (integrators_.empty()) {
            return nullptr;
        }
        // 二分查找: integrators_ 按各 buffer 时间单调递增
        auto it = std::upper_bound(
            integrators_.begin(), integrators_.end(), ts,
            [](double t, const std::unique_ptr<MidPointIntegrator>& ig) {
                if (ig->buffer.empty()) return true;
                return t < ig->buffer.front().timestamp;
            });
        // 检查前一个积分器是否覆盖 ts
        if (it != integrators_.begin()) {
            const auto& ig = *(it - 1);
            if (!ig->buffer.empty() && ts <= ig->buffer.back().timestamp) {
                return ig.get();
            }
        }
        // 检查当前积分器
        if (it != integrators_.end()) {
            const auto& ig = *it;
            if (!ig->buffer.empty() && ts >= ig->buffer.front().timestamp &&
                ts <= ig->buffer.back().timestamp) {
                return ig.get();
            }
        }
        return nullptr;
    }

    tassel_utils::IMUMeasurement interpolate(
        const std::vector<tassel_utils::IMUMeasurement>& buf, double ts) const {
        if (buf.empty()) {
            return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), ts};
        }
        if (ts <= buf.front().timestamp) {
            return buf.front();
        }
        if (ts >= buf.back().timestamp) {
            return buf.back();
        }

        auto it = std::lower_bound(
            buf.begin(), buf.end(), ts,
            [](const tassel_utils::IMUMeasurement& m, double t) { return m.timestamp < t; });

        if (it == buf.begin()) {
            return *it;
        }

        auto prev = it - 1;
        double dt = it->timestamp - prev->timestamp;
        if (dt < 1e-12) {
            return *prev;
        }

        double alpha = (ts - prev->timestamp) / dt;
        tassel_utils::IMUMeasurement result;
        result.timestamp = ts;
        result.acc = (1.0 - alpha) * prev->acc + alpha * it->acc;
        result.gyro = (1.0 - alpha) * prev->gyro + alpha * it->gyro;
        return result;
    }

    std::vector<std::unique_ptr<MidPointIntegrator>> integrators_;
};

}  // namespace tassel_core

#endif /* TASSEL_CORE_FACTOR_INTEGRATOR_MANAGER_H_ */
