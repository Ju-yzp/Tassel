#ifndef TASSEL_CORE_PREINTEGRATOR_H_
#define TASSEL_CORE_PREINTEGRATOR_H_

#include <Eigen/Core>
#include <vector>
#include "tassel_utils/types.h"

namespace tassel_core {
struct PreIntegrator {
    explicit PreIntegrator(Eigen::Vector3d ba, Eigen::Vector3d bg);

    void add_measurement(const tassel_utils::IMUMeasurement& measurement);

    void integrate();

    void reset();

    Eigen::Matrix<double, 15, 1> evaluate(
        const Eigen::Vector3d& Pi, const Eigen::Matrix3d& Qi, const Eigen::Vector3d& Vi,
        const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi, const Eigen::Vector3d& Pj,
        const Eigen::Matrix3d& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj,
        const Eigen::Vector3d& Bgj);

    std::vector<tassel_utils::IMUMeasurement> buffer_;

    Eigen::Matrix<double, 15, 15> convariance_;
    Eigen::Matrix<double, 15, 18> jacobian_;
    Eigen::Matrix<double, 18, 18> noise_;

    Eigen::Vector3d linearized_ba_, linearized_bg_;

    double sum_dt_;

    Eigen::Matrix3d final_delta_q_;
    Eigen::Vector3d final_delta_v_, final_delta_p_;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_PREINTEGRATOR_H_ */
