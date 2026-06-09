#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_

#include <Eigen/Core>

namespace tassel_core {

struct EstimatorOption {
    int num_iterations = 10;
    int num_threads = 1;

    double min_depth = 0.05;
    double max_depth = 10.0;

    double init_time_span = 5.0;

    double g_norm = 9.8;

    bool estimate_ba_init = false;
    Eigen::Vector3d init_ba = Eigen::Vector3d::Zero();
    double min_rot_excitation = 0.2;
    int min_excited_frames = 3;
    int num_init_iterations = 2;

    double acc_n = 0.01;
    double acc_w = 0.001;
    double gyr_n = 0.001;
    double gyr_w = 0.0001;

    Eigen::Matrix3d acc_correction_matrix = Eigen::Matrix3d::Identity();
    Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_
