#ifndef TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_
#define TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_

namespace tassel_core {

struct EstimatorOption {
    int num_iterations = 10;
    int num_threads = 1;

    double min_depth = 0.05;
    double max_depth = 10.0;

    double init_time_span = 5.0;

    double g_norm = 9.8;

    // IMU noise
    double acc_n = 0.01;
    double acc_w = 0.001;
    double gyr_n = 0.001;
    double gyr_w = 0.0001;
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_ESTIMATOR_ESTIMATOR_OPTION_H_
