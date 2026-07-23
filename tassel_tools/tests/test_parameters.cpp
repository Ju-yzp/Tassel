#include "parameters/parameters.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace {

std::filesystem::path configPath(const std::string& name) {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "config" /
           name;
}

TEST(ParametersTest, LoadsEuRoCRadTanModels) {
    const tassel_tools::Parameters params(configPath("euroc.yaml").string());
    EXPECT_EQ(params.camera_model_map.at(0), "radtan");
    EXPECT_EQ(params.camera_model_map.at(1), "radtan");
    EXPECT_DOUBLE_EQ(params.loop_min_probability, 0.03);
    EXPECT_DOUBLE_EQ(params.loop_pnp_fallback_min_score, 0.20);
    EXPECT_EQ(params.loop_pnp_max_candidates, 3);
    EXPECT_GT(params.loop_pnp_min_inlier_ratio, 0.0);
    EXPECT_LE(params.loop_pnp_min_inlier_ratio, 1.0);
    EXPECT_GT(params.loop_pnp_inlier_threshold, 0.0);
    EXPECT_EQ(params.loop_pnp_max_iterations, 1000);
    EXPECT_EQ(params.loop_pnp_variance_quantile_divisor, 4);
    EXPECT_DOUBLE_EQ(params.loop_pnp_max_translation_variance, 0.0);
    EXPECT_DOUBLE_EQ(params.loop_optimize_max_error, 3.0);
}

TEST(ParametersTest, LoadsTumViEquidistantCalibrationAndNoise) {
    const tassel_tools::Parameters params(configPath("tumvi_room2.yaml").string());
    EXPECT_EQ(params.camera_model_map.at(0), "equi");
    EXPECT_EQ(params.camera_model_map.at(1), "equi");
    EXPECT_EQ(params.rows, 1024);
    EXPECT_EQ(params.cols, 1024);
    EXPECT_NEAR(params.cam_intrinsic_map.at(0).at<double>(0, 0), 380.81042871360756, 1e-10);
    EXPECT_DOUBLE_EQ(params.acc_n, 0.0028);
    EXPECT_DOUBLE_EQ(params.acc_w, 0.00086);
    EXPECT_DOUBLE_EQ(params.gyr_n, 0.00016);
    EXPECT_DOUBLE_EQ(params.gyr_w, 0.000022);
    EXPECT_EQ(params.loop_likelihood_pool_size, 20);
}

}  // namespace
