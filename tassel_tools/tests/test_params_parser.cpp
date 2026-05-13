#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>

#include "parameters/params_parser.h"

namespace fs = std::filesystem;

class ParamsParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        // test YAML lives next to this source file
        fs::path test_dir = fs::path(__FILE__).parent_path();
        yaml_path_ = (test_dir / "test_params.yaml").string();
        parser_ = std::make_unique<tassel_tools::ParamsParser>(yaml_path_);
    }

    std::string yaml_path_;
    std::unique_ptr<tassel_tools::ParamsParser> parser_;
};

// ── scalar types ───────────────────────────────────────────────────────────

TEST_F(ParamsParserTest, ReadInt) { EXPECT_EQ(parser_->as<int>("int_val"), 42); }

TEST_F(ParamsParserTest, ReadDouble) {
    EXPECT_NEAR(parser_->as<double>("double_val"), 3.14159, 1e-10);
}

TEST_F(ParamsParserTest, ReadBoolTrue) { EXPECT_EQ(parser_->as<bool>("bool_true"), true); }

TEST_F(ParamsParserTest, ReadBoolFalse) { EXPECT_EQ(parser_->as<bool>("bool_false"), false); }

TEST_F(ParamsParserTest, ReadString) {
    EXPECT_EQ(parser_->as<std::string>("string_val"), "hello_tassel");
}

TEST_F(ParamsParserTest, ReadSizeT) { EXPECT_EQ(parser_->as<size_t>("size_val"), 100); }

// ── Eigen matrices ─────────────────────────────────────────────────────────

TEST_F(ParamsParserTest, ReadEigenFlatVector) {
    Eigen::Vector3d v = parser_->as<Eigen::Vector3d>("flat_vec3");
    EXPECT_DOUBLE_EQ(v(0), 1.0);
    EXPECT_DOUBLE_EQ(v(1), 2.0);
    EXPECT_DOUBLE_EQ(v(2), 3.0);
}

TEST_F(ParamsParserTest, ReadEigenFlatToMatrix) {
    // flat sequence of 9 doubles mapped to 3x3 (ColMajor = Eigen default)
    Eigen::Matrix3d m = parser_->as<Eigen::Matrix3d>("flat_vec9");
    EXPECT_DOUBLE_EQ(m(0, 0), 1);
    EXPECT_DOUBLE_EQ(m(1, 0), 2);
    EXPECT_DOUBLE_EQ(m(2, 0), 3);
    EXPECT_DOUBLE_EQ(m(0, 1), 4);
    EXPECT_DOUBLE_EQ(m(2, 2), 9);
}

TEST_F(ParamsParserTest, ReadEigenNestedMatrix) {
    Eigen::Matrix3d m = parser_->as<Eigen::Matrix3d>("nested_mat3x3");
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 1.0);
}

TEST_F(ParamsParserTest, ReadEigenNestedNonSquare) {
    Eigen::Matrix<double, 2, 3> m = parser_->as<Eigen::Matrix<double, 2, 3>>("nested_mat2x3");
    EXPECT_DOUBLE_EQ(m(0, 0), 10);
    EXPECT_DOUBLE_EQ(m(0, 2), 30);
    EXPECT_DOUBLE_EQ(m(1, 0), 40);
    EXPECT_DOUBLE_EQ(m(1, 2), 60);
}

TEST_F(ParamsParserTest, ReadEigenMatrix4d) {
    Eigen::Matrix4d T = parser_->as<Eigen::Matrix4d>("transform", "T_cam_imu");
    EXPECT_NEAR(T(0, 0), 0.999968, 1e-6);
    EXPECT_NEAR(T(0, 3), 0.009659, 1e-6);
    EXPECT_DOUBLE_EQ(T(3, 0), 0.0);
    EXPECT_DOUBLE_EQ(T(3, 3), 1.0);
}

// ── cv::Mat ────────────────────────────────────────────────────────────────

TEST_F(ParamsParserTest, ReadCvMatNested) {
    cv::Mat K = parser_->as<cv::Mat>("cam0", "intrinsics");
    ASSERT_EQ(K.rows, 3);
    ASSERT_EQ(K.cols, 3);
    ASSERT_EQ(K.type(), CV_64F);
    EXPECT_NEAR(K.at<double>(0, 0), 455.510864, 1e-6);
    EXPECT_NEAR(K.at<double>(0, 2), 328.529851, 1e-6);
    EXPECT_NEAR(K.at<double>(1, 1), 455.426715, 1e-6);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 2), 1.0);
}

TEST_F(ParamsParserTest, ReadCvMatFlat) {
    cv::Mat D = parser_->as<cv::Mat>("cam0_dist", "distortion_coeffs");
    ASSERT_EQ(D.type(), CV_64F);
    ASSERT_EQ(D.rows * D.cols, 5);
    EXPECT_NEAR(D.at<double>(0), 0.010831, 1e-6);
    EXPECT_NEAR(D.at<double>(1), -0.007841, 1e-6);
}

// ── nested key access ──────────────────────────────────────────────────────

TEST_F(ParamsParserTest, ReadNestedKey) {
    int v = parser_->as<int>("outer", "inner", "value");
    EXPECT_EQ(v, 777);
}

// ── error cases ────────────────────────────────────────────────────────────

TEST_F(ParamsParserTest, ThrowsOnMissingFile) {
    EXPECT_THROW(tassel_tools::ParamsParser("nonexistent_file.yaml"), std::runtime_error);
}

TEST_F(ParamsParserTest, ThrowsOnMissingKey) {
    EXPECT_THROW(parser_->as<int>("no_such_key"), std::runtime_error);
}

TEST_F(ParamsParserTest, ThrowsOnWrongType) {
    // "string_val" is a string, reading as int should fail
    EXPECT_THROW(parser_->as<int>("string_val"), std::runtime_error);
}

TEST_F(ParamsParserTest, ThrowsOnMissingNestedKey) {
    EXPECT_THROW(parser_->as<int>("cam0", "nonexistent"), std::runtime_error);
}

TEST_F(ParamsParserTest, PrintedContent) {
    // Verify the parser correctly reads and can be printed
    std::ostringstream oss;
    oss << "int_val: " << parser_->as<int>("int_val") << "\n"
        << "double_val: " << parser_->as<double>("double_val") << "\n"
        << "bool_true: " << parser_->as<bool>("bool_true") << "\n"
        << "string_val: " << parser_->as<std::string>("string_val") << "\n"
        << "size_val: " << parser_->as<size_t>("size_val") << "\n"
        << "nested inner.value: " << parser_->as<int>("outer", "inner", "value");

    std::string out = oss.str();
    EXPECT_NE(out.find("int_val: 42"), std::string::npos);
    EXPECT_NE(out.find("double_val: 3.14159"), std::string::npos);
    EXPECT_NE(out.find("bool_true: 1"), std::string::npos);
    EXPECT_NE(out.find("string_val: hello_tassel"), std::string::npos);
    EXPECT_NE(out.find("nested inner.value: 777"), std::string::npos);

    // also print to stdout for visual inspection
    std::cout << "\n─── ParamsParser read results ───\n" << out << "\n\n";
}
