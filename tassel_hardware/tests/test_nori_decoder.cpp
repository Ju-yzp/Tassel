#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "nori/nori_decoder.h"

namespace tassel_hardware {
namespace {

constexpr int kWidth = 1200;
constexpr int kHeight = 128;
constexpr int kCodeSize = 8;

void writeU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    bytes[offset] = static_cast<uint8_t>(value >> 24);
    bytes[offset + 1] = static_cast<uint8_t>(value >> 16);
    bytes[offset + 2] = static_cast<uint8_t>(value >> 8);
    bytes[offset + 3] = static_cast<uint8_t>(value);
}

void writeI16(std::vector<uint8_t>& bytes, size_t offset, int16_t value) {
    bytes[offset] = static_cast<uint8_t>(static_cast<uint16_t>(value) >> 8);
    bytes[offset + 1] = static_cast<uint8_t>(value);
}

void encodeLine(cv::Mat& image, int row, const std::vector<uint8_t>& bytes) {
    image.row(row).setTo(cv::Scalar(255, 255, 255));
    uint8_t* pixels = image.ptr<uint8_t>(row);
    pixels[(kCodeSize / 2) * 3 + 1] = 0;
    pixels[(kCodeSize / 2 + 1) * 3 + 1] = 0;
    int column = kCodeSize;
    for (uint8_t value : bytes) {
        for (int bit = 0; bit < 8; ++bit) {
            const uint8_t encoded = (value & (1 << bit)) ? 100 : 0;
            pixels[column * 3 + 1] = encoded;
            pixels[(column + 1) * 3 + 1] = encoded;
            column += kCodeSize;
        }
    }
}

cv::Mat packetImage(uint32_t exposure_end_us, uint32_t imu_timestamp_us) {
    cv::Mat image(kHeight, kWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<uint8_t> header(16, 0);
    header[1] = 0x11;  // ICM-42688, one record.
    writeU32(header, 8, exposure_end_us - 100);
    writeU32(header, 12, exposure_end_us);

    std::vector<uint8_t> imu(16, 0);
    writeU32(imu, 0, imu_timestamp_us);
    writeI16(imu, 4, 8192);
    writeI16(imu, 6, -8192);
    writeI16(imu, 8, 0);
    writeI16(imu, 10, 3277);
    writeI16(imu, 12, 0);
    writeI16(imu, 14, -3277);

    encodeLine(image, 2, header);
    encodeLine(image, 10, imu);
    return image;
}

TEST(NoriDecoderTest, DecodesTimestampAndPhysicalUnits) {
    NoriDecoder decoder;
    NoriFrameTiming timing;
    std::vector<tassel_utils::IMUMeasurement> measurements;

    ASSERT_TRUE(decoder.decode(packetImage(2000, 1900), timing, measurements));
    ASSERT_EQ(timing.exposure_start, 1900000);
    ASSERT_EQ(timing.exposure_end, 2000000);
    ASSERT_EQ(measurements.size(), 1u);
    EXPECT_NEAR(measurements[0].timestamp, 0.0019, 1e-12);
    EXPECT_NEAR(measurements[0].acc.x(), 9.80665, 1e-9);
    EXPECT_NEAR(measurements[0].acc.y(), -9.80665, 1e-9);
    EXPECT_NEAR(measurements[0].gyro.x(), 100.006103515625 * std::numbers::pi / 180.0, 1e-12);
    EXPECT_NEAR(measurements[0].gyro.z(), -100.006103515625 * std::numbers::pi / 180.0, 1e-12);
}

TEST(NoriDecoderTest, DecodesGrayscalePacket) {
    cv::Mat gray;
    cv::cvtColor(packetImage(2000, 1900), gray, cv::COLOR_BGR2GRAY);
    NoriDecoder decoder;
    NoriFrameTiming timing;
    std::vector<tassel_utils::IMUMeasurement> measurements;

    ASSERT_TRUE(decoder.decode(gray, timing, measurements));
    EXPECT_EQ(timing.exposure_end, 2000000);
    ASSERT_EQ(measurements.size(), 1u);
    EXPECT_NEAR(measurements[0].acc.x(), 9.80665, 1e-9);
}

TEST(NoriDecoderTest, ExtendsTimestampsAcrossThirtyTwoBitWrap) {
    NoriDecoder decoder;
    NoriFrameTiming timing;
    std::vector<tassel_utils::IMUMeasurement> measurements;

    ASSERT_TRUE(decoder.decode(packetImage(0xfffffff0U, 0xffffffe0U), timing, measurements));
    ASSERT_TRUE(decoder.decode(packetImage(0x00000030U, 0x00000020U), timing, measurements));
    ASSERT_EQ(
        timing.exposure_end,
        static_cast<tassel_utils::FrameId>((std::uint64_t{1} << 32) * 1000 + 0x30 * 1000));
    ASSERT_EQ(measurements.size(), 1u);
    EXPECT_NEAR(
        measurements[0].timestamp, (static_cast<double>(std::uint64_t{1} << 32) + 0x20) * 1e-6,
        1e-9);
}

}  // namespace
}  // namespace tassel_hardware
