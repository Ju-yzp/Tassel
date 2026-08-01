#include "nori/nori_decoder.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace tassel_hardware {
namespace {

constexpr int kCodeSize = 8;
constexpr int kFirstEncodedRowOffset = 3;
constexpr int kBytesPerGroup = 16;
constexpr uint8_t kIcm42688DeviceType = 1;
constexpr uint16_t kAccFullScaleMilliG = 4000;
constexpr uint16_t kGyroFullScaleDps = 1000;
constexpr uint8_t kZeroThreshold = 50;
constexpr uint8_t kGrayZeroThreshold = 130;
constexpr uint8_t kOneThreshold = 220;
constexpr std::uint64_t kTimestampModulo = std::uint64_t{1} << 32;
constexpr std::uint64_t kTimestampHalfRange = kTimestampModulo / 2;
constexpr double kGravity = 9.80665;

struct PacketHeader {
    std::array<uint8_t, 5> device_types{};
    std::array<uint8_t, 5> group_counts{};
    std::uint32_t exposure_start_us = 0;
    std::uint32_t exposure_end_us = 0;
};

uint32_t readU32(const uint8_t* bytes) {
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

int16_t readI16(const uint8_t* bytes) {
    return static_cast<int16_t>(
        (static_cast<uint16_t>(bytes[0]) << 8) | static_cast<uint16_t>(bytes[1]));
}

PacketHeader decodeHeader(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < kBytesPerGroup) {
        throw std::logic_error("Nori timestamp header is incomplete");
    }

    PacketHeader header;
    const bool legacy_header = std::equal(bytes.begin(), bytes.begin() + 8, bytes.begin() + 8);
    if (legacy_header) {
        header.device_types[0] = kIcm42688DeviceType;
        header.group_counts[0] = 11;
    } else {
        header.device_types[0] = static_cast<uint8_t>(((bytes[0] & 0x0f) << 4) | (bytes[1] >> 4));
        header.group_counts[0] = bytes[1] & 0x0f;
        header.device_types[1] = bytes[2];
        header.group_counts[1] = bytes[3] >> 4;
        header.device_types[2] = static_cast<uint8_t>(((bytes[3] & 0x0f) << 4) | (bytes[4] >> 4));
        header.group_counts[2] = bytes[4] & 0x0f;
        header.device_types[3] = bytes[5];
        header.group_counts[3] = bytes[6] >> 4;
        header.device_types[4] = static_cast<uint8_t>(((bytes[6] & 0x0f) << 4) | (bytes[7] >> 4));
        header.group_counts[4] = bytes[7] & 0x0f;
    }
    header.exposure_start_us = readU32(bytes.data() + 8);
    header.exposure_end_us = readU32(bytes.data() + 12);
    return header;
}

bool isInvalidIcmRecord(const uint8_t* bytes) {
    const int16_t ax = readI16(bytes + 4);
    const int16_t ay = readI16(bytes + 6);
    const int16_t az = readI16(bytes + 8);
    const int16_t gx = readI16(bytes + 10);
    return (ax == -1 && ay == -1 && az == -1) || gx == std::numeric_limits<int16_t>::min();
}

}  // namespace

bool NoriDecoder::decode(
    const cv::Mat& image, NoriFrameTiming& timing,
    std::vector<tassel_utils::IMUMeasurement>& measurements) {
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        throw std::invalid_argument("Nori IMU decoder requires a grayscale or BGR image");
    }
    if (image.rows <= kFirstEncodedRowOffset || image.cols < 2 * kCodeSize) {
        throw std::invalid_argument("Nori image is too small for timestamp data");
    }

    std::vector<uint8_t> packet;
    const int first_row = kFirstEncodedRowOffset - 1;
    int line_index = 0;
    while (packet.size() < kBytesPerGroup) {
        std::vector<uint8_t> line;
        if (!decodeLine(image, first_row + line_index * kCodeSize, line)) {
            return false;
        }
        packet.insert(packet.end(), line.begin(), line.end());
        ++line_index;
    }

    const PacketHeader header = decodeHeader(packet);
    const int group_count = 1 + std::accumulate(
                                    header.group_counts.begin(), header.group_counts.end(), 0);
    const size_t required_bytes = static_cast<size_t>(group_count) * kBytesPerGroup;
    while (packet.size() < required_bytes) {
        std::vector<uint8_t> line;
        if (!decodeLine(image, first_row + line_index * kCodeSize, line)) {
            return false;
        }
        packet.insert(packet.end(), line.begin(), line.end());
        ++line_index;
    }

    const uint64_t exposure_end_us = unwrapFrameEnd(header.exposure_end_us);
    timing.exposure_end = static_cast<tassel_utils::FrameId>(exposure_end_us * 1000);
    timing.exposure_start = static_cast<tassel_utils::FrameId>(
        timestampNearFrame(header.exposure_start_us, header.exposure_end_us, exposure_end_us) * 1000);

    measurements.clear();
    size_t group_index = 1;
    const double acc_scale =
        static_cast<double>(kAccFullScaleMilliG) * kGravity / (32768.0 * 1000.0);
    const double gyro_scale =
        static_cast<double>(kGyroFullScaleDps) * std::numbers::pi / (32768.0 * 180.0);
    for (size_t device_index = 0; device_index < header.device_types.size(); ++device_index) {
        const uint8_t device_type = header.device_types[device_index];
        const uint8_t count = header.group_counts[device_index];
        for (uint8_t record_index = 0; record_index < count; ++record_index, ++group_index) {
            const uint8_t* record = packet.data() + group_index * kBytesPerGroup;
            if (device_type != kIcm42688DeviceType || isInvalidIcmRecord(record)) {
                continue;
            }
            const uint64_t timestamp_us =
                timestampNearFrame(readU32(record), header.exposure_end_us, exposure_end_us);
            measurements.push_back({
                Eigen::Vector3d(
                    readI16(record + 4) * acc_scale, readI16(record + 6) * acc_scale,
                    readI16(record + 8) * acc_scale),
                Eigen::Vector3d(
                    readI16(record + 10) * gyro_scale, readI16(record + 12) * gyro_scale,
                    readI16(record + 14) * gyro_scale),
                static_cast<double>(timestamp_us) * 1e-6});
        }
    }
    return true;
}

bool NoriDecoder::decodeLine(
    const cv::Mat& image, int row_index, std::vector<uint8_t>& bytes) const {
    if (row_index < 0 || row_index >= image.rows) {
        return false;
    }
    const uint8_t* row = image.ptr<uint8_t>(row_index);
    const int channels = image.channels();
    const int sample_channel = channels == 3 ? 1 : 0;
    const int zero_threshold = channels == 3 ? kZeroThreshold : kGrayZeroThreshold;
    for (int pixel = 0; pixel < 4; ++pixel) {
        if (row[pixel * channels + sample_channel] <= kOneThreshold) {
            return false;
        }
    }

    int column = 0;
    while (column < image.cols && row[column * channels + sample_channel] >= kOneThreshold) {
        ++column;
    }
    column += kCodeSize / 2;
    std::vector<uint8_t> bits;
    while (column + 1 < image.cols) {
        const int value = row[column * channels + sample_channel] +
                          row[(column + 1) * channels + sample_channel];
        if (value < 2 * zero_threshold) {
            bits.push_back(0);
        } else if (value < 2 * kOneThreshold) {
            bits.push_back(1);
        } else {
            break;
        }
        column += kCodeSize;
    }

    bytes.clear();
    bytes.reserve(bits.size() / 8);
    for (size_t bit_index = 0; bit_index + 7 < bits.size(); bit_index += 8) {
        uint8_t value = 0;
        for (int bit = 0; bit < 8; ++bit) {
            value |= bits[bit_index + bit] << bit;
        }
        bytes.push_back(value);
    }
    return !bytes.empty();
}

uint64_t NoriDecoder::unwrapFrameEnd(uint32_t timestamp_us) {
    if (has_frame_timestamp_ && timestamp_us < last_frame_end_us_ &&
        static_cast<uint64_t>(last_frame_end_us_) - timestamp_us > kTimestampHalfRange) {
        frame_epoch_us_ += kTimestampModulo;
    }
    has_frame_timestamp_ = true;
    last_frame_end_us_ = timestamp_us;
    return frame_epoch_us_ + timestamp_us;
}

uint64_t NoriDecoder::timestampNearFrame(
    uint32_t timestamp_us, uint32_t frame_end_us, uint64_t unwrapped_frame_end_us) const {
    int64_t delta = static_cast<int64_t>(timestamp_us) - static_cast<int64_t>(frame_end_us);
    if (delta > static_cast<int64_t>(kTimestampHalfRange)) {
        delta -= static_cast<int64_t>(kTimestampModulo);
    } else if (delta < -static_cast<int64_t>(kTimestampHalfRange)) {
        delta += static_cast<int64_t>(kTimestampModulo);
    }
    return static_cast<uint64_t>(static_cast<int64_t>(unwrapped_frame_end_us) + delta);
}

}  // namespace tassel_hardware
