#ifndef TASSEL_HARDWARE_NORI_NORI_DECODER_H_
#define TASSEL_HARDWARE_NORI_NORI_DECODER_H_

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "tassel_utils/types.h"

namespace tassel_hardware {

struct NoriFrameTiming {
    tassel_utils::FrameId exposure_start = tassel_utils::kInvalidFrameId;
    tassel_utils::FrameId exposure_end = tassel_utils::kInvalidFrameId;
};

class NoriDecoder {
public:
    // 输入为 UVC MJPEG 解码得到的 top-down 灰度或 BGR 图像，时间戳条带从第 3 行开始，每条向下间隔 8 行。
    bool decode(
        const cv::Mat& image, NoriFrameTiming& timing,
        std::vector<tassel_utils::IMUMeasurement>& measurements);

private:
    bool decodeLine(const cv::Mat& image, int row_index, std::vector<uint8_t>& bytes) const;

    std::uint64_t unwrapFrameEnd(std::uint32_t timestamp_us);

    std::uint64_t timestampNearFrame(
        std::uint32_t timestamp_us, std::uint32_t frame_end_us,
        std::uint64_t unwrapped_frame_end_us) const;

    bool has_frame_timestamp_ = false;
    std::uint32_t last_frame_end_us_ = 0;
    std::uint64_t frame_epoch_us_ = 0;
};

}  // namespace tassel_hardware

#endif  // TASSEL_HARDWARE_NORI_NORI_DECODER_H_
