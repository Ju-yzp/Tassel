#ifndef TASSEL_HARDWARE_NORI_NORI_DEVICE_H_
#define TASSEL_HARDWARE_NORI_NORI_DEVICE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tassel_hardware {

struct NoriDeviceConfig {
    int width = 4000;
    int height = 1200;
    int fps = 30;
    std::uint32_t pixel_format = 0;
    std::uint32_t buffer_count = 4;
};

struct NoriCapture {
    std::vector<std::uint8_t> bytes;
    int width = 0;
    int height = 0;
    int bytes_per_line = 0;
    std::uint32_t pixel_format = 0;
};

class NoriDevice {
public:
    explicit NoriDevice(const std::string& path, NoriDeviceConfig config = {});
    ~NoriDevice();

    NoriDevice(const NoriDevice&) = delete;
    NoriDevice& operator=(const NoriDevice&) = delete;
    NoriDevice(NoriDevice&&) noexcept;
    NoriDevice& operator=(NoriDevice&&) noexcept;

    bool tryRead(NoriCapture& capture);
    const NoriDeviceConfig& config() const;
    int bytesPerLine() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tassel_hardware

#endif  // TASSEL_HARDWARE_NORI_NORI_DEVICE_H_
