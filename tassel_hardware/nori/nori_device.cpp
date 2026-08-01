#include "nori/nori_device.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

namespace tassel_hardware {
namespace {

int ioctlRetry(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

void checkIoctl(int fd, unsigned long request, void* argument, const char* name) {
    if (ioctlRetry(fd, request, argument) < 0) {
        throw std::runtime_error(std::string(name) + ": " + std::strerror(errno));
    }
}

struct MappedBuffer {
    void* data = nullptr;
    std::size_t size = 0;
};

}  // namespace

class NoriDevice::Impl {
public:
    Impl(const std::string& path, NoriDeviceConfig requested) : config(std::move(requested)) {
        if (config.width <= 0 || config.height <= 0 || config.fps <= 0 ||
            config.buffer_count < 2) {
            throw std::invalid_argument("Invalid Nori capture configuration");
        }
        if (config.pixel_format == 0) {
            config.pixel_format = V4L2_PIX_FMT_YUYV;
        }

        fd = open(path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            throw std::runtime_error("Failed to open " + path + ": " + std::strerror(errno));
        }
        try {
            configure();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void configure() {
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = config.width;
        format.fmt.pix.height = config.height;
        format.fmt.pix.pixelformat = config.pixel_format;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        checkIoctl(fd, VIDIOC_S_FMT, &format, "VIDIOC_S_FMT");
        if (static_cast<int>(format.fmt.pix.width) != config.width ||
            static_cast<int>(format.fmt.pix.height) != config.height ||
            format.fmt.pix.pixelformat != config.pixel_format) {
            throw std::runtime_error("Nori device rejected the requested capture format");
        }
        bytes_per_line = static_cast<int>(format.fmt.pix.bytesperline);

        v4l2_streamparm parameters{};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator = config.fps;
        checkIoctl(fd, VIDIOC_S_PARM, &parameters, "VIDIOC_S_PARM");
        const auto& interval = parameters.parm.capture.timeperframe;
        if (interval.numerator == 0 || interval.denominator == 0 ||
            interval.denominator != static_cast<std::uint32_t>(config.fps) * interval.numerator) {
            throw std::runtime_error("Nori device rejected the requested frame rate");
        }

        v4l2_requestbuffers request{};
        request.count = config.buffer_count;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        checkIoctl(fd, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
        if (request.count < 2) {
            throw std::runtime_error("Nori device returned too few mmap buffers");
        }
        buffers.resize(request.count);
        for (std::uint32_t index = 0; index < request.count; ++index) {
            v4l2_buffer buffer{};
            buffer.type = request.type;
            buffer.memory = request.memory;
            buffer.index = index;
            checkIoctl(fd, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
            buffers[index].size = buffer.length;
            buffers[index].data = mmap(
                nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buffer.m.offset);
            if (buffers[index].data == MAP_FAILED) {
                buffers[index].data = nullptr;
                throw std::runtime_error(std::string("mmap: ") + std::strerror(errno));
            }
            checkIoctl(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        checkIoctl(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
        streaming = true;
    }

    void release() noexcept {
        if (fd < 0) {
            return;
        }
        if (streaming) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctlRetry(fd, VIDIOC_STREAMOFF, &type);
            streaming = false;
        }
        for (auto& buffer : buffers) {
            if (buffer.data != nullptr) {
                munmap(buffer.data, buffer.size);
                buffer.data = nullptr;
            }
        }
        close(fd);
        fd = -1;
    }

    bool tryRead(NoriCapture& capture) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (ioctlRetry(fd, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) {
                return false;
            }
            throw std::runtime_error(std::string("VIDIOC_DQBUF: ") + std::strerror(errno));
        }
        if (buffer.index >= buffers.size() || buffer.bytesused > buffers[buffer.index].size) {
            throw std::logic_error("Nori returned an invalid capture buffer");
        }
        const auto* begin = static_cast<const std::uint8_t*>(buffers[buffer.index].data);
        capture.bytes.assign(begin, begin + buffer.bytesused);
        capture.width = config.width;
        capture.height = config.height;
        capture.bytes_per_line = bytes_per_line;
        capture.pixel_format = config.pixel_format;
        checkIoctl(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
        return true;
    }

    NoriDeviceConfig config;
    int bytes_per_line = 0;
    int fd = -1;
    bool streaming = false;
    std::vector<MappedBuffer> buffers;
};

NoriDevice::NoriDevice(const std::string& path, NoriDeviceConfig config)
    : impl_(std::make_unique<Impl>(path, std::move(config))) {}

NoriDevice::~NoriDevice() = default;
NoriDevice::NoriDevice(NoriDevice&&) noexcept = default;
NoriDevice& NoriDevice::operator=(NoriDevice&&) noexcept = default;

bool NoriDevice::tryRead(NoriCapture& capture) { return impl_->tryRead(capture); }

const NoriDeviceConfig& NoriDevice::config() const { return impl_->config; }

int NoriDevice::bytesPerLine() const { return impl_->bytes_per_line; }

}  // namespace tassel_hardware
