#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

namespace tassel_utils {

enum class TimeUnit { Millisecond, Microsecond, Nanosecond };

class Timer {
public:
    explicit Timer(std::string name, TimeUnit unit = TimeUnit::Millisecond)
        : name_(std::move(name)), unit_(unit) {}

    Timer() = delete;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(Timer&&) = delete;

    void start() { start_ = std::chrono::steady_clock::now(); }

    double end() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ns = std::chrono::duration<double, std::nano>(now - start_).count();
        double elapsed;
        const char* suffix;
        switch (unit_) {
            case TimeUnit::Microsecond:
                elapsed = elapsed_ns * 1e-3;
                suffix = "us";
                break;
            case TimeUnit::Nanosecond:
                elapsed = elapsed_ns;
                suffix = "ns";
                break;
            case TimeUnit::Millisecond:
            default:
                elapsed = elapsed_ns * 1e-6;
                suffix = "ms";
                break;
        }
        if (elapsed < 0.001) {
            spdlog::info(name_ + ": operation too short to measure in " + suffix);
        }
        std::ostringstream oss;
        oss << name_ << ": " << std::fixed << std::setprecision(3) << elapsed << " " << suffix;
        spdlog::info(oss.str());
        return elapsed;
    }

private:
    std::string name_;
    TimeUnit unit_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace tassel_utils
