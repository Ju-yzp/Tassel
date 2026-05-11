#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

namespace tassel_utils {

class Timer {
public:
    explicit Timer(std::string name) : name_(std::move(name)) {}

    Timer() = delete;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(Timer&&) = delete;
    void start() { start_ = std::chrono::steady_clock::now(); }

    double end() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration<double, std::milli>(now - start_).count();
        std::ostringstream oss;
        oss << name_ << ": " << std::fixed << std::setprecision(3) << elapsed_ms << " ms";
        spdlog::info(oss.str());
        return elapsed_ms;
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace tassel_utils
