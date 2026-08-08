#pragma once

namespace tassel_core::profiling {

void pauseVtuneCollection();
void resumeVtuneCollection();

class VtuneProfileScope {
public:
    VtuneProfileScope();
    ~VtuneProfileScope();

    VtuneProfileScope(const VtuneProfileScope&) = delete;
    VtuneProfileScope& operator=(const VtuneProfileScope&) = delete;
};

}  // namespace tassel_core::profiling
