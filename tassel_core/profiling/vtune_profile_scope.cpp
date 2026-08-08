#include "profiling/vtune_profile_scope.h"

#if defined(TASSEL_HAS_VTUNE_ITT)
#include <ittnotify.h>
#endif

namespace tassel_core::profiling {

void pauseVtuneCollection() {
#if defined(TASSEL_HAS_VTUNE_ITT)
    __itt_pause();
#endif
}

void resumeVtuneCollection() {
#if defined(TASSEL_HAS_VTUNE_ITT)
    __itt_resume();
#endif
}

VtuneProfileScope::VtuneProfileScope() {
    resumeVtuneCollection();
}

VtuneProfileScope::~VtuneProfileScope() {
    pauseVtuneCollection();
}

}  // namespace tassel_core::profiling
