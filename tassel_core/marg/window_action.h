#ifndef TASSEL_CORE_MARG_WINDOW_ACTION_H_
#define TASSEL_CORE_MARG_WINDOW_ACTION_H_

namespace tassel_core {

enum class RetainedHostAction {
    InitializeRetainedSlot,
    ReplaceRetainedSlot,
    MarginalizeOldestFrame,
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_MARG_WINDOW_ACTION_H_
