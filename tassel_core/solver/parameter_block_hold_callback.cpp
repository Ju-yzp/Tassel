#include "solver/parameter_block_hold_callback.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tassel_core {

ParameterBlockHoldCallback::ParameterBlockHoldCallback(std::vector<HeldParameterBlock> blocks)
    : blocks_(std::move(blocks)) {
    snapshots_.reserve(blocks_.size());
    for (const HeldParameterBlock& block : blocks_) {
        if (!block.data || block.size <= 0) {
            throw std::invalid_argument("Held parameter block is invalid");
        }
        snapshots_.emplace_back(block.data, block.data + block.size);
    }
}

ceres::CallbackReturnType ParameterBlockHoldCallback::operator()(
    const ceres::IterationSummary&) {
    restore();
    return ceres::SOLVER_CONTINUE;
}

void ParameterBlockHoldCallback::restore() const {
    for (size_t i = 0; i < blocks_.size(); ++i) {
        std::copy(snapshots_[i].begin(), snapshots_[i].end(), blocks_[i].data);
    }
}

}  // namespace tassel_core
