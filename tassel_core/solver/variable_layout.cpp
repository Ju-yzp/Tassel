#include "solver/variable_layout.h"

#include <stdexcept>

namespace tassel_core {

bool VariableKey::operator<(const VariableKey& other) const {
    if (owner != other.owner) {
        return owner < other.owner;
    }
    return kind < other.kind;
}

VariableLayout::VariableLayout(const std::vector<VariableSpec>& specs) {
    ordered_blocks_.reserve(specs.size());
    for (const VariableSpec& spec : specs) {
        if (spec.size <= 0) {
            throw std::invalid_argument("Variable block size must be positive");
        }
        const VariableBlock block{total_size_, spec.size, spec.role};
        if (!blocks_.emplace(spec.key, block).second) {
            throw std::invalid_argument("Variable layout contains a duplicate key");
        }
        ordered_blocks_.push_back(block);
        total_size_ += spec.size;
    }
}

const VariableBlock& VariableLayout::block(const VariableKey& key) const {
    const auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        throw std::out_of_range("Variable key is not present in the layout");
    }
    return it->second;
}

std::vector<int> VariableLayout::columns(VariableRole role) const {
    std::vector<int> result;
    for (const VariableBlock& block : ordered_blocks_) {
        if (block.role != role) {
            continue;
        }
        for (int column = block.offset; column < block.offset + block.size; ++column) {
            result.push_back(column);
        }
    }
    return result;
}

}  // namespace tassel_core
