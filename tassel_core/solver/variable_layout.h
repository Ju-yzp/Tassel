#ifndef TASSEL_CORE_SOLVER_VARIABLE_LAYOUT_H_
#define TASSEL_CORE_SOLVER_VARIABLE_LAYOUT_H_

#include <cstdint>
#include <map>
#include <vector>

namespace tassel_core {

enum class VariableKind {
    Pose,
    Velocity,
    AccelBias,
    GyroBias,
    Delay,
    Landmark,
};

enum class VariableRole {
    Active,
    Schmidt,
    Fixed,
};

struct VariableKey {
    int64_t owner = 0;
    VariableKind kind = VariableKind::Pose;

    bool operator<(const VariableKey& other) const;
};

struct VariableSpec {
    VariableKey key;
    int size = 0;
    VariableRole role = VariableRole::Active;
};

struct VariableBlock {
    int offset = 0;
    int size = 0;
    VariableRole role = VariableRole::Active;
};

class VariableLayout {
public:
    explicit VariableLayout(const std::vector<VariableSpec>& specs);

    const VariableBlock& block(const VariableKey& key) const;
    std::vector<int> columns(VariableRole role) const;
    int totalSize() const { return total_size_; }

private:
    // offset 严格遵循 specs 声明顺序；角色只控制更新策略，不删除联合切空间列。
    std::map<VariableKey, VariableBlock> blocks_;
    std::vector<VariableBlock> ordered_blocks_;
    int total_size_ = 0;
};

}  // namespace tassel_core

#endif  // TASSEL_CORE_SOLVER_VARIABLE_LAYOUT_H_
