#ifndef TASSEL_TOOLS_PARAMETERS_PARAMS_PARSER_H_
#define TASSEL_TOOLS_PARAMETERS_PARAMS_PARSER_H_

#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <utility>

namespace tassel_tools {
namespace fs = std::filesystem;

template <typename T>
concept IsMatrixType =
    std::derived_from<T, Eigen::DenseBase<T>> || std::same_as<std::decay_t<T>, cv::Mat>;

template <typename T>
concept IsStringLike = std::convertible_to<T, std::string_view>;

// ── yaml-cpp cross-version compatibility ───────────────────────────────────

namespace detail {

// yaml-cpp 0.6+ provides IsDefined(); 0.5.x only has IsNull().
// operator[] in 0.5.x creates a Null node for missing keys, so IsNull()
// catches a missing path. In 0.6+ a missing key returns an undefined node
// that is neither defined nor null — only IsDefined() catches it.
template <typename, typename = void>
inline constexpr bool kHasIsDefined = false;
template <typename T>
inline constexpr bool
    kHasIsDefined<T, std::void_t<decltype(std::declval<const T&>().IsDefined())>> = true;

inline bool node_valid(const YAML::Node& node) {
    if constexpr (kHasIsDefined<YAML::Node>) {
        return node.IsDefined();
    } else {
        return !node.IsNull();
    }
}

}  // namespace detail

// ── ParamsParser ──────────────────────────────────────────────────────────

class ParamsParser {
public:
    explicit ParamsParser(const std::string& path_str) {
        fs::path config_path(path_str);
        if (!fs::exists(config_path)) {
            throw std::runtime_error(
                "Config file not found: " + fs::absolute(config_path).string());
        }
        config_ = YAML::LoadFile(config_path.string());
    }

    template <IsMatrixType T, typename... Args>
    T as(Args&&... keys) const {
        YAML::Node node = get_node(std::forward<Args>(keys)...);
        if constexpr (std::derived_from<T, Eigen::DenseBase<T>>) {
            using Scalar = typename T::Scalar;
            constexpr int rows = T::RowsAtCompileTime;
            constexpr int cols = T::ColsAtCompileTime;
            constexpr int options = T::Options;
            T matrix;
            if (node.IsSequence() && node.size() > 0 && node[0].IsSequence()) {
                if (node.size() != rows || node[0].size() != cols) {
                    throw std::runtime_error("Nested matrix dimension mismatch");
                }
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j) matrix(i, j) = node[i][j].template as<Scalar>();
            } else if (node.IsSequence()) {
                auto vec = node.template as<std::vector<Scalar>>();
                if (vec.size() != static_cast<size_t>(rows * cols)) {
                    throw std::runtime_error("Flat matrix dimension mismatch");
                }
                matrix = Eigen::Map<const Eigen::Matrix<Scalar, rows, cols, options>>(vec.data());
            } else {
                throw std::runtime_error("Parameter is not a sequence");
            }
            return matrix;
        } else if constexpr (std::same_as<std::decay_t<T>, cv::Mat>) {
            if (node.IsSequence() && node.size() > 0 && node[0].IsSequence()) {
                size_t rows = node.size();
                size_t cols = node[0].size();
                cv::Mat mat(rows, cols, CV_64F);
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        mat.at<double>(i, j) = node[i][j].template as<double>();
                return mat;
            } else if (node.IsSequence()) {
                size_t rows = node.size();
                cv::Mat mat(rows, 1, CV_64F);
                for (size_t i = 0; i < rows; ++i)
                    mat.at<double>(i, 0) = node[i].template as<double>();
                return mat;
            } else {
                throw std::runtime_error("Parameter is not a sequence");
            }
        }
    }

    template <typename T, typename... Args>
    requires(!IsMatrixType<T>) T as(Args&&... keys)
    const { return get_node(std::forward<Args>(keys)...).template as<T>(); }

private:
    template <typename... Args>
    requires(IsStringLike<Args>&&...) YAML::Node get_node(Args&&... keys)
    const {
        YAML::Node node = config_;
        auto step = [&](auto&& key) {
            node.reset(node[std::forward<decltype(key)>(key)]);
            if (!detail::node_valid(node)) {
                throw std::runtime_error(
                    "Key [" + std::string(key) + "] not found in config hierarchy");
            }
        };
        (step(std::forward<Args>(keys)), ...);
        return node;
    }

    YAML::Node config_;
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMS_PARSER_H_
