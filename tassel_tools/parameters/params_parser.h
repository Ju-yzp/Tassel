#ifndef TASSEL_TOOLS_PARAMETERS_PARAMS_PARSER_H_
#define TASSEL_TOOLS_PARAMETERS_PARAMS_PARSER_H_

// yaml-cpp
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

// cpp
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <utility>

namespace tassel_tools {
namespace fs = std::filesystem;

template <typename T>
concept IsMatrixType =
    std::derived_from<T, Eigen::DenseBase<T>> || std::same_as<std::decay_t<T>, cv::Mat>;

template <typename T>
concept IsStringLike = std::convertible_to<T, std::string_view>;

// TODO: 必须升级到 yaml-cpp 0.7.0+ 以支持线程安全的引用计数并避免段错误。
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
            constexpr int Options = T::Options;
            T matrix;
            if (node.IsSequence() && node.size() > 0 && node[0].IsSequence()) {
                if (node.size() != rows || node[0].size() != cols) {
                    throw std::runtime_error("Nested matrix dimension mismatch!");
                }
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        matrix(i, j) = node[i][j].template as<Scalar>();
                    }
                }
            } else if (node.IsSequence()) {
                auto vec = node.template as<std::vector<Scalar>>();
                if (vec.size() != static_cast<size_t>(rows * cols)) {
                    throw std::runtime_error("Flat matrix dimension mismatch!");
                }
                matrix = Eigen::Map<const Eigen::Matrix<Scalar, rows, cols, Options>>(vec.data());
            } else {
                throw std::runtime_error("Parameter is not a sequence!");
            }

            return matrix;
        } else if constexpr (std::same_as<std::decay_t<T>, cv::Mat>) {
            if (node.IsSequence() && node.size() > 0 && node[0].IsSequence()) {
                size_t rows = node.size();
                size_t cols = node[0].size();
                cv::Mat mat(rows, cols, CV_64F);
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        mat.at<double>(i, j) = node[i][j].template as<double>();
                    }
                }
                return mat;
            } else if (node.IsSequence()) {
                size_t rows = node.size();
                cv::Mat mat(rows, 1, CV_64F);
                for (size_t i = 0; i < rows; ++i) {
                    mat.at<double>(i, 0) = node[i].template as<double>();
                }
                return mat;
            } else {
                throw std::runtime_error("Parameter is not a sequence!");
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
        // YAML::Node node = config_;
        // YAML::Node result;
        // return ((result = node[std::forward<Args>(keys)]), ...);
        std::vector<YAML::Node> node_path;
        node_path.reserve(sizeof...(keys) + 1);

        node_path.push_back(config_);
        auto traverse = [&](auto&& key) {
            YAML::Node next_node = node_path.back()[std::forward<decltype(key)>(key)];

            if (!next_node.IsDefined()) {
                throw std::runtime_error("Key [" + std::string(key) + "] not found in hierarchy!");
            }

            node_path.push_back(next_node);
        };

        (traverse(std::forward<Args>(keys)), ...);

        if (node_path.back().IsNull()) {
            throw std::runtime_error("Final node is null!");
        }

        return node_path.back();
    }
    YAML::Node config_;
};
}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_PARAMETERS_PARAMS_PARSER_H_
