#!/usr/bin/env bash

# Tassel 开发环境配置脚本。
# 推荐使用 source 执行，使 ROS_DISTRO 和 CMAKE_PREFIX_PATH 保留在当前终端：
#   source scripts/setup_environment.sh
#   source scripts/setup_environment.sh --install

if [[ "${1:-}" == "--install" ]]; then
    # 安装软件构建、数学库、ROS 2 消息和 Foxglove Bridge 依赖。
    if ! sudo apt update; then
        return 1 2>/dev/null || exit 1
    fi
    if ! sudo apt install -y \
        build-essential cmake ninja-build pkg-config \
        libeigen3-dev libopencv-dev libceres-dev libspdlog-dev libyaml-cpp-dev \
        libfastcdr-dev libgtest-dev python3 python3-yaml \
        ros-humble-rclcpp ros-humble-std-msgs ros-humble-sensor-msgs \
        ros-humble-nav-msgs ros-humble-geometry-msgs ros-humble-visualization-msgs \
        ros-humble-cv-bridge ros-humble-tf2 ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs ros-humble-foxglove-bridge; then
        return 1 2>/dev/null || exit 1
    fi
elif [[ $# -gt 0 ]]; then
    echo "用法: source scripts/setup_environment.sh [--install]" >&2
    return 2 2>/dev/null || exit 2
fi

# 加载 ROS 2 Humble。项目的 Viewer 和测试程序依赖 ROS 2 CMake 包。
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "错误: 未找到 /opt/ros/humble/setup.bash，请先安装 ROS 2 Humble。" >&2
    return 1 2>/dev/null || exit 1
fi
source /opt/ros/humble/setup.bash

# 将常用的本地安装前缀加入 CMake 搜索路径，供 Sophus 等依赖使用。
_tassel_prefixes=("$HOME/.local" "/usr/local")
if [[ -n "${SOPHUS_ROOT:-}" ]]; then
    _tassel_prefixes=("$SOPHUS_ROOT" "${_tassel_prefixes[@]}")
fi

for _prefix in "${_tassel_prefixes[@]}"; do
    if [[ -d "$_prefix" ]]; then
        case ":${CMAKE_PREFIX_PATH:-}:" in
            *":$_prefix:"*) ;;
            *) export CMAKE_PREFIX_PATH="$_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" ;;
        esac
    fi
done

unset _prefix _tassel_prefixes

echo "Tassel 环境已加载。"
echo "ROS_DISTRO=${ROS_DISTRO:-未设置}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-未设置}"
