cmake_minimum_required(VERSION 3.24)
project(
  tassel
  VERSION 1.0.0
  LANGUAGES CXX)

# ── Build type ────────────────────────────────────────────────────────────
# Debug          : -O0 -g          → 开发调试，不优化，gdb 友好 RelWithDebInfo : -O2 -g
# -DNDEBUG → 性能分析 / VTune / perf (推荐) Release        : -O3 -DNDEBUG    →
# 最终部署，无调试符号 MinSizeRel     : -Os -DNDEBUG    → 体积优化
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Build options ─────────────────────────────────────────────────────────
option(TASSEL_ENABLE_ASAN
       "Enable AddressSanitizer (leak / OOB / UAF detection)" OFF)
option(TASSEL_ENABLE_PROFILING
       "Add frame-pointer + debug symbols for perf / VTune profiling" OFF)
option(TASSEL_ENABLE_HARDWARE_TESTS
       "Build tests that require hardware (depthai camera)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Profiling flags (VTune / perf) ────────────────────────────────────────
if(TASSEL_ENABLE_PROFILING)
  message(STATUS "Profiling support: ENABLED  (-fno-omit-frame-pointer -g)")
  add_compile_options(-fno-omit-frame-pointer -g)
endif()

# ── AddressSanitizer ──────────────────────────────────────────────────────
if(TASSEL_ENABLE_ASAN)
  message(STATUS "AddressSanitizer: ENABLED")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O3)
  add_link_options(-fsanitize=address)
endif()

list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)

add_library(tassel_deps INTERFACE)
target_link_libraries(
  tassel_deps
  INTERFACE Eigen3::Eigen
            opencv_core
            opencv_imgproc
            opencv_highgui
            opencv_video
            opencv_calib3d
            spdlog::spdlog
            yaml-cpp
            Sophus::Sophus
            TBB::tbb)

add_subdirectory(tassel_utils)
add_subdirectory(tassel_core)
add_subdirectory(tassel_tools)

# ── scripts & foxglove package ───────────────────────────────────────────
install(PROGRAMS scripts/generate_foxglove_layout.py scripts/launch_foxglove.py
                 scripts/start_foxglove_bridge.sh
        DESTINATION lib/${PROJECT_NAME}/scripts)

install(
  DIRECTORY tassel_foxglove/
  DESTINATION lib/${PROJECT_NAME}/tassel_foxglove
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.yaml")

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
  PATTERN "*.yaml")
