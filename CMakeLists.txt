cmake_minimum_required(VERSION 3.22)
project(
  tassel
  VERSION 1.0.0
  DESCRIPTION "Research-oriented visual-inertial odometry"
  HOMEPAGE_URL "https://github.com/Ju-yzp/Tassel"
  LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(TasselProjectOptions)
include(TasselDependencies)

tassel_setup_project_options()
tassel_configure_dependencies()

option(TASSEL_WITH_RTABMAP "Build the RTAB-Map backend" ON)
option(TASSEL_WITH_RTABMAP_VIEWER "Build the RTAB-Map DatabaseViewer" OFF)
if(TASSEL_WITH_RTABMAP)
  set(BUILD_APP
      OFF
      CACHE BOOL "Disable RTAB-Map applications" FORCE)
  if(TASSEL_WITH_RTABMAP_VIEWER)
    set(BUILD_TOOLS
        ON
        CACHE BOOL "Build RTAB-Map tools" FORCE)
    set(WITH_QT
        ON
        CACHE BOOL "Enable RTAB-Map Qt support" FORCE)
    set(RTABMAP_EMBEDDED_MINIMAL_OPENCV
        OFF
        CACHE BOOL "Use full OpenCV for RTAB-Map GUI" FORCE)
    # The viewer is an in-tree executable; do not export RTAB-Map targets that
    # depend on Tassel's private specialized Ceres target.
    set(RTABMAP_EMBEDDED_NO_EXPORT
        ON
        CACHE BOOL "Disable RTAB-Map package exports" FORCE)
  else()
    set(BUILD_TOOLS
        OFF
        CACHE BOOL "Disable RTAB-Map tools" FORCE)
    set(WITH_QT
        OFF
        CACHE BOOL "Disable RTAB-Map Qt support" FORCE)
    set(RTABMAP_EMBEDDED_MINIMAL_OPENCV
        ON
        CACHE BOOL "Use Tassel OpenCV subset" FORCE)
    set(RTABMAP_EMBEDDED_NO_EXPORT
        ON
        CACHE BOOL "Disable embedded RTAB-Map exports" FORCE)
  endif()
  set(BUILD_EXAMPLES
      OFF
      CACHE BOOL "Disable RTAB-Map examples" FORCE)
  set(BUILD_DOCUMENTATION
      OFF
      CACHE BOOL "Disable RTAB-Map documentation" FORCE)
  set(WITH_CERES
      ON
      CACHE BOOL "Use Tassel Ceres in RTAB-Map" FORCE)
  add_subdirectory(third_party/rtabmap EXCLUDE_FROM_ALL)
endif()

enable_testing()

add_subdirectory(tassel_utils)
if(EXISTS "${CMAKE_SOURCE_DIR}/tassel_hardware/CMakeLists.txt")
  add_subdirectory(tassel_hardware)
endif()
add_subdirectory(tassel_core)
add_subdirectory(tassel_tools)

install(FILES tassel_tools/__init__.py
        DESTINATION lib/${PROJECT_NAME}/tassel_tools)

install(
  DIRECTORY tassel_tools/viewer/
  DESTINATION lib/${PROJECT_NAME}/tassel_tools/viewer
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.json")
