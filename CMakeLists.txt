cmake_minimum_required(VERSION 3.22)
project(
  tassel
  VERSION 1.0.0
  DESCRIPTION
    "Research-oriented monocular visual-inertial SLAM with loop closure"
  HOMEPAGE_URL "https://github.com/Ju-yzp/Tassel"
  LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(TasselProjectOptions)
include(TasselDependencies)

tassel_setup_project_options()
tassel_configure_dependencies()

enable_testing()

add_subdirectory(tassel_utils)
add_subdirectory(tassel_loop)
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
