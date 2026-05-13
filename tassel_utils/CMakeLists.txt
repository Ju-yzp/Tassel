add_library(tassel_utils INTERFACE)
target_include_directories(
  tassel_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(tassel_utils INTERFACE tassel_deps)

# ── Tests ────────────────────────────────────────────────────────────────
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")
include(TasselTest)

find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

message(STATUS "tassel_utils tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  LINK_LIBS
  tassel_utils
  tassel_core
  tassel_tools
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
