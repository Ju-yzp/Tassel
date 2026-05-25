file(
  GLOB
  TASSEL_CORE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/frond_end/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/factor/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/estimator/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/state/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cam/*.cpp)
# Remove commented-out source files
list(FILTER TASSEL_CORE_SRCS EXCLUDE REGEX "vio_estimator\\.cpp$")

add_library(tassel_core SHARED ${TASSEL_CORE_SRCS})

target_link_libraries(tassel_core PUBLIC tassel_deps tassel_utils)

target_include_directories(
  tassel_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                     $<INSTALL_INTERFACE:include>)

# ── Tests ────────────────────────────────────────────────────────────────
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")
include(TasselTest)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

find_package(depthai CONFIG REQUIRED)
if(TASSEL_ENABLE_HARDWARE_TESTS)
  find_package(SyncTrigger REQUIRED)
endif()

include(GoogleTest)
enable_testing()

message(STATUS "tassel_core tests:")
# Only test_vo_estimator is active; other tests depend on old State API
# (commented out).
file(GLOB _ALL_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
set(_ACTIVE_TESTS test_vo_estimator test_camera test_visual_factor
                  test_prior_factor)
foreach(_src ${_ALL_TEST_SRCS})
  get_filename_component(_name ${_src} NAME_WE)
  if(NOT _name IN_LIST _ACTIVE_TESTS)
    continue()
  endif()

  file(READ "${_src}" _content)
  string(REGEX MATCH "gtest/gtest.h" _is_gtest "${_content}")
  string(REGEX MATCH "depthai" _is_hw "${_content}")

  add_executable(${_name} ${_src})
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${_name} PRIVATE tassel_core tassel_tools)
  if(_is_hw)
    target_link_libraries(${_name} PRIVATE depthai::core)
  endif()
  if(_is_gtest)
    target_link_libraries(${_name} PRIVATE GTest::GTest GTest::Main)
    if(_is_hw)
      gtest_discover_tests(${_name} PROPERTIES LABELS "hardware")
      message(STATUS "  ${_name}  [hardware / gtest]")
    else()
      gtest_discover_tests(${_name} PROPERTIES LABELS "unit")
      message(STATUS "  ${_name}  [unit / gtest]")
    endif()
  else()
    if(_is_hw AND NOT TASSEL_ENABLE_HARDWARE_TESTS)
      message(STATUS "  ${_name}  [hardware / bare] — skipped")
    else()
      add_test(NAME ${_name} COMMAND ${_name})
      set_tests_properties(${_name} PROPERTIES LABELS "hardware")
      message(STATUS "  ${_name}  [hardware / bare]")
    endif()
  endif()
endforeach()
