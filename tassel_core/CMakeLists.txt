file(
  GLOB
  TASSEL_CORE_SRCS
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/frond_end/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/factor/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/marg/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/estimator/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/state/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cam/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/initial/*.cpp)
add_library(tassel_core SHARED ${TASSEL_CORE_SRCS})

target_link_libraries(tassel_core PUBLIC tassel_deps tassel_utils)

target_include_directories(
  tassel_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tassel_tools>
         $<INSTALL_INTERFACE:include>)

include(TasselTest)

find_package(GTest REQUIRED)
if(TASSEL_ENABLE_HARDWARE_TESTS)
  find_package(depthai CONFIG REQUIRED)
  find_package(SyncTrigger REQUIRED)
endif()

include(GoogleTest)

message(STATUS "tassel_core tests:")
set(TASSEL_CORE_ACTIVE_TESTS
    test_estimator
    test_camera
    test_visual_factor
    test_prior_factor
    test_landmark_block
    test_imu_factor
    test_two_view_factor
    test_reprojection_factor)
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  TEST_NAMES
  ${TASSEL_CORE_ACTIVE_TESTS}
  LINK_LIBS
  tassel_core
  tassel_tools
  HARDWARE_LIBS
  $<$<BOOL:${TASSEL_ENABLE_HARDWARE_TESTS}>:depthai::core>
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")
