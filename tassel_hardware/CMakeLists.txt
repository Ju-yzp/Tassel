add_library(
  tassel_hardware SHARED
  nori/nori_decoder.cpp
  nori/nori_device.cpp)

target_link_libraries(tassel_hardware PUBLIC tassel_deps tassel_utils)

target_include_directories(
  tassel_hardware
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include>)

include(TasselTest)
find_package(GTest REQUIRED)
include(GoogleTest)

message(STATUS "tassel_hardware tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  LINK_LIBS
  tassel_hardware
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")
