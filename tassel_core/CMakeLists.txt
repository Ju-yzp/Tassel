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

target_link_libraries(tassel_core PUBLIC tassel_deps tassel_loop tassel_utils)

target_include_directories(
  tassel_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tassel_tools>
         $<INSTALL_INTERFACE:include>)

include(TasselTest)

find_package(GTest REQUIRED)

include(GoogleTest)

message(STATUS "tassel_core tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  LINK_LIBS
  tassel_core
  tassel_loop
  tassel_tools
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")
