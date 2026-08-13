file(
  GLOB
  TASSEL_CORE_SRCS
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/behavior/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frond_end/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/factor/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/marg/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/estimator/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/state/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cam/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/initial/*.cpp)
list(APPEND TASSEL_CORE_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/evaluation/trajectory_evaluator.cpp)
add_library(tassel_core SHARED ${TASSEL_CORE_SRCS})

target_link_libraries(tassel_core PUBLIC tassel_deps tassel_utils)

target_include_directories(
  tassel_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tassel_tools>
         $<INSTALL_INTERFACE:include>)

include(TasselTest)

find_package(GTest REQUIRED)

include(GoogleTest)

message(STATUS "tassel_core tests:")
set(TASSEL_CORE_TEST_LIBS tassel_core tassel_tools)
if(TARGET tassel_hardware)
  list(APPEND TASSEL_CORE_TEST_LIBS tassel_hardware)
endif()
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  MANUAL_TEST_NAMES
  test_nori_estimator
  LINK_LIBS
  ${TASSEL_CORE_TEST_LIBS}
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_euroc evaluation/euroc.cpp)
target_include_directories(test_euroc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_euroc PRIVATE ${TASSEL_CORE_TEST_LIBS})
message(STATUS "  test_euroc  [manual / evaluation]")
