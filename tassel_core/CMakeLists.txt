file(
  GLOB
  TASSEL_CORE_SRCS
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/behavior/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frond_end/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/factor/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/marg/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/marg/schmidt/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/profiling/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver/*.cpp
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

find_path(TASSEL_VTUNE_ITT_INCLUDE_DIR ittnotify.h
          PATHS /opt/intel/oneapi/vtune/latest/sdk/include
                /opt/intel/oneapi/vtune/latest/include
                /opt/intel/oneapi/vtune/2026.3/sdk/include
                /opt/intel/oneapi/vtune/2026.3/include)
find_library(TASSEL_VTUNE_ITT_LIBRARY ittnotify
             PATHS /opt/intel/oneapi/vtune/latest/sdk/lib64
                   /opt/intel/oneapi/vtune/latest/lib64
                   /opt/intel/oneapi/vtune/2026.3/sdk/lib64
                   /opt/intel/oneapi/vtune/2026.3/lib64)
if(TASSEL_VTUNE_ITT_INCLUDE_DIR AND TASSEL_VTUNE_ITT_LIBRARY)
  target_compile_definitions(tassel_core PUBLIC TASSEL_HAS_VTUNE_ITT)
  target_include_directories(tassel_core PUBLIC ${TASSEL_VTUNE_ITT_INCLUDE_DIR})
  target_link_libraries(tassel_core PUBLIC ${TASSEL_VTUNE_ITT_LIBRARY})
  message(STATUS "VTune ITT profiling markers: ENABLED")
else()
  message(STATUS "VTune ITT profiling markers: DISABLED")
endif()

include(TasselTest)

find_package(GTest REQUIRED)

include(GoogleTest)

message(STATUS "tassel_core tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  MANUAL_TEST_NAMES
  test_euroc
  test_nori_estimator
  LINK_LIBS
  tassel_core
  tassel_hardware
  tassel_tools
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")
