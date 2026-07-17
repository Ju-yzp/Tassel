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
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  LINK_LIBS
  tassel_core
  tassel_tools
  HARDWARE_LIBS
  $<$<BOOL:${TASSEL_ENABLE_HARDWARE_TESTS}>:depthai::core>
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")

# Temporarily keep the hardware estimator executable buildable without running
# it in CTest.
if(TEST test_estimator)
  set_tests_properties(test_estimator PROPERTIES DISABLED TRUE)
endif()
