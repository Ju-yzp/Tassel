file(
  GLOB
  TASSEL_CORE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/frond_end/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linearization/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/marginalization/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/estimator/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/state/*.cpp)

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

if(TASSEL_ENABLE_HARDWARE_TESTS)
  find_package(depthai CONFIG REQUIRED)
  find_package(SyncTrigger REQUIRED)
endif()

include(GoogleTest)
enable_testing()

message(STATUS "tassel_core tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  LINK_LIBS
  tassel_core
  tassel_tools
  HARDWARE_LIBS
  depthai::core
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}")
