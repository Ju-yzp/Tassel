add_library(
  tassel_loop SHARED
  geometric_verifier.cpp
  loop_closure.cpp
  loop_database.cpp
  loop_hypothesis_tracker.cpp
  pnp_verifier.cpp
  pose_graph.cpp
  trajectory_corrector.cpp)

target_link_libraries(tassel_loop PUBLIC tassel_deps tassel_utils)
target_include_directories(
  tassel_loop PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                     $<INSTALL_INTERFACE:include>)

add_executable(train_brief_vocabulary train_brief_vocabulary.cpp)
target_link_libraries(train_brief_vocabulary PRIVATE tassel_loop)

include(TasselTest)
find_package(GTest REQUIRED)
include(GoogleTest)

message(STATUS "tassel_loop tests:")
tassel_add_tests(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests" LINK_LIBS
                 tassel_loop INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
