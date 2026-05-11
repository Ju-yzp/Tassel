include_guard(GLOBAL)

# TasselTest — 从 tests/ 目录自动发现测试目标。
#
# 用法: tassel_add_tests( TEST_DIR     <dir> LINK_LIBS    <libraries...>
# INCLUDE_DIRS <dirs...> )
#
# 每个 tests/<name>.cpp 生成可执行文件 <name>。 包含 <depthai/depthai.hpp>  → 硬件集成测试 包含
# <gtest/gtest.h>        → GTest 单元测试（自动发现）

function(tassel_add_tests)
  cmake_parse_arguments(ARG "" "TEST_DIR" "LINK_LIBS;INCLUDE_DIRS" ${ARGN})

  if(NOT ARG_TEST_DIR)
    message(FATAL_ERROR "tassel_add_tests: TEST_DIR is required")
  endif()

  file(GLOB TEST_SRCS "${ARG_TEST_DIR}/*.cpp")

  foreach(test_src ${TEST_SRCS})
    get_filename_component(target_name ${test_src} NAME_WE)

    file(READ ${test_src} _contents)
    string(REGEX MATCH "depthai" _is_hardware "${_contents}")
    string(REGEX MATCH "gtest" _is_gtest "${_contents}")

    add_executable(${target_name} ${test_src})
    target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBS})
    target_include_directories(${target_name} PRIVATE ${ARG_INCLUDE_DIRS})

    if(_is_hardware)
      message(STATUS "  ${target_name}  [depthai / hardware]")
    elseif(_is_gtest)
      target_link_libraries(${target_name} PRIVATE GTest::GTest GTest::Main)
      gtest_discover_tests(${target_name})
      message(STATUS "  ${target_name}  [gtest]")
    else()
      message(STATUS "  ${target_name}  [bare]")
    endif()
  endforeach()
endfunction()
