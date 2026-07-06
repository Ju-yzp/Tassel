include_guard(GLOBAL)

# TasselTest — 从 tests/ 目录自动发现测试目标并分类。
#
# 用法: tassel_add_tests( TEST_DIR     <dir> LINK_LIBS    <libraries...>
# INCLUDE_DIRS <dirs...> HARDWARE_LIBS <libraries...>   # 仅硬件测试需要的库 )
#
# 分类规则 (扫描文件内容): depthai/depthai.hpp  → 硬件集成测试  [hardware] gtest/gtest.h →
# 软件单元测试  [software / gtest] 其他                  → 裸可执行文件  [bare] (不注册为 CTest)
#
# 编译选项: TASSEL_ENABLE_HARDWARE_TESTS=ON  (默认) — 编译硬件测试
# TASSEL_ENABLE_HARDWARE_TESTS=OFF        — 跳过硬件测试，无需 depthai
#
# CTest 标签: ctest -L software   — 仅运行软件测试 ctest -L hardware   — 仅运行硬件测试

function(tassel_add_tests)
  cmake_parse_arguments(
    ARG "" "TEST_DIR" "TEST_NAMES;LINK_LIBS;INCLUDE_DIRS;HARDWARE_LIBS" ${ARGN})

  if(NOT ARG_TEST_DIR)
    message(FATAL_ERROR "tassel_add_tests: TEST_DIR is required")
  endif()

  file(GLOB TEST_SRCS CONFIGURE_DEPENDS "${ARG_TEST_DIR}/*.cpp")

  foreach(test_src ${TEST_SRCS})
    get_filename_component(target_name ${test_src} NAME_WE)

    if(ARG_TEST_NAMES AND NOT target_name IN_LIST ARG_TEST_NAMES)
      continue()
    endif()

    file(READ ${test_src} _contents)
    string(REGEX MATCH "depthai" _is_hardware "${_contents}")
    string(REGEX MATCH "gtest" _is_gtest "${_contents}")

    if(_is_hardware AND NOT TASSEL_ENABLE_HARDWARE_TESTS)
      message(
        STATUS
          "  ${target_name}  [hardware] — skipped (TASSEL_ENABLE_HARDWARE_TESTS=OFF)"
      )
      continue()
    endif()

    add_executable(${target_name} ${test_src})
    target_include_directories(${target_name} PRIVATE ${ARG_INCLUDE_DIRS})

    if(_is_hardware)
      target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBS}
                                                   ${ARG_HARDWARE_LIBS})
      if(_is_gtest)
        target_link_libraries(${target_name} PRIVATE GTest::GTest GTest::Main)
        gtest_discover_tests(${target_name} PROPERTIES LABELS "hardware")
        message(STATUS "  ${target_name}  [hardware / gtest]")
      else()
        add_test(NAME ${target_name} COMMAND ${target_name})
        set_tests_properties(${target_name} PROPERTIES LABELS "hardware")
        message(STATUS "  ${target_name}  [hardware / bare]")
      endif()
    elseif(_is_gtest)
      target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBS})
      target_link_libraries(${target_name} PRIVATE GTest::GTest GTest::Main)
      gtest_discover_tests(${target_name} PROPERTIES LABELS "software")
      message(STATUS "  ${target_name}  [software / gtest]")
    else()
      target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBS})
      add_test(NAME ${target_name} COMMAND ${target_name})
      set_tests_properties(${target_name} PROPERTIES LABELS "software")
      message(STATUS "  ${target_name}  [software / bare]")
    endif()
  endforeach()
endfunction()

# 顶层便利目标
if(NOT TARGET test_software)
  add_custom_target(
    test_software
    COMMAND ${CMAKE_CTEST_COMMAND} -L software --output-on-failure
    USES_TERMINAL)
endif()

if(NOT TARGET test_hardware)
  add_custom_target(
    test_hardware
    COMMAND ${CMAKE_CTEST_COMMAND} -L hardware --output-on-failure
    USES_TERMINAL)
endif()
