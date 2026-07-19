include_guard(GLOBAL)

# TasselTest — 从 tests/ 目录自动发现测试目标。
#
# 用法: tassel_add_tests( TEST_DIR <dir> LINK_LIBS <libraries...> INCLUDE_DIRS
# <dirs...> )
#
# 包含 gtest 的源文件注册为 GoogleTest，其他源文件注册为普通 CTest。

function(tassel_add_tests)
  cmake_parse_arguments(ARG "" "TEST_DIR" "TEST_NAMES;LINK_LIBS;INCLUDE_DIRS"
                        ${ARGN})

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
    string(REGEX MATCH "gtest" _is_gtest "${_contents}")

    add_executable(${target_name} ${test_src})
    target_include_directories(${target_name} PRIVATE ${ARG_INCLUDE_DIRS})

    if(_is_gtest)
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
