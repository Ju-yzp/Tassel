include_guard(GLOBAL)

option(TASSEL_ENABLE_ASAN
       "Enable AddressSanitizer (leak / OOB / UAF detection)" OFF)
option(TASSEL_ENABLE_PROFILING
       "Add frame-pointer + debug symbols for perf / VTune profiling" OFF)
function(tassel_setup_project_options)
  if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
  endif()

  if(NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                                 RelWithDebInfo MinSizeRel)
    message(STATUS "Tassel build type: ${CMAKE_BUILD_TYPE}")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      message(
        WARNING
          "Debug disables compiler optimization and makes real-time VIO/Ceres runs very slow. "
          "Use -DCMAKE_BUILD_TYPE=Release for dataset and hardware runs.")
    endif()
  endif()

  set(CMAKE_CXX_STANDARD
      20
      PARENT_SCOPE)
  set(CMAKE_CXX_STANDARD_REQUIRED
      ON
      PARENT_SCOPE)
  set(CMAKE_CXX_EXTENSIONS
      OFF
      PARENT_SCOPE)
  set(CMAKE_EXPORT_COMPILE_COMMANDS
      ON
      PARENT_SCOPE)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<CONFIG:Release>:-O3>)
    message(STATUS "Release optimization: -O3")
  endif()

  if(TASSEL_ENABLE_PROFILING)
    message(STATUS "Profiling support: ENABLED  (-fno-omit-frame-pointer -g)")
    add_compile_options(-fno-omit-frame-pointer -g)
  endif()

  if(TASSEL_ENABLE_ASAN)
    message(STATUS "AddressSanitizer: ENABLED")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O)
    add_link_options(-fsanitize=address)
  endif()
endfunction()
