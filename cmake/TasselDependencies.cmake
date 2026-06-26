include_guard(GLOBAL)

function(tassel_configure_dependencies)
  list(PREPEND CMAKE_PREFIX_PATH "/usr/local" "$ENV{HOME}/.local")
  set(CMAKE_PREFIX_PATH
      "${CMAKE_PREFIX_PATH}"
      PARENT_SCOPE)

  find_package(Eigen3 REQUIRED)
  # Use system OpenCV (same 4.5.4 as Ceres) to avoid TBB version conflict
  # between ~/.local custom OpenCV (libtbb.so.12) and system OpenCV
  # (libtbb.so.2) pulled by depthai/ROS2.
  find_package(OpenCV REQUIRED HINTS /usr/lib/x86_64-linux-gnu/cmake/opencv4
               NO_DEFAULT_PATH)
  find_package(spdlog REQUIRED)
  find_package(yaml-cpp REQUIRED)
  find_package(Sophus REQUIRED)
  find_package(Ceres REQUIRED)
  find_package(fastcdr REQUIRED)

  if(NOT TARGET tassel_deps)
    add_library(tassel_deps INTERFACE)
    target_link_libraries(
      tassel_deps
      INTERFACE Eigen3::Eigen
                opencv_core
                opencv_imgproc
                opencv_highgui
                opencv_video
                opencv_calib3d
                spdlog::spdlog
                yaml-cpp
                Sophus::Sophus
                Ceres::ceres)
  endif()
endfunction()
