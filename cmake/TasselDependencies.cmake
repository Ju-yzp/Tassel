include_guard(GLOBAL)

function(tassel_configure_dependencies)
  list(PREPEND CMAKE_PREFIX_PATH "/usr/local" "$ENV{HOME}/.local")
  set(CMAKE_PREFIX_PATH
      "${CMAKE_PREFIX_PATH}"
      PARENT_SCOPE)

  find_package(Eigen3 REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(spdlog REQUIRED)
  find_package(yaml-cpp REQUIRED)
  find_package(Sophus REQUIRED)
  find_package(TBB REQUIRED)
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
                TBB::tbb
                Ceres::ceres)
  endif()
endfunction()
