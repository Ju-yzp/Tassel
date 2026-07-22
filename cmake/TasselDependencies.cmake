include_guard(GLOBAL)

function(tassel_configure_dependencies)
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local" "/usr/local")
  set(TASSEL_EIGEN_DIR
      "/usr/share/eigen3/cmake"
      CACHE PATH "Eigen3 package directory shared by Ceres and GTSAM")
  set(TASSEL_MATH_PREFIX
      "$ENV{HOME}/.local"
      CACHE PATH "Prefix containing Sophus and Ceres built against TASSEL_EIGEN_DIR")
  set(TASSEL_VISION_PREFIX
      ""
      CACHE PATH "Prefix containing a consistent OpenCV and DBoW3 build")
  if(TASSEL_MATH_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${TASSEL_MATH_PREFIX}")
  endif()
  if(TASSEL_VISION_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${TASSEL_VISION_PREFIX}")
  endif()
  set(CMAKE_PREFIX_PATH
      "${CMAKE_PREFIX_PATH}"
      PARENT_SCOPE)

  set(Eigen3_DIR
      "${TASSEL_EIGEN_DIR}"
      CACHE PATH "Eigen3 package directory" FORCE)
  find_package(Eigen3 REQUIRED NO_DEFAULT_PATH)
  if(TASSEL_VISION_PREFIX)
    set(OpenCV_DIR
        "${TASSEL_VISION_PREFIX}/lib/cmake/opencv4"
        CACHE PATH "OpenCV package directory" FORCE)
    find_package(OpenCV REQUIRED NO_DEFAULT_PATH)
    find_path(
      DBOW3_INCLUDE_DIR
      NAMES DBoW3/DBoW3.h
      HINTS "${TASSEL_VISION_PREFIX}/include"
      NO_DEFAULT_PATH REQUIRED)
    find_library(
      DBOW3_LIBRARY
      NAMES DBoW3
      HINTS "${TASSEL_VISION_PREFIX}/lib"
      NO_DEFAULT_PATH REQUIRED)
  else()
    find_package(OpenCV REQUIRED HINTS /usr/lib/x86_64-linux-gnu/cmake/opencv4
                 NO_DEFAULT_PATH)
  endif()
  find_package(spdlog REQUIRED)
  find_package(yaml-cpp REQUIRED)
  if(TASSEL_MATH_PREFIX)
    set(Sophus_DIR
        "${TASSEL_MATH_PREFIX}/share/sophus/cmake"
        CACHE PATH "Sophus package directory" FORCE)
    find_package(Sophus REQUIRED NO_DEFAULT_PATH)
    set(Ceres_DIR
        "${TASSEL_MATH_PREFIX}/lib/cmake/Ceres"
        CACHE PATH "Ceres package directory" FORCE)
    find_package(Ceres REQUIRED NO_DEFAULT_PATH)
  else()
    find_package(Sophus REQUIRED)
    find_package(Ceres REQUIRED)
  endif()
  if(NOT CERES_EIGEN_VERSION VERSION_EQUAL Eigen3_VERSION)
    message(
      FATAL_ERROR
        "Ceres Eigen ${CERES_EIGEN_VERSION} does not match project Eigen ${Eigen3_VERSION}")
  endif()
  find_package(GTSAM REQUIRED)
  find_package(fastcdr REQUIRED)

  if(NOT TARGET tassel_deps)
    add_library(tassel_deps INTERFACE)
    target_link_libraries(
      tassel_deps
      INTERFACE Eigen3::Eigen
                opencv_core
                opencv_imgproc
                opencv_imgcodecs
                opencv_highgui
                opencv_video
                opencv_calib3d
                spdlog::spdlog
                yaml-cpp
                Sophus::Sophus
                Ceres::ceres)
    target_link_libraries(tassel_deps INTERFACE gtsam)
    if(TASSEL_VISION_PREFIX)
      target_include_directories(tassel_deps INTERFACE "${DBOW3_INCLUDE_DIR}")
      target_link_libraries(
        tassel_deps INTERFACE opencv_features2d opencv_xfeatures2d
                              "${DBOW3_LIBRARY}")
    endif()
  endif()
endfunction()
