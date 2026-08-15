find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
set(TASSEL_TOOLS_SRCS viewer/viewer.cpp)
if(TASSEL_WITH_RTABMAP)
  list(APPEND TASSEL_TOOLS_SRCS rtabmap/rtabmap_backend.cpp)
endif()
add_library(tassel_tools SHARED ${TASSEL_TOOLS_SRCS})

target_link_libraries(
  tassel_tools
  PUBLIC tassel_core
         rclcpp::rclcpp
         sensor_msgs::sensor_msgs__rosidl_generator_cpp
         sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
         nav_msgs::nav_msgs__rosidl_generator_cpp
         nav_msgs::nav_msgs__rosidl_typesupport_cpp
         geometry_msgs::geometry_msgs__rosidl_generator_cpp
         geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
         tf2::tf2
         tf2_ros::tf2_ros
         tf2_geometry_msgs::tf2_geometry_msgs)

if(TASSEL_WITH_RTABMAP)
  target_link_libraries(tassel_tools PUBLIC rtabmap::core)
  target_compile_definitions(tassel_tools PUBLIC TASSEL_WITH_RTABMAP)
endif()

target_include_directories(
  tassel_tools
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

message(STATUS "tassel_tools tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  MANUAL_TEST_NAMES
  test_viewer
  LINK_LIBS
  tassel_tools
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
