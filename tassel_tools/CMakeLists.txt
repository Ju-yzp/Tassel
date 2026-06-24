find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
add_library(tassel_tools SHARED viewer/viewer.cpp)

target_link_libraries(
  tassel_tools
  PUBLIC tassel_core
         tassel_deps
         rclcpp::rclcpp
         cv_bridge::cv_bridge
         std_msgs::std_msgs__rosidl_generator_cpp
         std_msgs::std_msgs__rosidl_typesupport_cpp
         sensor_msgs::sensor_msgs__rosidl_generator_cpp
         sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
         nav_msgs::nav_msgs__rosidl_generator_cpp
         nav_msgs::nav_msgs__rosidl_typesupport_cpp
         geometry_msgs::geometry_msgs__rosidl_generator_cpp
         geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
         visualization_msgs::visualization_msgs__rosidl_generator_cpp
         visualization_msgs::visualization_msgs__rosidl_typesupport_cpp
         tf2::tf2
         tf2_ros::tf2_ros
         tf2_geometry_msgs::tf2_geometry_msgs)

target_include_directories(
  tassel_tools
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

include(TasselTest)

find_package(GTest REQUIRED)
if(TASSEL_ENABLE_HARDWARE_TESTS)
  find_package(depthai CONFIG REQUIRED)
endif()

include(GoogleTest)

message(STATUS "tassel_tools tests:")
tassel_add_tests(
  TEST_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  LINK_LIBS
  tassel_tools
  HARDWARE_LIBS
  $<$<BOOL:${TASSEL_ENABLE_HARDWARE_TESTS}>:depthai::core>
  INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
