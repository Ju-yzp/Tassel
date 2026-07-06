add_library(tassel_utils INTERFACE)
target_include_directories(
  tassel_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                         $<INSTALL_INTERFACE:include>)
target_link_libraries(tassel_utils INTERFACE tassel_deps)
