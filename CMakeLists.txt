cmake_minimum_required(VERSION 2.8.3)
project(simple_rgbd)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  relative_nav
  roscpp
  rospy
  sensor_msgs
  cv_bridge
)
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3)


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(simple_rgbd
  src/simple_rgbd.cpp
  include/simple_rgbd/simple_rgbd.h
)

target_link_libraries(simple_rgbd
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_dependencies(simple_rgbd
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  relative_nav_generate_msgs_cpp
)
