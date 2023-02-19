cmake_minimum_required(VERSION 3.0.2)
project(trajectory_generator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cppoptlib
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    cppoptlib
    std_msgs
)

include_directories(
  include
  ${cppoptlib_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_compile_options(-std=c++17)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
