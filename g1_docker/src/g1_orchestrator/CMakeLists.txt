cmake_minimum_required(VERSION 3.5)
project(g1_orchestrator)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(orchestrator_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)

# Include directories - SDK headers
include_directories(
  include
  include/common
  include/g1
  include/nlohmann
)

set(DEPENDENCY_LIST
  rclcpp
  std_msgs
  orchestrator_msgs
  unitree_go
  unitree_hg
  unitree_api
)

# Main orchestrator node
add_executable(g1_orchestrator_node src/g1_orchestrator_node.cpp)
ament_target_dependencies(g1_orchestrator_node ${DEPENDENCY_LIST})
target_compile_features(g1_orchestrator_node PRIVATE cxx_std_17)

# Install targets
install(TARGETS
  g1_orchestrator_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
