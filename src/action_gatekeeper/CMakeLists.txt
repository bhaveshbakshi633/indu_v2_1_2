cmake_minimum_required(VERSION 3.5)
project(action_gatekeeper)

# C++17 standard - modern C++ features ke liye
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies dhundho
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gatekeeper_msgs REQUIRED)
find_package(orchestrator_msgs REQUIRED)
find_package(unitree_api REQUIRED)

# Include directories
include_directories(include)
# nlohmann/json ke liye - g1_orchestrator se inherit
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../g1_orchestrator/include)

# Action Gatekeeper executable
add_executable(action_gatekeeper_node
  src/action_gatekeeper_node.cpp
)

# Dependencies link karo
ament_target_dependencies(action_gatekeeper_node
  rclcpp
  std_msgs
  gatekeeper_msgs
  orchestrator_msgs
  unitree_api
)

# Install targets
install(TARGETS
  action_gatekeeper_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY include/
  DESTINATION include
)

ament_package()
