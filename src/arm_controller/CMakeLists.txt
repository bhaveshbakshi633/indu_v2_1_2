cmake_minimum_required(VERSION 3.8)
project(arm_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)

# Include directories
include_directories(include)

# Arm controller node executable
add_executable(arm_controller_node
  src/arm_controller_node.cpp
  src/motor_crc_hg.cpp
)

ament_target_dependencies(arm_controller_node
  rclcpp
  std_msgs
  unitree_hg
  unitree_api
)

# Install targets
install(TARGETS
  arm_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
