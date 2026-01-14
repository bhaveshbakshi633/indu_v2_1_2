cmake_minimum_required(VERSION 3.8)
project(audio_player)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_api REQUIRED)

# Include directories - g1_orchestrator ke headers use karenge
# Include both g1 and common directories for proper resolution
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../g1_orchestrator/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../g1_orchestrator/include/common
)

# TTS Audio Player C++ node
add_executable(tts_audio_player
  src/tts_audio_player.cpp
)

ament_target_dependencies(tts_audio_player
  rclcpp
  std_msgs
  unitree_api
)

# Install C++ targets
install(TARGETS
  tts_audio_player
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/audio_receiver.py
  scripts/announcer_utils.py
  scripts/startup_health_check.py
  scripts/status_announcer.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
