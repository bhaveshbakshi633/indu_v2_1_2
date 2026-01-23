cmake_minimum_required(VERSION 3.5)
project(gatekeeper_msgs)

# C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Message files generate karo
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VoiceActionRequest.msg"
  "msg/GatekeeperResponse.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
