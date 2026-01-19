cmake_minimum_required(VERSION 3.5)
project(shakal_ros)

# C++17 standard (ROS2 Foxy compatible)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================
# Find ROS2 packages
# ============================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ============================================
# Find OpenCV (prefer custom build path)
# ============================================
find_package(OpenCV REQUIRED
  HINTS /usr/local/lib/cmake/opencv4
)

# ============================================
# Find yaml-cpp
# ============================================
find_package(yaml-cpp REQUIRED)

# ============================================
# OBSBOT SDK paths
# ============================================

# Detect architecture for OBSBOT lib
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
else()
endif()

# ============================================
# Message and Service Generation
# ============================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Face.msg"
  "msg/FaceArray.msg"
  "srv/Enroll.srv"
  "srv/Remove.srv"
  "srv/ListPersons.srv"
  "srv/SetFov.srv"
  "srv/SetZoom.srv"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

# ============================================
# Core Library (shared by all nodes)
# ============================================
set(CORE_SOURCES
  lib/core/FaceDetector.cpp
  lib/core/FaceEncoder.cpp
  lib/core/FaceDatabase.cpp
  lib/core/AntiSpoof.cpp
  lib/utils/Logger.cpp
  lib/utils/Timer.cpp
  lib/gpu/CudaUtils.cpp
  # Inference backend abstraction
  lib/inference/BackendFactory.cpp
  lib/inference/OpenCVBackend.cpp
  lib/inference/HardwareDetector.cpp
)

# TensorRT support (optional, enable with -DUSE_TENSORRT=ON)
option(USE_TENSORRT "Enable TensorRT backend support" OFF)
if(USE_TENSORRT)
  find_package(CUDA REQUIRED)
  find_library(NVINFER_LIB nvinfer)
  find_library(NVONNXPARSER_LIB nvonnxparser)
  if(NVINFER_LIB AND NVONNXPARSER_LIB)
    list(APPEND CORE_SOURCES
      lib/inference/TensorRTBackend.cpp
      lib/inference/TensorRTEngine.cpp
    )
    add_definitions(-DUSE_TENSORRT)
    message(STATUS "TensorRT support enabled")
  else()
    message(WARNING "TensorRT libraries not found, disabling TensorRT backend")
  endif()
endif()

add_library(shakal_core SHARED ${CORE_SOURCES})
target_include_directories(shakal_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shakal_core
  ${OpenCV_LIBS}
  yaml-cpp
  dl  # For dlopen in HardwareDetector
)

# Link TensorRT if enabled
if(USE_TENSORRT AND NVINFER_LIB AND NVONNXPARSER_LIB)
  target_include_directories(shakal_core PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(shakal_core ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${CUDA_LIBRARIES})
endif()

# ============================================
# Face Recognition Node
# ============================================
add_executable(face_recognition_node
  src/face_recognition_node.cpp
)
target_include_directories(face_recognition_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
target_link_libraries(face_recognition_node
  shakal_core
  ${OpenCV_LIBS}
  yaml-cpp
)
ament_target_dependencies(face_recognition_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)
rosidl_target_interfaces(face_recognition_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# ============================================
# Enrollment Service Node
# ============================================
add_executable(enrollment_service_node
  src/enrollment_service.cpp
)
target_include_directories(enrollment_service_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
target_link_libraries(enrollment_service_node
  shakal_core
  ${OpenCV_LIBS}
  yaml-cpp
)
ament_target_dependencies(enrollment_service_node
  rclcpp
  std_msgs
  cv_bridge
)
rosidl_target_interfaces(enrollment_service_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# ============================================
# ============================================
# Standalone Enrollment Tool (non-ROS)
# ============================================
add_executable(enroll_standalone
  src/enroll_standalone.cpp
)
target_include_directories(enroll_standalone PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
target_link_libraries(enroll_standalone
  shakal_core
  ${OpenCV_LIBS}
  yaml-cpp
)

# ============================================
# Install Targets
# ============================================
# Install shared library to lib/ so ROS2 finds it
install(TARGETS shakal_core
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Install executables to lib/${PROJECT_NAME}
install(TARGETS
  face_recognition_node
  enrollment_service_node
  enroll_standalone
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  lib/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp"
)

ament_package()
