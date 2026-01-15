cmake_minimum_required(VERSION 3.18)
project(shakal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(USE_TENSORRT "Enable TensorRT for GPU inference" OFF)
option(USE_CUDA "Enable CUDA support" OFF)

# Find packages
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA enabled")
endif()

if(USE_TENSORRT)
    find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    find_library(TENSORRT_ONNX nvonnxparser HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    add_definitions(-DUSE_TENSORRT)
    message(STATUS "TensorRT enabled")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

set(CORE_SOURCES
    src/core/FaceDetector.cpp
    src/core/FaceEncoder.cpp
    src/core/FaceDatabase.cpp
    src/core/AntiSpoof.cpp
)

set(PIPELINE_SOURCES
    src/pipeline/Pipeline.cpp
)

set(UTILS_SOURCES
    src/utils/Logger.cpp
    src/utils/Timer.cpp
)

set(GPU_SOURCES
    src/gpu/CudaUtils.cpp
)

# Main executable
add_executable(shakal
    src/main.cpp
    ${CORE_SOURCES}
    ${PIPELINE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
)

target_link_libraries(shakal
    ${OpenCV_LIBS}
    yaml-cpp
)

if(USE_CUDA)
    target_link_libraries(shakal ${CUDA_LIBRARIES})
endif()

if(USE_TENSORRT)
    target_link_libraries(shakal ${TENSORRT_LIB} ${TENSORRT_ONNX})
endif()

# Enrollment tool
add_executable(enroll
    src/enroll.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
)

target_link_libraries(enroll
    ${OpenCV_LIBS}
    yaml-cpp
)

# Test executable
add_executable(test_recognition
    tests/test_recognition.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
)

target_link_libraries(test_recognition
    ${OpenCV_LIBS}
    yaml-cpp
)

# OBSBOT Tiny 2 configuration tool
set(OBSBOT_SDK_BASE "${CMAKE_SOURCE_DIR}/libdev_v2_1_0_7/libdev_v2.1.0_7")
set(OBSBOT_INCLUDE_DIR "${OBSBOT_SDK_BASE}/include")
set(OBSBOT_LIB_DIR "${OBSBOT_SDK_BASE}/linux(beta)/x86_64-release")

add_executable(obsbot_config
    tools/obsbot_config.cpp
)

target_include_directories(obsbot_config PRIVATE ${OBSBOT_INCLUDE_DIR})
target_link_directories(obsbot_config PRIVATE ${OBSBOT_LIB_DIR})
target_link_libraries(obsbot_config dev pthread)

# Set RPATH so it finds libdev.so at runtime
set_target_properties(obsbot_config PROPERTIES
    INSTALL_RPATH "${OBSBOT_LIB_DIR}"
    BUILD_RPATH "${OBSBOT_LIB_DIR}"
)
