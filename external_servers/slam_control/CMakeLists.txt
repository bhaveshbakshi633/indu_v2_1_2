cmake_minimum_required(VERSION 3.5)
project(slam_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

# Unitree SDK paths
set(UNITREE_SDK_PATH "$ENV{HOME}/unitree_sdk2-copy")

include_directories(
    ${UNITREE_SDK_PATH}/include
    ${UNITREE_SDK_PATH}/thirdparty/include
    ${UNITREE_SDK_PATH}/thirdparty/include/ddscxx
)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH "aarch64")
else()
    set(ARCH "x86_64")
endif()

link_directories(
    ${UNITREE_SDK_PATH}/lib/${ARCH}
    ${UNITREE_SDK_PATH}/thirdparty/lib/${ARCH}
)

add_executable(slam_control slam_control.cpp)

target_link_libraries(slam_control
    unitree_sdk2
    ddsc
    ddscxx
    pthread
)
