# G1 Orchestrator Docker - C++ ROS2
# ==================================
# Base: ROS2 Humble on Ubuntu 22.04 (ARM64)
# Includes: orchestrator, arm_controller, audio_player, health_check

FROM ros:humble-ros-base-jammy

LABEL maintainer="SSI" \
      description="G1 Orchestrator - C++ ROS2 based robot control"

ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ============================================
# Core System Dependencies
# ============================================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ros-humble-rmw-cyclonedds-cpp \
    nlohmann-json3-dev \
    libyaml-cpp-dev \
    curl \
    wget \
    net-tools \
    iputils-ping \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python packages for audio and health check
RUN pip3 install \
    flask \
    edge-tts \
    paramiko \
    pyyaml \
    soundfile \
    scipy \
    requests \
    numpy opencv-python-headless

# ============================================
# ROS2 Workspace
# ============================================
WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src

COPY src /ros2_ws/src

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install

COPY config /ros2_ws/config

# ============================================
# Entrypoint
# ============================================
COPY docker_scripts/entrypoint.sh /ros2_ws/entrypoint.sh
RUN chmod +x /ros2_ws/entrypoint.sh

ENTRYPOINT ["/ros2_ws/entrypoint.sh"]
CMD ["ros2", "run", "g1_orchestrator", "g1_orchestrator_node"]
