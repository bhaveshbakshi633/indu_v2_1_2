# G1 Orchestrator Docker - C++ ROS2
# ==================================
# Base: ROS2 Humble on Ubuntu 22.04 (ARM64)
# Includes: orchestrator, arm_controller, audio_player, shakal

FROM ros:humble-ros-base-jammy

LABEL maintainer="SSI" \
      description="G1 Orchestrator with Shakal Face Recognition"

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
    git \
    pkg-config \
    unzip \
    # OpenCV build dependencies
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Build OpenCV 4.8.0 from source (CPU only)
# ============================================
WORKDIR /tmp
RUN wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip && \
    wget -q -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.0.zip && \
    unzip -q opencv.zip && unzip -q opencv_contrib.zip && \
    rm opencv.zip opencv_contrib.zip

RUN mkdir -p /tmp/opencv-4.8.0/build && \
    cd /tmp/opencv-4.8.0/build && \
    cmake .. \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-4.8.0/modules \
        -D WITH_CUDA=OFF \
        -D WITH_CUDNN=OFF \
        -D WITH_CUBLAS=OFF \
        -D ENABLE_FAST_MATH=ON \
        -D WITH_TBB=ON \
        -D WITH_V4L=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=OFF \
        -D WITH_GSTREAMER=OFF \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_opencv_python3=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/opencv-4.8.0 /tmp/opencv_contrib-4.8.0

# Verify OpenCV version
RUN pkg-config --modversion opencv4

# ============================================
# ROS2 cv_bridge from source (links to our OpenCV)
# ============================================
RUN apt-get update && apt-get install -y \
    ros-humble-image-transport libboost-python-dev libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# Build cv_bridge from source to link with OpenCV 4.8
WORKDIR /cv_bridge_ws
RUN mkdir -p src && cd src && git clone -b humble https://github.com/ros-perception/vision_opencv.git
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    cd /cv_bridge_ws && \
    colcon build --packages-select cv_bridge --cmake-args -DOpenCV_DIR=/usr/local/lib/cmake/opencv4

# Python packages
RUN pip3 install \
    flask \
    edge-tts \
    paramiko \
    pyyaml \
    soundfile \
    scipy \
    requests \
    numpy

# ============================================
# ROS2 Workspace
# ============================================
WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src

COPY src /ros2_ws/src

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    source /cv_bridge_ws/install/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install --cmake-args -DOpenCV_DIR=/usr/local/lib/cmake/opencv4

COPY config /ros2_ws/config
COPY models /ros2_ws/models
COPY data /ros2_ws/data

# ============================================
# Entrypoint
# ============================================
COPY docker_scripts/entrypoint.sh /ros2_ws/entrypoint.sh
RUN chmod +x /ros2_ws/entrypoint.sh

# Update entrypoint to source cv_bridge
RUN sed -i 's|source /opt/ros/humble/setup.bash|source /opt/ros/humble/setup.bash\nsource /cv_bridge_ws/install/setup.bash|' /ros2_ws/entrypoint.sh

ENTRYPOINT ["/ros2_ws/entrypoint.sh"]
CMD ["ros2", "run", "g1_orchestrator", "g1_orchestrator_node"]
