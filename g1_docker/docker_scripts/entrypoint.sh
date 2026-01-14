#!/bin/bash
# G1 Orchestrator Entrypoint
# Source ROS2 workspace setup karo
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Execute command jo pass kiya gaya
exec "$@"
