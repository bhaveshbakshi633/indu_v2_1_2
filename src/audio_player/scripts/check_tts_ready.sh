#!/bin/bash
# tts_audio_player health check script
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 topic list 2>/dev/null | grep -q /g1/tts
