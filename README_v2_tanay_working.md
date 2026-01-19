# v2.tanay - Shakal Face Recognition (CPU-Only)

## Working Configuration

**Image:** g1_orchestrator:v2.tanay (3.35GB)
**Base:** ros:humble-ros-base-jammy (Ubuntu 22.04)
**OpenCV:** 4.8.0 (built from source, CPU-only)

---

## Quick Start

### Start Shakal
```bash
docker run -d --rm --name shakal \
  --network host \
  --privileged \
  --device=/dev/video0 \
  --device=/dev/video1 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CUDA_VISIBLE_DEVICES= \
  -v /home/unitree/deployed/v2_tanay/models:/ros2_ws/models \
  -v /home/unitree/deployed/v2_tanay/data:/ros2_ws/data \
  -v /home/unitree/deployed/v2_tanay/src/shakal_ros/config/shakal_params.yaml:/ros2_ws/install/shakal_ros/share/shakal_ros/config/shakal_params.yaml \
  g1_orchestrator:v2.tanay \
  ros2 launch shakal_ros shakal.launch.py
```

### Stop Shakal
```bash
docker stop shakal
```

### View Logs
```bash
docker logs -f shakal
```

---

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| /shakal/names | std_msgs/String | Comma-separated detected names |
| /shakal/faces | shakal_ros/FaceArray | Full face data with bounding boxes |

### Check Face Detection
```bash
docker exec -it shakal bash
source /opt/ros/humble/setup.bash
source /cv_bridge_ws/install/setup.bash
source /ros2_ws/install/setup.bash
ros2 topic echo /shakal/names
```

### Check FPS
```bash
ros2 topic hz /shakal/names
```

---

## ROS2 Services

| Service | Description |
|---------|-------------|
| /shakal/enroll | Enroll new person |
| /shakal/remove | Remove person from database |
| /shakal/list_persons | List all enrolled persons |

### Enrollment Commands
```bash
# Inside container
source /opt/ros/humble/setup.bash
source /cv_bridge_ws/install/setup.bash
source /ros2_ws/install/setup.bash

# List enrolled persons
ros2 service call /shakal/list_persons shakal_ros/srv/ListPersons {}

# Enroll new person (camera ke saamne khade raho)
ros2 service call /shakal/enroll shakal_ros/srv/Enroll {name: PersonName, mode: capture, num_captures: 10}

# Remove person
ros2 service call /shakal/remove shakal_ros/srv/Remove {name: PersonName}
```

---

## Enrolled Persons (Current)
Suryanshu, Bhavesh, Tanay, Arya, Krishna, Rajesh

---

## Camera Configuration

**Device:** OBSBOT Tiny 2
**Path:** /dev/video0, /dev/video1
**Resolution:** 1920x1080 @ 30fps
**Config:** device_id: 0 in shakal_params.yaml

### Check Camera Devices
```bash
v4l2-ctl --list-devices
```

---

## Critical Files

| What | Path |
|------|------|
| Models | ~/deployed/v2_tanay/models/ |
| Face Database | ~/deployed/v2_tanay/data/embeddings/database.bin |
| Config | ~/deployed/v2_tanay/src/shakal_ros/config/shakal_params.yaml |
| Dockerfile | ~/deployed/v2_tanay/Dockerfile |

### Models (must match B PC)
- face_detection.onnx (YuNet) - 232KB
- face_recognition.onnx - 260MB (md5: e7d9c8d75698dc35bcd2e342c3bb42e5)

---

## Key Settings

### CPU-Only Mode
```bash
-e CUDA_VISIBLE_DEVICES=
```
GPU invisible to container, forces CPU inference.

### Volume Mounts (Required)
```bash
-v ~/deployed/v2_tanay/models:/ros2_ws/models
-v ~/deployed/v2_tanay/data:/ros2_ws/data
-v ~/deployed/v2_tanay/src/shakal_ros/config/shakal_params.yaml:/ros2_ws/install/shakal_ros/share/shakal_ros/config/shakal_params.yaml
```

---

## Build Image (if needed)

```bash
cd ~/deployed/v2_tanay
docker build -t g1_orchestrator:v2.tanay .
```

Build time: ~45 min (OpenCV from source)

---

## Troubleshooting

### Unknown instead of name
- Model mismatch - ensure face_recognition.onnx md5 matches B PC
- Person not enrolled - use enroll service

### Camera not opening
- Check device_id in shakal_params.yaml
- Verify camera path: v4l2-ctl --list-devices
- OBSBOT usually at /dev/video0

### Container won't start
- Remove old container: docker rm -f shakal
- Check if camera in use by another container

---

## Architecture

```
Camera (OBSBOT /dev/video0)
    |
    v
FaceDetectorYN (YuNet model, CPU)
    |
    v
FaceEncoder (ArcFace model, CPU)
    |
    v
Database matching (6 enrolled persons)
    |
    v
ROS2 Topics (/shakal/names, /shakal/faces)
```

---

## Date: 2026-01-19
