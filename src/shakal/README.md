# Shakal - Real-time Facial Recognition System

C++ based facial recognition system with GPU acceleration and headless mode support.

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Build | ✅ Working | GPU + CPU |
| Camera | ✅ Working | OBSBOT Tiny 2, V4L2 + MJPG |
| Face Detection | ✅ Working | YuNet (CUDA backend) |
| Face Recognition | ✅ Working | ArcFace 512-dim (CUDA backend) |
| Enrollment | ✅ Working | Camera capture mode |
| GPU Acceleration | ✅ Working | OpenCV DNN CUDA |
| Headless Mode | ✅ Working | Text output only |
| OBSBOT Control | ✅ Working | FOV/Zoom via SDK |
| Anti-Spoof | ⏸️ Disabled | Code ready, needs better model |

### Requirements
- OpenCV 4.8+ with CUDA support
- NVIDIA GPU with CUDA 11.0+
- yaml-cpp

---

## Usage

### GUI Mode
```bash
./obsbot_config && ./shakal -c ../config/config.yaml
./shakal -c ../config/config.yaml -r 1080p           # 1920x1080 @ 30fps
./shakal -c ../config/config.yaml -r 1080p -f 60     # 1920x1080 @ 60fps
./shakal -c ../config/config.yaml -r 720p            # 1280x720
./shakal -c ../config/config.yaml -r 1440p           # 2560x1440
./shakal -c ../config/config.yaml -r 2160p           # 3840x2160
```
- Resolution: `-r 720p/1080p/1440p/2160p`
- FPS: `-f <fps>` (default: 30, range: 1-120)
- Unknown faces hidden in GUI (only recognized names shown)

### Headless Mode (text output)
```bash
./shakal -c ../config/config.yaml --headless
```
Output: `Tanay`, `Bhavesh, Tanay`, `Unknown`
- Change-based: prints only when faces change
- Unknown: prints after 3 sec of unrecognized face (headless only)

### OBSBOT Camera Config
```bash
./obsbot_config              # Set FOV=86° (widest), Zoom=1x
./obsbot_config --fov medium # 78° FOV
./obsbot_config --fov narrow # 65° FOV
./obsbot_config --zoom 1.5   # 1.5x zoom
```

### Enrollment
```bash
./enroll capture "Name"      # Capture from camera
./enroll add "Name" /path/   # Add from folder
./enroll list                # List enrolled
./enroll remove "Name"       # Remove person
```

---

## Tech Stack

| Component | Library/Model |
|-----------|---------------|
| Language | C++17 |
| Detection | YuNet (OpenCV Zoo) |
| Recognition | ArcFace ResNet-100 (Arc2Face) |
| Inference | OpenCV DNN (CUDA backend) |
| Camera Control | OBSBOT SDK v2.1.0.7 |
| Config | yaml-cpp |

---

## Models

| Model | File | Size | Input |
|-------|------|------|-------|
| YuNet | face_detection.onnx | 227KB | Dynamic |
| ArcFace | face_recognition.onnx | 249MB | 112x112 |

---

## Hardware

### Development Machine
- OS: Ubuntu 22.04
- GPU: NVIDIA RTX A2000 12GB
- CUDA: 12.4, cuDNN 9.17.1
- OpenCV: 4.14.0 (CUDA build)
- Camera: OBSBOT Tiny 2 (/dev/video0)

### Target
- Jetson Orin NX
- JetPack SDK
- TensorRT optimization (planned)

---

## Project Structure

```
shakal/
├── CMakeLists.txt
├── config/config.yaml
├── data/embeddings/         # Face database
├── models/                  # ONNX models
├── libdev_v2_1_0_7/        # OBSBOT SDK
├── src/
│   ├── main.cpp
│   ├── enroll.cpp
│   ├── core/               # Detection, encoding, database
│   ├── pipeline/           # Main processing pipeline
│   ├── utils/              # Logger, timer
│   └── gpu/                # CUDA utilities
├── tools/
│   └── obsbot_config.cpp   # OBSBOT camera config tool
└── tests/
```

---

## Recognition Threshold

| Threshold | Behavior |
|-----------|----------|
| 0.4 | Lenient (more false positives) |
| 0.5 | Balanced |
| 0.6 | Strict (recommended) |

---

## Enrolled Persons
- Tanay (15 embeddings)
- Arya (15 embeddings)
- Bhavesh (15 embeddings)
