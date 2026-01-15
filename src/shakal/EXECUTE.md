# Commands

## Dependencies
```bash
sudo apt install -y cmake build-essential libyaml-cpp-dev
```

## Upgrade OpenCV (Required - 4.5.4 has bugs)
```bash
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-dev
```
```bash
cd ~ && git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git
```
```bash
cd ~/opencv && git checkout 4.9.0 && cd ~/opencv_contrib && git checkout 4.9.0
```
```bash
cd ~/opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D WITH_TBB=ON -D BUILD_TESTS=OFF -D BUILD_EXAMPLES=OFF ..
```
```bash
make -j$(nproc)
```
```bash
sudo make install && sudo ldconfig
```

## Download Models
```bash
cd /home/b/RaOne/shakal/models && wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -O face_detection.onnx
```
```bash
cd /home/b/RaOne/shakal/models && wget https://huggingface.co/FoivosPar/Arc2Face/resolve/main/arcface.onnx -O face_recognition.onnx
```

## Build (CPU)
```bash
cd /home/b/RaOne/shakal && mkdir -p build && cd build && cmake .. -DUSE_CUDA=OFF -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 && make -j$(nproc)
```

## Build (Jetson GPU)
```bash
cd /home/b/RaOne/shakal && mkdir -p build && cd build && cmake .. -DUSE_CUDA=ON -DUSE_TENSORRT=ON && make -j$(nproc)
```

## OBSBOT Camera Config (run before shakal)
```bash
cd /home/b/RaOne/shakal/build && ./obsbot_config
```

## OBSBOT Config Options
```bash
./obsbot_config --fov wide      # 86° FOV (default)
./obsbot_config --fov medium    # 78° FOV
./obsbot_config --fov narrow    # 65° FOV
./obsbot_config --zoom 1.5      # Set zoom (1.0-2.0)
./obsbot_config --info          # Show camera info only
```

## Run
```bash
cd /home/b/RaOne/shakal/build && ./obsbot_config && ./shakal -c ../config/config.yaml
```

## Run with Resolution and FPS
```bash
./shakal -c ../config/config.yaml -r 1080p           # 1920x1080 @ 30fps (default)
./shakal -c ../config/config.yaml -r 1080p -f 60     # 1920x1080 @ 60fps
./shakal -c ../config/config.yaml -r 720p            # 1280x720
./shakal -c ../config/config.yaml -r 1440p           # 2560x1440
./shakal -c ../config/config.yaml -r 2160p           # 3840x2160
```
- Resolution: `-r 720p/1080p/1440p/2160p`
- FPS: `-f <fps>` (default: 30, range: 1-120)
- Unknown faces hidden in GUI (only recognized names shown with pill badge)

## Run Headless (text output only)
```bash
./shakal -c ../config/config.yaml --headless
```
Output format: `Tanay`, `Bhavesh, Tanay`, `Unknown`
- Change-based: only prints when detected faces change
- Comma separated: multiple faces shown together
- Unknown persistence: face must stay 3 sec before tagged as "Unknown" (headless only)

## Enroll (camera)
```bash
cd /home/b/RaOne/shakal/build && ./enroll capture "Name"
```

## Enroll (folder)
```bash
cd /home/b/RaOne/shakal/build && ./enroll add "Name" /path/to/images/
```

## List enrolled
```bash
cd /home/b/RaOne/shakal/build && ./enroll list
```

## Remove person
```bash
cd /home/b/RaOne/shakal/build && ./enroll remove "Name"
```

## Check cameras
```bash
v4l2-ctl --list-devices
```

## TensorRT convert (Jetson)
```bash
cd /home/b/RaOne/shakal && python3 scripts/convert_to_tensorrt.py models/face_detection.onnx && python3 scripts/convert_to_tensorrt.py models/face_recognition.onnx
```
