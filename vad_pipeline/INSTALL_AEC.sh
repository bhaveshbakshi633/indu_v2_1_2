#!/bin/bash
# Install WebRTC Audio Processing for Echo Cancellation

echo "Installing WebRTC Audio Processing dependencies..."

# Try pip install first (easiest if available)
pip install webrtc-audio-processing

# If that fails, we need to build from source
if [ $? -ne 0 ]; then
    echo "Pip install failed, building from source..."

    # Install build dependencies
    sudo apt-get update
    sudo apt-get install -y build-essential autoconf libtool pkg-config python3-dev

    # Clone repository
    if [ ! -d "python-webrtc-audio-processing" ]; then
        git clone --recursive https://github.com/xiongyihui/python-webrtc-audio-processing.git
    fi

    cd python-webrtc-audio-processing
    git submodule update --init --recursive

    # Build
    python3 setup.py build
    sudo python3 setup.py install

    cd ..
fi

echo "Installation complete!"
