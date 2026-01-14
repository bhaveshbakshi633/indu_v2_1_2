# Indu V2.1.2 - G1 Robot Voice Assistant System

Complete robotics voice assistant system for Unitree G1 robot with arm control, orchestration, and TTS speaker output.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LAPTOP                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Chatterbox TTS │    │    brain_v2     │    │  Whisper Server │         │
│  │  (localhost:8000)│◄──│  Flask Server   │◄──│  (STT)          │         │
│  └─────────────────┘    │  - VAD Pipeline │    └─────────────────┘         │
│                         │  - RAG Agent    │                                 │
│                         │  - Ollama LLM   │                                 │
│                         └────────┬────────┘                                 │
│                                  │ HTTP POST                                │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         G1 ROBOT (Docker)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ audio_receiver  │───▶│ tts_audio_player│───▶│   G1 Speaker    │         │
│  │ (Flask:5050)    │    │ (C++ AudioClient)│    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
│           │ ROS2 Topics                                                     │
│           ▼                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                                │
│  │  arm_controller │    │  g1_orchestrator│                                │
│  │  - Arm movements│    │  - Init/Damp    │                                │
│  │  - Teach mode   │    │  - Stand up     │                                │
│  │  - Recording    │    │  - FSM control  │                                │
│  └─────────────────┘    └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
indu_v2_1_2/
├── brain_v2/                    # Laptop - Voice Assistant
│   ├── server.py                # Main Flask server
│   ├── config.json              # Configuration
│   ├── indu_rag.py              # RAG agent (LangChain + FAISS)
│   ├── indu_knowledge_base.txt  # Knowledge base
│   ├── indu_system_prompt.txt   # LLM system prompt
│   ├── chatterbox_tts_client.py # Chatterbox TTS client
│   ├── whisper_server.py        # Whisper STT server
│   ├── vad_pipeline/            # Voice Activity Detection
│   ├── templates/               # Web UI
│   └── static/                  # Static assets
│
└── g1_docker/                   # G1 Robot - Docker deployment
    ├── Dockerfile
    ├── docker-compose.yml
    └── src/
        ├── arm_controller/      # Arm control (v2.8)
        │   ├── src/arm_controller_node.cpp
        │   └── include/arm_controller/arm_controller.hpp
        ├── g1_orchestrator/     # Robot orchestration
        │   ├── src/g1_orchestrator_node.cpp
        │   └── include/         # Unitree client APIs
        ├── audio_player/        # TTS audio bridge
        │   ├── src/tts_audio_player.cpp
        │   └── scripts/audio_receiver.py
        ├── orchestrator_msgs/   # Custom ROS2 messages
        ├── unitree_api/         # Unitree API interfaces
        ├── unitree_hg/          # Unitree HG messages
        └── unitree_go/          # Unitree Go messages
```

---

## Quick Start

### 1. G1 Robot Setup

```bash
# SSH to G1
sshpass -p '123' ssh unitree@192.168.123.164

# Navigate to deployment folder
cd ~/deployed/v2_1

# Build Docker image
sudo docker build -t g1_orchestrator:v2.10 .

# Start all services
sudo docker-compose up -d

# OR start individually:
sudo docker run -d --name g1_orchestrator --network host --privileged \
    g1_orchestrator:v2.10 ros2 run g1_orchestrator g1_orchestrator_node

sudo docker run -d --name arm_controller --network host --privileged \
    g1_orchestrator:v2.10 ros2 run arm_controller arm_controller_node

sudo docker run -d --name tts_audio_player --network host --privileged \
    g1_orchestrator:v2.10 ros2 run audio_player tts_audio_player

sudo docker run -d --name audio_receiver --network host --privileged \
    g1_orchestrator:v2.10 ros2 run audio_player audio_receiver
```

### 2. Laptop - Chatterbox TTS Server

```bash
cd ~/Music/chatterbox
source .venv/bin/activate
python tts_server.py

# Verify
curl http://localhost:8000/health
```

### 3. Laptop - Brain V2 Server

```bash
cd brain_v2
pip install -r requirements.txt
python server.py

# Open browser: http://localhost:8080
```

---

## G1 Robot Commands

### Orchestrator Commands

```bash
# Enter Docker container
sudo docker exec -it g1_orchestrator bash
source /ros2_ws/install/setup.bash

# Get FSM state
ros2 topic pub --once /orchestrator/action_command orchestrator_msgs/msg/ActionCommand \
    '{action_name: "getfsm", parameters: [], priority: 1}'

# Init robot (stand up)
ros2 topic pub --once /orchestrator/action_command orchestrator_msgs/msg/ActionCommand \
    '{action_name: "init", parameters: [], priority: 1}'

# Damp robot (safe shutdown)
ros2 topic pub --once /orchestrator/action_command orchestrator_msgs/msg/ActionCommand \
    '{action_name: "damp", parameters: [], priority: 1}'
```

### Arm Controller Commands

```bash
# Enter Docker container
sudo docker exec -it arm_controller bash
source /ros2_ws/install/setup.bash

# Init arms (stiffness ramp)
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'init_arms'"

# Move to home position
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'move_home'"

# Move to custom position
ros2 topic pub --once /arm_ctrl/command std_msgs/String \
    "data: 'move_to:{\"left\": [0.3, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0], \"right\": [0.3, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]}'"

# Stop motion
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'stop'"

# Cancel task
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'cancel'"

# Teach mode
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'enter_teach'"
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'start_recording'"
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'stop_recording'"
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'play'"
ros2 topic pub --once /arm_ctrl/command std_msgs/String "data: 'exit_teach'"
```

---

## Configuration

### brain_v2/config.json

```json
{
  "g1_audio": {
    "enabled": true,
    "host": "192.168.123.164",
    "port": 5050,
    "gain": 4.0
  },
  "tts_backend": "chatterbox",
  "stt_backend": "whisper_server",
  "tts": {
    "backend": "chatterbox",
    "chatterbox": {
      "host": "127.0.0.1",
      "port": 8000,
      "fallback": "edge",
      "fallback_voice": "en-IN-NeerjaExpressiveNeural"
    }
  },
  "stt": {
    "whisper_server": {
      "host": "172.16.4.226",
      "port": 8001
    }
  },
  "llm": {
    "model": "llama3.1:8b",
    "use_rag": true
  }
}
```

**Key Settings:**
| Setting | Description |
|---------|-------------|
| `g1_audio.gain` | Audio amplification (1.0 - 5.0) |
| `g1_audio.host` | G1 robot IP |
| `tts_backend` | TTS engine: "chatterbox" or "edge" |
| `stt_backend` | STT engine: "whisper_server" or "google" |

---

## Audio Pipeline Test

```bash
# From laptop - test TTS to G1 speaker
python3 << 'EOF'
import requests, soundfile as sf, numpy as np
from scipy import signal

# Generate with Chatterbox
resp = requests.get("http://127.0.0.1:8000/tts",
    params={"text": "Hello from Chatterbox!"}, timeout=60)
open("/tmp/test.wav", "wb").write(resp.content)

# Process audio
data, sr = sf.read("/tmp/test.wav")
if len(data.shape) > 1: data = data.mean(axis=1)
if sr != 16000: data = signal.resample(data, int(len(data) * 16000 / sr))
data = np.clip(data * 4.0, -1.0, 1.0)  # 4x gain
pcm = (data * 32767).astype(np.int16).tobytes()

# Send to G1
resp = requests.post("http://192.168.123.164:5050/play_audio", data=pcm)
print(resp.json())
EOF
```

---

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/arm_ctrl/command` | String | Arm controller commands |
| `/arm_ctrl/status` | String | Arm controller status (JSON) |
| `/orchestrator/action_command` | ActionCommand | Orchestrator commands |
| `/g1/tts/audio_output` | UInt8MultiArray | PCM audio data |
| `/g1/tts/speaking` | Bool | TTS playback status |
| `/g1/tts/stop` | Empty | Stop audio playback |

---

## Docker Management

```bash
# View all containers
sudo docker ps -a

# View logs
sudo docker logs g1_orchestrator
sudo docker logs arm_controller
sudo docker logs tts_audio_player
sudo docker logs audio_receiver

# Stop all
sudo docker stop g1_orchestrator arm_controller tts_audio_player audio_receiver

# Remove all
sudo docker rm g1_orchestrator arm_controller tts_audio_player audio_receiver

# Rebuild image
sudo docker build -t g1_orchestrator:v2.10 .
```

---

## Troubleshooting

### Arms don't move
```bash
# Check FSM state (must be 801)
ros2 topic pub --once /orchestrator/action_command orchestrator_msgs/msg/ActionCommand \
    '{action_name: "getfsm"}'

# Init robot first if FSM != 801
ros2 topic pub --once /orchestrator/action_command orchestrator_msgs/msg/ActionCommand \
    '{action_name: "init"}'
```

### Audio volume low
Edit `brain_v2/config.json`:
```json
"g1_audio": {
  "gain": 5.0
}
```

### Chatterbox not responding
```bash
curl http://localhost:8000/health
# If fails, restart:
cd ~/Music/chatterbox && source .venv/bin/activate && python tts_server.py
```

---

## Version Info

- **arm_controller**: v2.8 (teach mode, recording, playback)
- **g1_orchestrator**: v2.7
- **audio_player**: v1.0
- **brain_v2**: v2.1.2
- **Docker image**: g1_orchestrator:v2.10
