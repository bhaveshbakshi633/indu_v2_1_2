# INDU Voice-Controlled Humanoid Robot System v2.1.6

Complete voice control system for G1 humanoid robot with distributed architecture across multiple PCs.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NETWORK: 172.16.x.x                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Isaac PC   │    │    PC "b"    │    │   Main PC    │    │  G1 Robot │ │
│  │ 172.16.4.226 │    │ 172.16.4.250 │    │ 172.16.6.19  │    │172.16.2.242│ │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤    ├───────────┤ │
│  │              │    │              │    │              │    │           │ │
│  │   Ollama     │    │   Whisper    │    │  brain_v2    │    │  Docker   │ │
│  │   :11434     │    │   :8001      │    │  :8080       │    │  Compose  │ │
│  │              │    │              │    │              │    │           │ │
│  │  llama3.1:8b │    │  STT Server  │    │  Chatterbox  │    │ orchestr. │ │
│  │              │    │              │    │  :8000       │    │ arm_ctrl  │ │
│  │              │    │              │    │              │    │ http_brdg │ │
│  │              │    │              │    │              │    │  :5051    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
│  Voice Flow: Mic → brain_v2 → Whisper → Intent → Ollama → Chatterbox → G1  │
│  Action Flow: brain_v2 → HTTP :5051 → http_action_bridge → ROS2 → Robot    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components Overview

| Component | Machine | IP | Port | Purpose |
|-----------|---------|-----|------|---------|
| Ollama LLM | Isaac PC | 172.16.4.226 | 11434 | Language model inference |
| Whisper STT | PC "b" | 172.16.4.250 | 8001 | Speech-to-text |
| Chatterbox TTS | Main PC | 172.16.6.19 | 8000 | Text-to-speech |
| brain_v2 | Main PC | 172.16.6.19 | 8080 | Voice assistant server |
| G1 Audio Receiver | G1 Robot | 172.16.2.242 | 5050 | Audio playback |
| HTTP Action Bridge | G1 Robot | 172.16.2.242 | 5051 | Voice command relay |

## Directory Structure

```
indu_v2_1_2/
├── brain_v2/              # Voice assistant server (Main PC)
│   ├── server.py          # Main Flask/WebSocket server
│   ├── intent_reasoner.py # Voice command detection
│   ├── action_registry.py # Robot action definitions
│   ├── stop_fast_path.py  # Emergency stop detection
│   ├── voice_bridge.py    # ROS2 bridge (legacy)
│   ├── indu_rag.py        # RAG agent for knowledge
│   ├── config.json.example # Configuration template
│   ├── templates/         # Web UI
│   └── static/            # Static assets
│
├── g1_docker/             # G1 Robot Docker deployment
│   ├── docker-compose.yml # All services definition
│   ├── Dockerfile         # Base image build
│   ├── config/            # Robot configuration
│   └── src/               # ROS2 packages
│       ├── g1_orchestrator/     # Robot state machine
│       ├── arm_controller/      # Arm movements
│       ├── http_action_bridge/  # HTTP-to-ROS2 bridge
│       ├── audio_player/        # TTS audio output
│       ├── action_gatekeeper/   # Safety validation
│       └── ...
│
├── whisper_server/        # Whisper STT server (PC "b")
│   └── whisper_server.py
│
├── chatterbox_tts/        # Chatterbox TTS server (Main PC)
│   └── tts_server.py
│
└── scripts/               # Utility scripts
    ├── g1_services.sh     # Service manager (Main PC)
    └── launch_version.sh  # Version selector (G1)
```

---

## Setup Instructions

### Prerequisites

- Ubuntu 20.04+ on all PCs
- NVIDIA GPU on Isaac PC (for Ollama)
- Python 3.10+
- Docker 26+ on G1 robot
- All PCs on same network (172.16.x.x)

---

### 1. Isaac PC - Ollama LLM (172.16.4.226)

**SSH:** `ssh isaac@172.16.4.226` (password: `7410`)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure to listen on all interfaces
sudo tee /etc/systemd/system/ollama.service << 'EOF'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=default.target
EOF

# Reload and start
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull the model
ollama pull llama3.1:8b

# Verify
curl http://localhost:11434/api/tags
```

---

### 2. PC "b" - Whisper STT Server (172.16.4.250)

**SSH:** `ssh b@172.16.4.250` (password: `1997`)

```bash
# Create directory
mkdir -p ~/Whisper
cd ~/Whisper

# Create virtual environment
python3 -m venv whisper_venv
source whisper_venv/bin/activate

# Install dependencies
pip install flask faster-whisper numpy

# Copy whisper_server.py from this repo
# (from Main PC): scp whisper_server/whisper_server.py b@172.16.4.250:~/Whisper/

# Run server
nohup python3 whisper_server.py > /tmp/whisper.log 2>&1 &

# Verify
curl http://localhost:8001/health
```

---

### 3. Main PC - Chatterbox TTS (172.16.6.19)

```bash
# Create directory
mkdir -p ~/Music/chatterbox
cd ~/Music/chatterbox

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Chatterbox (follow official instructions at github.com/resemble-ai/chatterbox)
pip install chatterbox-tts torch torchaudio

# Copy tts_server.py from this repo
cp /path/to/repo/chatterbox_tts/tts_server.py ~/Music/chatterbox/

# Run server
nohup python3 tts_server.py > /tmp/chatterbox.log 2>&1 &

# Verify
curl http://localhost:8000/health
```

---

### 4. Main PC - brain_v2 Voice Assistant (172.16.6.19)

```bash
# Setup directory
mkdir -p ~/Downloads
cd ~/Downloads
cp -r /path/to/repo/brain_v2 indu_brain_v2_1_6
cd indu_brain_v2_1_6

# Create config from example
cp config.json.example config.json

# Edit config.json (see Configuration section below)

# Generate SSL certificates (for HTTPS microphone access)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"

# Install dependencies
pip install -r requirements.txt

# Source ROS2 (optional, for legacy voice_bridge)
source /opt/ros/humble/setup.bash

# Run server
python3 server.py
```

---

### 5. G1 Robot - Docker Services (172.16.2.242)

**SSH:** `ssh unitree@172.16.2.242` (password: `123`)

```bash
# Copy g1_docker folder to G1
# (from Main PC):
scp -r /path/to/repo/g1_docker unitree@172.16.2.242:~/deployed/v2_1_6

# On G1:
cd ~/deployed/v2_1_6

# Install docker compose v2 (if not already installed)
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-aarch64 \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Verify
docker compose version  # Should show v2.24.0

# Build Docker image (first time only, takes ~10 mins)
docker build -t g1_orchestrator:v2.13 .

# Start all services
docker compose up -d

# Verify all services running
docker compose ps
curl http://localhost:5050/health  # Audio receiver
curl http://localhost:5051/health  # HTTP action bridge
```

---

### 6. Scripts Setup (Main PC)

```bash
# Copy service manager script
mkdir -p ~/scripts
cp /path/to/repo/scripts/g1_services.sh ~/scripts/
chmod +x ~/scripts/g1_services.sh

# Test
~/scripts/g1_services.sh status all
```

---

## Configuration

### brain_v2/config.json

```json
{
  "ollama_host": "172.16.4.226",
  "ollama_port": 11434,
  "model": "llama3.1:8b",
  "stt": {
    "backend": "whisper_server",
    "whisper_host": "172.16.4.250",
    "whisper_port": 8001
  },
  "tts": {
    "backend": "chatterbox",
    "host": "127.0.0.1",
    "port": 8000
  },
  "g1_audio": {
    "enabled": true,
    "host": "172.16.2.242",
    "port": 5050,
    "gain": 1.0
  },
  "vad": {
    "threshold": 0.5,
    "silence_timeout": 1.5
  }
}
```

| Setting | Description |
|---------|-------------|
| `ollama_host` | Isaac PC IP for LLM |
| `stt.whisper_host` | PC "b" IP for STT |
| `g1_audio.host` | G1 robot IP for audio |
| `g1_audio.gain` | Audio amplification (0.5 - 2.0) |

---

## Running the System

### Start Order

1. **Isaac PC**: Ollama (auto-starts via systemd)
2. **PC "b"**: Whisper server
3. **Main PC**: Chatterbox TTS
4. **G1 Robot**: Docker services
5. **Main PC**: brain_v2 server

### Quick Start (from Main PC)

```bash
# Start all external services
~/scripts/g1_services.sh start all

# Check status
~/scripts/g1_services.sh status all

# Expected output:
#   ● Chatterbox TTS (172.16.6.19:8000) - Running
#   ● Whisper STT (172.16.4.250:8001) - Running
#   ● Ollama LLM (172.16.4.226:11434) - Running
#   ● Voice Assistant (172.16.6.19:8080) - Running
```

### Access Web Interface

Open browser: `https://172.16.6.19:8080/stream`

(Accept the self-signed certificate warning)

---

## Voice Commands

### Gestures (LOW risk - immediate execution)
| Command | Action |
|---------|--------|
| "wave", "wave at me" | Wave gesture |
| "shake hands" | Handshake gesture |
| "hug", "give me a hug" | Hug gesture |
| "high five" | High five gesture |
| "headshake" | Shake head |

### Motion (MEDIUM risk - needs "yes" confirmation)
| Command | Action |
|---------|--------|
| "walk forward" | Walk forward 2 seconds |
| "walk backward" | Walk backward 2 seconds |
| "turn left" | Turn left 1.5 seconds |
| "turn right" | Turn right 1.5 seconds |
| "stop" | Emergency stop (immediate) |

### Posture (MEDIUM risk - needs confirmation)
| Command | Action |
|---------|--------|
| "stand up" | Stand from sitting |
| "sit down" | Sit down |
| "squat" | Squat position |

### System (HIGH risk - needs confirmation)
| Command | Action |
|---------|--------|
| "initialize" | Full boot sequence |
| "ready mode" | Enter ready state (FSM 801) |
| "damp mode" | Enter damp mode (limp) |

---

## Service Management

### g1_services.sh Commands

```bash
~/scripts/g1_services.sh status all     # Check all services
~/scripts/g1_services.sh start all      # Start all services
~/scripts/g1_services.sh stop all       # Stop all services
~/scripts/g1_services.sh restart brain  # Restart voice assistant
~/scripts/g1_services.sh restart whisper # Restart Whisper STT
```

### G1 Docker Commands

```bash
# SSH to G1
ssh unitree@172.16.2.242

# Check running containers
docker compose ps

# View logs
docker logs http_action_bridge
docker logs g1_orchestrator
docker logs audio_receiver

# Restart specific service
docker compose restart http_action_bridge

# Restart all
docker compose down && docker compose up -d
```

---

## Troubleshooting

### Voice commands not working

```bash
# 1. Check all services
~/scripts/g1_services.sh status all

# 2. Check brain_v2 logs
tail -f /tmp/brain_v2.log | grep -v VAD

# 3. Test HTTP action bridge
curl -X POST http://172.16.2.242:5051/action \
  -H "Content-Type: application/json" \
  -d '{"action": "WAVE", "source": "test"}'
```

### Audio not playing on G1 speaker

```bash
# Test audio endpoint
curl http://172.16.2.242:5050/health

# Check audio_receiver logs
ssh unitree@172.16.2.242 "docker logs audio_receiver"
```

### Robot not responding to actions

```bash
# Check orchestrator logs
ssh unitree@172.16.2.242 "docker logs g1_orchestrator"

# Verify HTTP bridge is publishing to ROS2
ssh unitree@172.16.2.242 "docker logs http_action_bridge"
```

### STT not transcribing

```bash
# Check Whisper server
curl http://172.16.4.250:8001/health

# Check Whisper logs
ssh b@172.16.4.250 "tail -f /tmp/whisper.log"
```

---

## Network Requirements

| Port | Service | Protocol |
|------|---------|----------|
| 11434 | Ollama LLM | TCP |
| 8001 | Whisper STT | TCP |
| 8000 | Chatterbox TTS | TCP |
| 8080 | brain_v2 (HTTPS) | TCP |
| 5050 | G1 Audio Receiver | TCP |
| 5051 | HTTP Action Bridge | TCP |

Ensure firewall allows these ports between the machines.

---

## Version History

- **v2.1.6** - HTTP action bridge for voice commands, docker compose v2, distributed architecture
- **v2.1.5** - Voice control integration with intent reasoner
- **v2.1.4** - Action gatekeeper safety validation
- **v2.1.3** - Initial voice command support

---

## File Locations Summary

| Component | Machine | Path |
|-----------|---------|------|
| brain_v2 | Main PC | `~/Downloads/indu_brain_v2_1_6/` |
| Chatterbox | Main PC | `~/Music/chatterbox/` |
| Whisper | PC "b" | `~/Whisper/` |
| G1 Docker | G1 Robot | `~/deployed/v2_1_6/` |
| Service Script | Main PC | `~/scripts/g1_services.sh` |

---

## Credits

Developed for G1 Humanoid Robot by SS Innovations.
