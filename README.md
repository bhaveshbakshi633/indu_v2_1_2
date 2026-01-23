# Naamika Voice Assistant - G1 Robot Control System v2.1.7

## Overview

Naamika is a voice-controlled assistant system for the Unitree G1 humanoid robot. It provides:
- **Voice Recognition** (Whisper STT)
- **Natural Language Understanding** (Ollama LLM with RAG)
- **Text-to-Speech** (Chatterbox TTS)
- **Robot Motion Control** (gestures, locomotion, posture)
- **SLAM Navigation** (mapping, waypoints, autonomous navigation)
- **Face Recognition** (Shakal)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DISTRIBUTED SYSTEM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │         THIS PC (ssi)                │    │       ISAAC PC                  │   │
│  │         172.16.6.19                  │    │       172.16.4.226              │   │
│  │                                      │    │                                 │   │
│  │  ┌──────────────────────────────┐   │    │   ┌───────────────────────┐    │   │
│  │  │   Naamika Brain (server.py)  │   │    │   │   Ollama LLM Server   │    │   │
│  │  │   Port: 8080 (HTTPS)         │   │    │   │   Port: 11434         │    │   │
│  │  │                              │   │    │   │   Model: naamika:v1   │    │   │
│  │  │   - WebSocket VAD Pipeline   │   │    │   └───────────────────────┘    │   │
│  │  │   - Intent Reasoner          │──────────────────►                       │   │
│  │  │   - SLAM Client              │   │    │                                 │   │
│  │  │   - Action Dispatcher        │   │    └─────────────────────────────────┘   │
│  │  └──────────────────────────────┘   │                                          │
│  │               │                      │    ┌─────────────────────────────────┐   │
│  │               │                      │    │       NEW PC (b)                │   │
│  │  ┌──────────────────────────────┐   │    │       172.16.4.250              │   │
│  │  │   Chatterbox TTS Server      │   │    │                                 │   │
│  │  │   Port: 8000                 │   │    │   ┌───────────────────────┐    │   │
│  │  └──────────────────────────────┘   │    │   │   Whisper STT Server  │    │   │
│  │                                      │    │   │   Port: 8001          │    │   │
│  └──────────────────────────────────────┘    │   └───────────────────────┘    │   │
│                    │                          │              ▲                 │   │
│                    │                          └──────────────│─────────────────┘   │
│                    │                                         │                      │
│                    │         HTTP                            │ HTTP                 │
│                    ▼                                         │                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                           G1 ROBOT (172.16.2.242)                             │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                         DOCKER CONTAINERS                                │ │  │
│  │  │                                                                          │ │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │  │
│  │  │  │ audio_receiver  │  │ tts_audio_player│  │   startup_health_check  │ │ │  │
│  │  │  │ Port: 5050      │  │ speaking topic  │  │   External svc checker  │ │ │  │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │  │
│  │  │                                                                          │ │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │  │
│  │  │  │ g1_orchestrator │  │  arm_controller │  │   http_action_bridge    │ │ │  │
│  │  │  │ Main FSM        │  │  Arm movements  │  │   Port: 5051            │ │ │  │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │  │
│  │  │                                                                          │ │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │  │
│  │  │  │ talking_gestures│  │ status_announcer│  │       shakal            │ │ │  │
│  │  │  │ Auto gestures   │  │ Voice feedback  │  │   Face Recognition      │ │ │  │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │  │
│  │  │                                                                          │ │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      SLAM HTTP SERVER (Native)                          │ │  │
│  │  │                      Port: 5052                                          │ │  │
│  │  │                      /home/unitree/slam_control/slam_http_server.py     │ │  │
│  │  │                      systemd: slam_server.service                        │ │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                               │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Network Configuration

| Service | Machine | IP Address | Port | Protocol |
|---------|---------|------------|------|----------|
| **Voice Assistant (Brain)** | This PC (ssi) | 172.16.6.19 | 8080 | HTTPS |
| **Chatterbox TTS** | This PC (ssi) | 172.16.6.19 | 8000 | HTTP |
| **Whisper STT** | New PC (b) | 172.16.4.250 | 8001 | HTTP |
| **Ollama LLM** | Isaac PC | 172.16.4.226 | 11434 | HTTP |
| **G1 Audio Receiver** | G1 Robot | 172.16.2.242 | 5050 | HTTP |
| **G1 HTTP Action Bridge** | G1 Robot | 172.16.2.242 | 5051 | HTTP |
| **G1 SLAM Server** | G1 Robot | 172.16.2.242 | 5052 | HTTP |

---

## Complete File Structure with Absolute Paths

### Brain (This PC: /home/ssi/Downloads/naamika_brain_v2_1_7/)

```
/home/ssi/Downloads/naamika_brain_v2_1_7/
├── server.py                    # Main voice assistant server (Flask + WebSocket)
├── config.json                  # Configuration for all services
├── intent_reasoner.py           # LLM-based intent detection + pattern matching
├── action_registry.py           # Action definitions with risk levels
├── stop_fast_path.py            # Emergency STOP keyword detection
├── slam_client.py               # HTTP client for SLAM operations
├── voice_bridge.py              # ROS2 bridge (disabled - HTTP only)
├── naamika_rag.py               # RAG system with FAISS vector store
├── naamika_system_prompt.txt    # System prompt for LLM
├── naamika_knowledge_base.txt   # Knowledge base for RAG
├── chatterbox_tts_client.py     # Chatterbox TTS HTTP client
├── whisper_server.py            # Local whisper reference
├── requirements.txt             # Python dependencies
├── setup.py                     # Installation script
├── start.sh                     # Startup script
├── cert.pem                     # SSL certificate
├── key.pem                      # SSL private key
│
├── vad_pipeline/                # Voice Activity Detection
│   ├── __init__.py
│   ├── config.py                # VAD configuration
│   ├── audio_input.py           # Microphone input handler
│   ├── audio_player.py          # Audio playback
│   ├── audio_recorder.py        # Audio recording
│   ├── vad_processor.py         # WebRTC VAD processor
│   └── pipeline_manager.py      # Main VAD pipeline
│
├── templates/                   # Web interface templates
│   └── *.html
│
├── static/                      # Static web assets
│   └── *.js, *.css
│
├── naamika_vectorstore/         # FAISS vector store for RAG
│
├── slam_backup/                 # Local backup of SLAM data
│   ├── waypoints.json           # Saved waypoints
│   └── map_registry.json        # Saved maps registry
│
├── filler_audio/                # Filler sounds during processing
│
├── docs/                        # Documentation
│
├── external_servers/            # External server source code
│   ├── chatterbox/
│   │   └── tts_server.py        # Chatterbox TTS server
│   ├── whisper/
│   │   └── whisper_server.py    # Whisper STT server
│   └── slam_control/
│       ├── CMakeLists.txt       # CMake build config
│       ├── slam_control.cpp     # C++ SLAM binary (Unitree SDK)
│       └── slam_http_server.py  # Python HTTP wrapper
│
└── g1_deployed_v2_1_7/          # G1 Robot deployment files
    ├── docker-compose.yml       # Docker orchestration
    ├── config/
    │   ├── external_services.yaml
    │   └── orchestrator_config.yaml
    ├── maps/
    │   ├── waypoints.json
    │   └── map_registry.json
    └── src/                     # ROS2 packages
        ├── g1_orchestrator/     # Main FSM controller
        ├── arm_controller/      # Arm motion control
        ├── audio_player/        # Audio playback + TTS
        ├── http_action_bridge/  # HTTP to ROS2 bridge
        ├── action_gatekeeper/   # Safety validation
        ├── shakal_ros/          # Face recognition
        ├── orchestrator_msgs/   # Custom ROS2 messages
        ├── gatekeeper_msgs/     # Gatekeeper messages
        ├── unitree_api/         # Unitree API definitions
        ├── unitree_go/          # Go1 message types
        └── unitree_hg/          # HG message types
```

### G1 Robot (unitree@172.16.2.242)

```
/home/unitree/
├── deployed/
│   └── v2_1_7/                          # Main deployment (same as g1_deployed_v2_1_7)
│       ├── docker-compose.yml
│       ├── config/
│       │   ├── external_services.yaml   # External service configs
│       │   └── orchestrator_config.yaml
│       ├── maps/
│       │   ├── waypoints.json           # Saved waypoints (persistent)
│       │   └── map_registry.json        # Map metadata
│       ├── waypoints/                   # Docker volume mount
│       ├── docker_build/                # Docker-compiled binaries
│       │   └── install/
│       │       ├── g1_orchestrator/
│       │       └── audio_player/
│       └── src/                         # Source code
│
├── slam_control/                        # SLAM control binary & server
│   ├── build/
│   │   └── slam_control                 # Compiled binary
│   ├── slam_control.cpp
│   ├── slam_http_server.py
│   └── CMakeLists.txt
│
├── maps/                                # SLAM server working directory
│   ├── waypoints.json
│   └── map_registry.json
│
└── unitree_sdk2-copy/                   # Unitree SDK for SLAM
    ├── include/
    ├── lib/
    └── thirdparty/
```

### Whisper Server (b@172.16.4.250)

```
/home/b/Whisper/
├── whisper_server.py            # Whisper STT HTTP server
└── whisper_venv/                # Python virtual environment
    └── bin/python3
```

### Chatterbox TTS (ssi@172.16.6.19)

```
/home/ssi/Music/chatterbox/
├── tts_server.py                # Chatterbox TTS HTTP server
└── .venv/                       # Python virtual environment
```

### Ollama LLM (isaac@172.16.4.226)

```
/usr/share/ollama/.ollama/models/    # Ollama model storage
Ollama runs as system service: ollama serve
Model: naamika:v1 (custom fine-tuned)
```

---

## Voice Commands

### Gestures (LOW Risk - Immediate Execution)

| Command | Action | Description |
|---------|--------|-------------|
| "wave", "hello", "bye" | WAVE | Wave hand gesture |
| "shake hand", "handshake" | SHAKE_HAND | Handshake gesture |
| "hug", "give me a hug" | HUG | Hug gesture |
| "high five" | HIGH_FIVE | High five gesture |
| "namaste", "namaskar" | NAMASTE1 | Namaste gesture |
| "heart", "show love" | HEART | Heart gesture |
| "shake head", "no no" | HEADSHAKE | Head shake |

### Motion (MEDIUM Risk - Requires Confirmation)

| Command | Action | Description |
|---------|--------|-------------|
| "walk forward", "go ahead" | FORWARD | Walk forward 2 seconds |
| "go back", "walk backward" | BACKWARD | Walk backward 2 seconds |
| "turn left" | LEFT | Turn left 1.5 seconds |
| "turn right" | RIGHT | Turn right 1.5 seconds |
| "stop" | STOP | Emergency stop (fast-path) |

### Posture (MEDIUM Risk)

| Command | Action | Description |
|---------|--------|-------------|
| "stand up" | STANDUP | Stand from sitting |
| "sit down" | SIT | Sit down |
| "squat" | SQUAT | Squat position |
| "stand tall" | HIGH_STAND | Maximum height |
| "stand low" | LOW_STAND | Lower height |

### System (HIGH Risk)

| Command | Action | Description |
|---------|--------|-------------|
| "initialize", "boot up" | INIT | Initialize robot |
| "ready mode" | READY | Enter ready mode (enables arms) |
| "relax", "damp" | DAMP | Damping mode |

### SLAM Navigation

| Command | Action | Description |
|---------|--------|-------------|
| "start mapping" | START_MAPPING | Begin SLAM mapping |
| "stop mapping" | STOP_MAPPING | Stop mapping, save map |
| "stop mapping as office" | STOP_MAPPING:office | Stop mapping, save as "office" |
| "start navigation" | START_NAV | Load map, start localization |
| "start navigation home" | START_NAV:home | Load "home" map |
| "save this as kitchen" | SAVE_WAYPOINT:kitchen | Save current position |
| "go to kitchen" | GOTO_WAYPOINT:kitchen | Navigate to waypoint |
| "list waypoints" | LIST_WAYPOINTS | List all saved waypoints |
| "list maps" | LIST_MAPS | List all saved maps |
| "pause" | PAUSE_NAV | Pause navigation |
| "resume" | RESUME_NAV | Resume navigation |

---

## API Endpoints

### Brain Server (172.16.6.19:8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/stream` | WebSocket | Audio streaming + VAD |
| `/api/health` | GET | Health check |
| `/api/transcript` | POST | Process text transcript |

### G1 HTTP Action Bridge (172.16.2.242:5051)

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/action` | POST | `{"action": "WAVE"}` | Execute robot action |
| `/health` | GET | - | Health check |

### G1 SLAM Server (172.16.2.242:5052)

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/slam/start_mapping` | POST | - | Start SLAM mapping |
| `/slam/stop_mapping` | POST | `{"map_name": "office"}` | Stop mapping, save map |
| `/slam/relocate` | POST | `{"map_name": "office"}` | Load map, start navigation |
| `/slam/goto` | POST | `{"x":1,"y":0,"z":0,"qw":1,"qx":0,"qy":0,"qz":0}` | Navigate to pose |
| `/slam/goto_waypoint` | POST | `{"name": "kitchen"}` | Navigate to waypoint |
| `/slam/pause` | POST | - | Pause navigation |
| `/slam/resume` | POST | - | Resume navigation |
| `/slam/status` | GET | - | Get SLAM status |
| `/slam/pose` | GET | - | Get current robot pose |
| `/waypoint/save` | POST | `{"name": "kitchen"}` | Save current pose as waypoint |
| `/waypoint/list` | GET | - | List all waypoints |
| `/waypoint/delete` | POST | `{"name": "kitchen"}` | Delete waypoint |
| `/map/list` | GET | - | List saved maps |
| `/map/delete` | POST | `{"name": "office"}` | Delete map |

### Chatterbox TTS (172.16.6.19:8000)

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/synthesize` | POST | `{"text": "Hello"}` | Generate speech audio |
| `/health` | GET | - | Health check |

### Whisper STT (172.16.4.250:8001)

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/transcribe` | POST | Audio file | Transcribe audio to text |
| `/health` | GET | - | Health check |

### Ollama LLM (172.16.4.226:11434)

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/api/generate` | POST | `{"model": "naamika:v1", "prompt": "..."}` | Generate response |
| `/api/chat` | POST | `{"model": "naamika:v1", "messages": [...]}` | Chat completion |
| `/api/tags` | GET | - | List models |

---

## How to Start the System

### 1. Start External Services (on respective machines)

#### Chatterbox TTS (This PC)
```bash
cd ~/Music/chatterbox
source .venv/bin/activate
python tts_server.py
# Or use service manager:
~/scripts/g1_services.sh start chatterbox
```

#### Whisper STT (New PC - b@172.16.4.250)
```bash
ssh b@172.16.4.250
cd ~/Whisper
source whisper_venv/bin/activate
python whisper_server.py
# Or use service manager from This PC:
~/scripts/g1_services.sh start whisper
```

#### Ollama LLM (Isaac PC - isaac@172.16.4.226)
```bash
ssh isaac@172.16.4.226
OLLAMA_HOST=0.0.0.0:11434 ollama serve
# Or use service manager from This PC:
~/scripts/g1_services.sh start ollama
```

### 2. Start G1 Robot Services

```bash
# SSH to G1
sshpass -p '123' ssh unitree@172.16.2.242

# Start SLAM server (if not already running via systemd)
sudo systemctl start slam_server

# Start Docker containers
cd ~/deployed/v2_1_7
docker-compose up -d

# Verify all containers are running
docker ps
```

### 3. Start Brain (Voice Assistant)

```bash
cd ~/Downloads/naamika_brain_v2_1_7
python3 server.py
# Or use service manager:
~/scripts/g1_services.sh start brain
```

### 4. Access Web Interface

Open browser: **https://172.16.6.19:8080**

(Accept self-signed certificate warning)

---

## Service Manager (~/scripts/g1_services.sh)

```bash
# Start all services
~/scripts/g1_services.sh start all

# Start individual service
~/scripts/g1_services.sh start brain
~/scripts/g1_services.sh start chatterbox
~/scripts/g1_services.sh start whisper
~/scripts/g1_services.sh start ollama

# Check status
~/scripts/g1_services.sh status

# Stop services
~/scripts/g1_services.sh stop all

# Restart
~/scripts/g1_services.sh restart brain
```

---

## Systemd Services

### SLAM Server on G1 (/etc/systemd/system/slam_server.service)

```ini
[Unit]
Description=SLAM HTTP Server for G1 Robot
After=network.target

[Service]
Type=simple
User=unitree
WorkingDirectory=/home/unitree/slam_control
ExecStart=/usr/bin/python3 /home/unitree/slam_control/slam_http_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Commands:
```bash
sudo systemctl start slam_server
sudo systemctl stop slam_server
sudo systemctl status slam_server
sudo systemctl enable slam_server  # Auto-start on boot
```

---

## Docker Containers on G1

| Container | Purpose | Key Ports/Topics |
|-----------|---------|------------------|
| `tts_audio_player` | Play TTS audio on speaker | `/g1/tts/audio_output`, `/g1/tts/speaking` |
| `audio_receiver` | HTTP endpoint for audio | Port 5050 |
| `startup_health_check` | Check external services | Creates `/tmp/startup_complete` |
| `g1_orchestrator` | Main robot FSM | `/orchestrator/action_command` |
| `arm_controller` | Arm movements | API 7106, 7108 |
| `status_announcer` | Voice status feedback | Uses Chatterbox TTS |
| `shakal` | Face recognition | `/shakal/names`, `/shakal/faces` |
| `http_action_bridge` | HTTP to ROS2 actions | Port 5051 |
| `talking_gestures` | Auto gestures while speaking | Triggers explain actions |

---

## FSM States

| ID | State | Description | Arm Actions? |
|----|-------|-------------|--------------|
| 0 | ZERO_TORQUE | All motors off | No |
| 1 | DAMP | Damping mode | No |
| 2 | SQUAT | Squat position | No |
| 3 | SIT | Sitting | No |
| 4 | STANDUP | Stand from lying | No |
| 500 | START | Sport/active mode | Yes |
| 801 | READY | Ready for arm control | Yes |

---

## SLAM Navigation Workflow

### Creating a New Map

1. **Start mapping**: "start mapping"
   - Robot starts SLAM, building a map as you move it around

2. **Walk the robot around** the area you want to map

3. **Stop mapping**: "stop mapping as office"
   - Saves map to `/home/unitree/office.pcd`

### Using a Saved Map

1. **Start navigation**: "start navigation office"
   - Loads the "office" map
   - Robot localizes itself on the map

2. **Save waypoints**: Walk robot to location, then "save this as kitchen"
   - Saves current (x, y, z, quaternion) position

3. **Navigate**: "go to kitchen"
   - Robot autonomously navigates to saved waypoint

4. **Pause/Resume**: "pause" / "resume"

### Current Saved Waypoints (home map)

| Name | Position (x, y, z) |
|------|-------------------|
| home | (0, 0, 0) - origin |
| test_position_1 | (0.37, -0.95, 0) |
| alpha | (1.79, -0.33, 0) |
| beta | (3.88, -0.60, 0) |
| whiteboard | (4.28, 0.39, 0) |
| poster | (4.60, -3.50, 0) |
| cart | (3.94, -11.92, 0) |

---

## Troubleshooting

### Brain not starting
```bash
# Check if port 8080 is in use
lsof -i :8080
# Kill if needed
kill -9 <PID>
```

### SLAM server not responding
```bash
# On G1
sudo systemctl status slam_server
sudo journalctl -u slam_server -f
# Restart
sudo systemctl restart slam_server
```

### Docker containers not starting
```bash
# On G1
cd ~/deployed/v2_1_7
docker-compose logs -f
docker-compose down && docker-compose up -d
```

### Audio not playing
```bash
# Check audio_receiver is healthy
curl http://172.16.2.242:5050/health
```

### Navigation not working
- Make sure navigation mode is active: "start navigation"
- Check if localization succeeded (robot needs to be near mapped area)
- Try from a known position first

---

## SSH Access

| Machine | Command | Password |
|---------|---------|----------|
| G1 Robot | `sshpass -p '123' ssh unitree@172.16.2.242` | 123 |
| Isaac PC | `ssh isaac@172.16.4.226` | 7410 |
| New PC (b) | `ssh b@172.16.4.250` | 1997 |

Or use the alias for G1:
```bash
chintu  # Uses ~/.local/bin/chintu_ssh.exp
```

---

## Building SLAM Control Binary

On G1 (requires Unitree SDK):

```bash
cd ~/slam_control
mkdir -p build && cd build
cmake ..
make

# Test
./slam_control pose
./slam_control start_mapping
./slam_control stop_mapping /home/unitree/test.pcd
```

---

## Configuration Files

### Brain Config (/home/ssi/Downloads/naamika_brain_v2_1_7/config.json)

Key settings:
- `stt_backend`: "whisper_server"
- `tts_backend`: "chatterbox"
- `ollama_model`: "naamika:v1"
- `enable_rag`: true
- `slam.host`: "172.16.2.242"
- `slam.port`: 5052

### G1 External Services (/home/unitree/deployed/v2_1_7/config/external_services.yaml)

Defines health check endpoints and SSH commands for auto-restart.

---

## Safety Features

1. **Emergency STOP**: Say "stop", "halt", "freeze", "ruk", "bas" - immediate DAMP mode
2. **Action Whitelist**: Only 22 pre-approved actions can execute
3. **Risk Levels**:
   - LOW (gestures) - immediate
   - MEDIUM (locomotion) - needs confirmation
   - HIGH (system) - extra validation
4. **Semantic Veto**: Question words block action execution
5. **Time-based locomotion**: Hardcoded 2s forward/back, 1.5s turn - auto-stop

---

## Version History

- **v2.1.7** (2026-01-22): SLAM navigation, waypoints, namaste gesture
- **v2.1.6** (2026-01-21): Talking gestures, TTS debounce, Docker build fix
- **v2.1.5**: RAG integration, hallucination prevention
- **v2.1.4**: Voice control, intent reasoner
- **v2.1.3**: Face recognition (Shakal)

---

## License

Internal SSi Mantra use only.

---

## Contact

For issues: Create ticket in internal tracker or contact development team.
