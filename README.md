# G1 Orchestrator v2.1 - Configuration Reference

> Last Updated: 2026-01-15 22:59
> Ye file reference ke liye hai - koi bhi config change ho to yahan check karo

---

## Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NETWORK MAP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  G1 Robot (unitree@172.16.2.242)                                │
│  ├── Docker Containers                                          │
│  │   ├── audio_receiver (port 5050)                             │
│  │   ├── tts_audio_player                                       │
│  │   ├── startup_health_check                                   │
│  │   ├── g1_orchestrator                                        │
│  │   ├── arm_controller                                         │
│  │   └── status_announcer                                       │
│  └── Password: 123                                              │
│                                                                  │
│  SSI PC (ssi@192.168.123.169)                                   │
│  ├── Chatterbox TTS (port 8000)                                 │
│  ├── Voice Assistant / brain_v2 (port 8080, HTTPS)              │
│  └── Password: Ssi@113322                                       │
│                                                                  │
│  Isaac PC (isaac@192.168.123.230)                               │
│  ├── Whisper STT (port 8001)                                    │
│  ├── Ollama LLM (port 11434)                                    │
│  └── Password: 7410                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## External Services (Health Check monitors these)

### 1. Chatterbox TTS
| Field | Value |
|-------|-------|
| Host | 192.168.123.169 |
| Port | 8000 |
| Health Endpoint | /health |
| Location | ~/Music/chatterbox |
| Start Command | `cd ~/Music/chatterbox && source .venv/bin/activate && python tts_server.py` |
| Log | /tmp/chatterbox.log |

### 2. Whisper STT
| Field | Value |
|-------|-------|
| Host | 192.168.123.230 |
| Port | 8001 |
| Health Endpoint | /health |
| Location | /home/isaac/Videos/Whisper |
| Venv | /home/isaac/Videos/Whisper/whisper_venv |
| Start Command | `whisper_venv/bin/python3 whisper_server.py` |
| Log | /tmp/whisper.log |

### 3. Ollama LLM
| Field | Value |
|-------|-------|
| Host | 192.168.123.230 |
| Port | 11434 |
| Health Endpoint | /api/tags |
| Model | llama3.1:8b |
| Systemd Service | ollama.service (enabled) |
| Config | /etc/systemd/system/ollama.service |
| IMPORTANT | OLLAMA_HOST=0.0.0.0:11434 set hona chahiye |

### 4. Voice Assistant (brain_v2)
| Field | Value |
|-------|-------|
| Host | 192.168.123.169 |
| Port | 8080 |
| Protocol | HTTPS |
| Health Endpoint | /api/health |
| Location | ~/Downloads/indu_brain_v2_1_1 |
| Start Command | `cd ~/Downloads/indu_brain_v2_1_1 && python3 server.py` |
| Log | /tmp/brain_v2.log |

---

## brain_v2 Config (192.168.123.169)

**File:** `~/Downloads/indu_brain_v2_1_1/config.json`

| Setting | Value |
|---------|-------|
| STT Backend | whisper_server |
| TTS Backend | chatterbox |
| LLM Model | llama3.1:8b |
| LLM Deployment | remote |
| LLM Remote Host | 192.168.123.230:11434 |
| Whisper Host | 192.168.123.230:8001 |
| Chatterbox Host | 127.0.0.1:8000 (local) |
| G1 Audio Host | 192.168.123.164:5050 |
| RAG Enabled | true |

---

## Docker Services (G1)

**Image:** g1_orchestrator:v2.13

### Startup Order:
1. `tts_audio_player` - ROS2 audio player
2. `audio_receiver` - HTTP audio receiver (healthcheck: port 5050)
3. `startup_health_check` - External services checker (creates /tmp/startup_complete)
4. `g1_orchestrator` - Main orchestrator (waits for health check)
5. `arm_controller` - Arm control (waits for health check)
6. `status_announcer` - Continuous status monitoring

### Environment Variables:
- `GAIN` - Audio gain (default 0.5, range 0.1-6.0)
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`

### Commands:
```bash
# Start all
sudo GAIN=2.5 docker-compose up -d

# View logs
docker logs -f startup_health_check
docker logs -f g1_orchestrator

# Restart health check
docker restart startup_health_check

# Stop all
sudo docker-compose down
```

---

## Ollama Systemd Service (Isaac PC)

**File:** `/etc/systemd/system/ollama.service`

```ini
[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Environment=OLLAMA_HOST=0.0.0.0:11434
```

### Commands:
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
sudo systemctl stop ollama

# Check binding
ss -tlnp | grep 11434
# Should show *:11434 not 127.0.0.1:11434
```

---

## Troubleshooting

### Health Check Fails
1. Check which service is DOWN in logs:
   ```bash
   docker logs startup_health_check
   ```

2. Manually test service:
   ```bash
   curl http://192.168.123.169:8000/health  # Chatterbox
   curl http://192.168.123.230:8001/health  # Whisper
   curl http://192.168.123.230:11434/api/tags  # Ollama
   curl -k https://192.168.123.169:8080/api/health  # brain_v2
   ```

### Ollama Not Accessible from G1
- Check OLLAMA_HOST=0.0.0.0:11434 in systemd service
- Restart: `sudo systemctl restart ollama`

### Voice Assistant Error
- Check brain_v2 config.json has correct LLM remote host
- LLM remote host should be 192.168.123.230 (NOT localhost)

---

## File Locations

| What | Where |
|------|-------|
| G1 Deployment | ~/deployed/v2_1 |
| External Services Config | ~/deployed/v2_1/config/external_services.yaml |
| Orchestrator Config | ~/deployed/v2_1/config/orchestrator_config.yaml |
| brain_v2 Code | ssi@192.168.123.169:~/Downloads/indu_brain_v2_1_1 |
| brain_v2 Config | ssi@192.168.123.169:~/Downloads/indu_brain_v2_1_1/config.json |
| Whisper Server | isaac@192.168.123.230:/home/isaac/Videos/Whisper |
| Chatterbox | ssi@192.168.123.169:~/Music/chatterbox |

---

## Version History
- **v2.1.4.1-backup** - 2026-01-15 23:48 - SSH timeout fix, Ollama OLLAMA_HOST & OLLAMA_MODELS env vars

- v2.1.4.0 - Current
- v2.1.3.0 - Previous

