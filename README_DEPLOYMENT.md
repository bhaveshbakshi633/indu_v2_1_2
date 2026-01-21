# NAAMIKA Brain v2.1 - Deployment Package

**Version**: 2.1
**Release Date**: 2026-01-13
**Description**: Production-ready NAAMIKA voice assistant with RAG (Retrieval Augmented Generation)

---

## 🎯 What's Included

### Core System Files
- `server.py` - Main Flask server with WebSocket support
- `naamika_rag.py` - RAG engine with FAISS vector search
- `naamika_system_prompt.txt` - NAAMIKA personality and behavior rules
- `naamika_knowledge_base.txt` - SSi medical robotics knowledge base
- `config.json` - System configuration

### Frontend
- `templates/` - HTML templates (stream.html, config.html)
- `static/` - Static assets (CSS, JS, images)

### Audio
- `filler_audio/` - Pre-generated filler audio files

### Documentation
- `docs/README.md` - Original project documentation
- `docs/QUICKSTART.md` - Quick start guide
- `README_DEPLOYMENT.md` - This deployment guide

### Dependencies
- `requirements.txt` - Python package dependencies

---

## 🚀 Quick Deployment

### Prerequisites
- Python 3.10 or higher
- Ollama installed with `gemma2:2b` model
- 4GB+ RAM
- Ubuntu/Linux (recommended) or Windows/macOS

### Installation Steps

1. **Navigate to deployment folder**
```bash
cd naamika_brain_v2_1
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Install Ollama model**
```bash
ollama pull gemma2:2b
```

5. **Run setup script** (creates vectorstore)
```bash
python setup.py
```

6. **Start the server**
```bash
python server.py
```

7. **Access the interface**
Open browser: `http://localhost:5000`

---

## ⚙️ Configuration

### Edit `config.json` to customize:

```json
{
  "stt_backend": "google",           // STT: google, whisper_server, whisper
  "tts_backend": "edge",             // TTS: edge, piper
  "ollama_model": "gemma2:2b",       // LLM model
  "ollama_base_url": "http://localhost:11434",
  "enable_rag": true,                // Enable/disable RAG
  "vad_threshold": 0.5,              // Voice activity detection sensitivity
  "silence_duration": 0.5,           // Silence before stopping recording (seconds)
  "interrupt_cooldown": 1.5          // Time before allowing interrupts (seconds)
}
```

---

## 📋 System Requirements

### Minimum Specs
- CPU: 4 cores
- RAM: 4GB
- Storage: 10GB free
- Network: Internet connection for Google STT & Edge TTS

### Recommended Specs
- CPU: 8+ cores
- RAM: 8GB+
- Storage: 20GB+ SSD
- GPU: Optional (for Whisper STT)

---

## 🔧 Updating Knowledge Base

### To update NAAMIKA's knowledge:

1. **Edit knowledge base**
```bash
nano naamika_knowledge_base.txt
```

2. **Rebuild vectorstore**
```bash
rm -rf naamika_vectorstore
python setup.py
```

3. **Restart server**
```bash
python server.py
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'naamika_rag'"
**Solution**: Make sure you're in the correct directory and venv is activated

### Issue: "Ollama connection refused"
**Solution**: Start Ollama service
```bash
ollama serve
```

### Issue: "Empty responses from NAAMIKA"
**Solution**: Check vectorstore exists
```bash
python setup.py  # Rebuilds vectorstore
```

### Issue: "STT not working"
**Solution**: Check microphone permissions and config.json STT backend

---

## 📁 File Structure

```
naamika_brain_v2_1/
├── server.py                      # Main server
├── naamika_rag.py                    # RAG engine
├── naamika_system_prompt.txt         # NAAMIKA personality
├── naamika_knowledge_base.txt        # Knowledge base
├── config.json                    # Configuration
├── requirements.txt               # Dependencies
├── setup.py                       # Setup script
├── README_DEPLOYMENT.md           # This file
├── templates/
│   ├── stream.html               # Voice interface
│   └── config.html               # Config interface
├── static/
│   └── [CSS, JS, images]
├── filler_audio/
│   └── [Audio files]
├── docs/
│   ├── README.md
│   └── QUICKSTART.md
└── naamika_vectorstore/             # Created on first run
    └── [FAISS index files]
```

---

## 🔐 Security Notes

1. **API Keys**: Store Google STT credentials securely
2. **Network**: Use HTTPS in production (not HTTP)
3. **Firewall**: Restrict access to port 5000
4. **Updates**: Keep dependencies updated regularly

---

## 📊 Performance Optimization

### For Better Performance:
1. Use local Whisper STT instead of Google API
2. Enable GPU acceleration for Whisper
3. Use faster Ollama models (gemma2:2b is optimized for speed)
4. Increase `chunk_size` in naamika_rag.py for faster retrieval
5. Use SSD storage for vectorstore

---

## 🎤 Voice Interaction Flow

```
User Speaks → VAD Detects → STT Transcribes → RAG Retrieves Context
    ↓
LLM Generates Response → TTS Synthesizes → Audio Playback
    ↓
User Can Interrupt Anytime (Interrupt Detection Active)
```

---

## 📞 Support

For issues, refer to:
- `docs/README.md` - Detailed documentation
- `docs/QUICKSTART.md` - Quick start guide
- GitHub Issues (if applicable)

---

## 🔄 Version History

### v2.1 (2026-01-13)
- Fixed conversation loop issues
- Added surgery workflow knowledge
- Improved NAAMIKA identity distinction
- Added founder credit information
- Removed word count constraints
- Enhanced conversation flow handling

---

## 📝 License

[Add your license information here]

---

## 🙏 Credits

Developed for SSi (SS Innovations)
Knowledge base based on SSi Mantra surgical robotic system documentation
