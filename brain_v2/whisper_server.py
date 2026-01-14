#!/usr/bin/env python3
"""
Whisper ASR Inference Server
FastAPI server for multi-language speech recognition using Whisper
Supports multiple models and GPU acceleration
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ============================================================
# CONFIGURATION
# ============================================================

MODELS_AVAILABLE = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3"
}

DEFAULT_MODEL = "medium"
DEVICE = "cuda"  # Use GPU acceleration
COMPUTE_TYPE = "float16"  # GPU-optimized
HOST = "0.0.0.0"
PORT = 8001  # Using 8001 since 8000 is occupied

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Whisper ASR Server",
    description="Multilingual Speech Recognition with GPU acceleration",
    version="1.0.0"
)

# Model cache
model_cache = {}

# ============================================================
# MODELS
# ============================================================

class TranscribeRequest(BaseModel):
    """Transcription request"""
    language: Optional[str] = None  # e.g., "en", "hi", "es"
    task: str = "transcribe"  # transcribe or translate


class TranscribeResponse(BaseModel):
    """Transcription response"""
    text: str
    language: str
    model: str
    duration_seconds: float
    processing_time_seconds: float


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_model(model_name: str = DEFAULT_MODEL):
    """Load Whisper model with caching"""
    if model_name not in MODELS_AVAILABLE:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in model_cache:
        print(f"📦 Using cached {model_name} model")
        return model_cache[model_name]

    print(f"📥 Loading {model_name} model on GPU...")
    model = WhisperModel(
        MODELS_AVAILABLE[model_name],
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    model_cache[model_name] = model
    print(f"✅ Model {model_name} loaded successfully")
    return model


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """Server info"""
    return {
        "name": "Whisper ASR Server",
        "status": "online",
        "device": DEVICE,
        "available_models": list(MODELS_AVAILABLE.keys()),
        "default_model": DEFAULT_MODEL,
        "endpoints": {
            "GET /": "Server info",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "POST /transcribe": "Transcribe audio file",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": len(model_cache),
        "timestamp": time.time()
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(MODELS_AVAILABLE.keys()),
        "default_model": DEFAULT_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "models_in_cache": list(model_cache.keys()),
        "model_info": {
            "tiny": {"size": "39M", "vram": "1GB", "accuracy": "Low"},
            "base": {"size": "74M", "vram": "2GB", "accuracy": "Good"},
            "small": {"size": "244M", "vram": "3GB", "accuracy": "Better"},
            "medium": {"size": "769M", "vram": "5GB", "accuracy": "Excellent"},
            "large-v3": {"size": "1.5B", "vram": "10GB", "accuracy": "Best"}
        }
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL, description="Model size: tiny, base, small, medium, large"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'hi', 'es')")
):
    """
    Transcribe audio file using Whisper

    Supported languages: English, Hindi, Spanish, French, German, Chinese, Japanese, etc.

    Examples:
    - English: language=en
    - Hindi: language=hi
    - Spanish: language=es
    """

    # Validate model
    if model not in MODELS_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {list(MODELS_AVAILABLE.keys())}"
        )

    try:
        # Load audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load model
        whisper_model = load_model(model)

        # Transcribe
        print(f"\n🎤 Transcribing {file.filename}...")
        start_time = time.time()

        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            task="transcribe",
            beam_size=5 if model != "tiny" else 1
        )

        # Collect results
        text = " ".join([segment.text for segment in segments]).strip()
        processing_time = time.time() - start_time
        duration = info.duration

        print(f"✅ Transcription complete!")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Language: {info.language}")
        print(f"   Text: {text[:100]}...\n")

        # Cleanup
        os.unlink(tmp_path)

        return TranscribeResponse(
            text=text,
            language=info.language,
            model=model,
            duration_seconds=float(duration),
            processing_time_seconds=float(processing_time)
        )

    except Exception as e:
        print(f"❌ Error: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# STARTUP/SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup():
    """Load default model on startup"""
    print("\n" + "=" * 70)
    print("🚀 WHISPER ASR SERVER STARTING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Compute Type: {COMPUTE_TYPE}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Server: http://{HOST}:{PORT}")
    print("=" * 70 + "\n")

    # Pre-load default model
    load_model(DEFAULT_MODEL)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("\n" + "=" * 70)
    print("🛑 WHISPER ASR SERVER SHUTTING DOWN")
    print("=" * 70 + "\n")
    model_cache.clear()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
