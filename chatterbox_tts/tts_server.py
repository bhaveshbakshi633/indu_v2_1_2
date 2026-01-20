import io
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI(title="Chatterbox TTS Server")

# Load model on startup
model = None
DEFAULT_VOICE = "Priyanka_Chopra.wav"

@app.on_event("startup")
async def load_model():
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Chatterbox Turbo on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print("Model loaded!")

@app.get("/tts")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query(DEFAULT_VOICE, description="Path to voice reference audio"),
):
    """Generate speech from text and return audio file."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        wav = model.generate(text, audio_prompt_path=voice)

        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech_post(
    text: str,
    voice: str = DEFAULT_VOICE,
):
    """POST endpoint for longer text."""
    return await text_to_speech(text=text, voice=voice)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)