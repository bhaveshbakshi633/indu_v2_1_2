# Recording Test Guide

Follow these steps to diagnose and fix the AEC (Acoustic Echo Cancellation) issue.

## Step 1: Record TTS Only (Stay Silent)

```bash
python3 record_step1_tts_only.py
```

**What it does:**
- Generates TTS audio
- Plays it through speakers
- Records what your microphone picks up
- Saves to `test_recordings/mic_tts_only.wav`

**Your job:** STAY SILENT during the recording (let mic pick up only speakers)

---

## Step 2: Record TTS + You Speaking

```bash
python3 record_step2_tts_human.py
```

**What it does:**
- Plays the same TTS through speakers
- Records while you speak over it
- Saves to `test_recordings/mic_tts_human.wav`

**Your job:** SPEAK during the recording - say "Wait", "Stop", "Hello", interrupt the TTS

---

## Step 3: Analyze Recordings

```bash
python3 analyze_recordings.py
```

**What it does:**
- Compares TTS reference with microphone recordings
- Calculates amplitude scaling factor needed
- Tests different suppression factors
- Finds optimal parameters
- Creates visualization: `test_recordings/aec_analysis.png`
- **Gives exact code changes needed for server.py**

---

## What You'll Learn

The analysis will tell you:

1. **Amplitude Scaling Factor**: How much to multiply TTS audio before subtraction
   - Example: TTS is range [-1, 1], mic is [-32768, 32768] → need to scale by ~32000x

2. **Optimal Suppression Factor**: Best value for echo_suppression_factor
   - Tests 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
   - Finds which gives lowest residual energy

3. **Exact Code Changes**: Shows exactly what to change in server.py

---

## Example Output

```
AMPLITUDE SCALING FACTOR
⚠️  TTS needs to be scaled by: 18500.0x

TESTING ECHO SUBTRACTION
  Factor 0.5: Cleaned RMS = 450.2
  Factor 0.6: Cleaned RMS = 320.5
  Factor 0.7: Cleaned RMS = 215.3
  Factor 0.8: Cleaned RMS = 156.7  ⭐ BEST
  Factor 0.9: Cleaned RMS = 198.1
  Factor 1.0: Cleaned RMS = 289.4

⭐ Optimal suppression factor: 0.8

RECOMMENDATIONS FOR server.py:
  playback_scaled = playback_chunk * 18500.0
  audio_np = audio_np - (playback_scaled * 0.8)
```

---

## Files Created

- `test_recordings/tts_reference.mp3` - TTS audio used for testing
- `test_recordings/mic_tts_only.wav` - Mic recording (TTS only, you stayed silent)
- `test_recordings/mic_tts_human.wav` - Mic recording (TTS + you speaking)
- `test_recordings/aec_analysis.png` - Visualization of signals and AEC result

---

## Quick Start (All Steps)

```bash
# Step 1: Record TTS only (stay silent)
python3 record_step1_tts_only.py

# Step 2: Record TTS + human (speak during this)
python3 record_step2_tts_human.py

# Step 3: Analyze and get recommendations
python3 analyze_recordings.py
```

Then apply the recommended code changes to [server.py](server.py) and test!
