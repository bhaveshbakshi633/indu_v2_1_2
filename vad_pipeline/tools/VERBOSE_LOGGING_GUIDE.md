# Verbose Logging Guide for VAD Pipeline

## Overview

Verbose logging is now enabled in `server.py` to help diagnose VAD-based speech detection and interrupt monitoring. This guide explains what each log line means.

---

## Initialization Logs

When the server starts, you'll see:

```
🤖 WebStreamingAssistant initialized:
   LLM: llama3.2:latest @ 172.16.6.19:11434
   TTS: edge (voice: hi-IN-SwaraNeural)
   VAD threshold: 0.5 (regular speech)
   Interrupt threshold: 0.95 (67% rule, 0.5s frames)
   Silence timeout: 1.5s
   TTS chunks: 50 (first), 150 (rest)
   Interrupt monitor: VADInterruptMonitor initialized

Loading Silero VAD model...
VAD model loaded. Threshold: 0.5, Frames needed: 3
Interrupt detection: 1 frames of 0.5s at 0.95 confidence
  (15 chunks per interrupt frame)
```

**What this means:**
- VAD is configured with 0.5 threshold for regular speech detection
- Interrupt detection requires 0.95 threshold (95% confidence)
- Silero VAD model loaded successfully
- Each interrupt frame is 0.5 seconds = 15 chunks of 512 samples

---

## Regular Speech Detection Logs (LISTENING State)

### Every 5th Frame (Every ~170ms)

```
  [VAD CHUNK] Frame #5 | Chunk 1/3 | VAD prob: 0.234 | is_speech: False | speech_started: False | silence: 0.0s
  [VAD CHUNK] Frame #10 | Chunk 1/3 | VAD prob: 0.456 | is_speech: False | speech_started: False | silence: 0.0s
  [VAD CHUNK] Frame #15 | Chunk 1/3 | VAD prob: 0.678 | is_speech: True | speech_started: False | silence: 0.0s
```

**Fields explained:**
- `Frame #X`: Total WebSocket chunks received (each chunk = 1365 samples = ~85ms)
- `Chunk 1/3`: Which 512-sample sub-chunk within the WebSocket chunk (1365 samples split into 3 chunks)
- `VAD prob`: Silero VAD probability (0.0-1.0)
  - < 0.5 = silence/noise
  - >= 0.5 = speech detected
- `is_speech`: Boolean result (prob >= threshold)
- `speech_started`: Whether recording has begun
- `silence`: Accumulated silence duration

### Speech Start Detection

```
🎤 Speech STARTED! (VAD avg: 0.876, max: 0.912) - Recording...
```

**What this means:**
- VAD detected speech with average probability 0.876
- Maximum probability across sub-chunks was 0.912
- Recording has begun, buffering audio

---

## Silence Detection Logs (RECORDING State)

### During Silence

```
🔇 Silence: 0.5s (need 1.5s, VAD avg: 0.123, min: 0.089)
🔇 Silence: 1.0s (need 1.5s, VAD avg: 0.098, min: 0.067)
```

**Fields explained:**
- `0.5s`: Current accumulated silence duration
- `need 1.5s`: Threshold before stopping recording
- `VAD avg`: Average VAD probability across sub-chunks (should be < 0.5)
- `min`: Minimum VAD probability (closer to 0 = more confident silence)

### Speech End Detection

```
✅ Speech ENDED! (silence=1.5s, final VAD: 0.089) - Sending to STT...
   📊 Buffer size: 34 chunks = 2.89s of audio
```

**What this means:**
- 1.5 seconds of silence detected, stopping recording
- Final VAD probability was 0.089 (confident silence)
- Recorded 34 WebSocket chunks = 2.89 seconds of audio
- Sending to Google STT for transcription

---

## Interrupt Detection Logs (SPEAKING State)

### Interrupt Monitor Startup

```
🗣️ State → SPEAKING
[VAD] Interrupt monitor starting (threshold: 0.95, 67% rule, 0.5s frames)
```

**What this means:**
- AI is speaking, playback started
- Background thread monitoring for user interrupts
- Using strict 0.95 threshold + 67% rule

### During Interrupt Monitoring

The VAD processor (in `vad_pipeline/vad_processor.py`) logs detailed frame information:

```
  [INTERRUPT CHECK] Chunk 5/15 | Current VAD: 0.234 | Frame Avg: 0.198 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 10/15 | Current VAD: 0.456 | Frame Avg: 0.312 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 15/15 | Current VAD: 0.678 | Frame Avg: 0.445 | Frames confirmed: 0/1

  [FRAME COMPLETE] Stats: avg=0.445, max=0.678, min=0.089, chunks>=0.95=0/15 (need 10)
  [FRAME FAILED] Not all chunks >= 0.95, resetting counter (was 0)
```

**Fields explained:**
- `Chunk X/15`: Which 512-sample chunk within the 0.5-second frame
- `Current VAD`: VAD probability for this specific chunk
- `Frame Avg`: Running average VAD probability across the frame
- `Frames confirmed`: How many consecutive frames passed (need 1)

**When a frame completes:**
- Shows statistics: avg, max, min probabilities
- `chunks>=0.95=X/15`: How many chunks had VAD >= 0.95
- `need 10`: Requires 10 out of 15 chunks (67% rule)
- `FRAME PASSED` if >= 10 chunks passed
- `FRAME FAILED` if < 10 chunks passed (resets counter)

### Successful Interrupt Detection

```
  [INTERRUPT CHECK] Chunk 13/15 | Current VAD: 0.978 | Frame Avg: 0.965 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 14/15 | Current VAD: 0.982 | Frame Avg: 0.968 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 15/15 | Current VAD: 0.989 | Frame Avg: 0.971 | Frames confirmed: 0/1

  [FRAME COMPLETE] Stats: avg=0.971, max=0.995, min=0.952, chunks>=0.95=14/15 (need 10)
  [FRAME PASSED] 14/15 chunks >= 0.95 (67%+ rule)! Frames confirmed: 1/1
  [INTERRUPT CONFIRMED!] 1 frames at VAD >= 0.95

🚨 [VAD INTERRUPT] CONFIRMED! (VAD: 0.971, total chunks processed: 87)
    🛑 Interrupt callback returned True - STOPPING playback
[VAD] Interrupt monitor stopping
[VAD] Monitor thread exiting (processed 87 chunks total)
```

**What this means:**
- Frame had 14 out of 15 chunks with VAD >= 0.95
- 14 >= 10 (67% threshold), so frame PASSED
- Confirmed 1 consecutive frame (need 1), interrupt confirmed!
- Playback stopped immediately
- Monitor thread shut down after processing 87 total chunks

### Additional Verbose Logs Every 10 Chunks

```
  [INTERRUPT MONITOR] Chunk #10 | VAD prob: 0.234 | Frames confirmed: 0/1 | Buffer: 5/15
  [INTERRUPT MONITOR] Chunk #20 | VAD prob: 0.456 | Frames confirmed: 0/1 | Buffer: 10/15
  [INTERRUPT MONITOR] Chunk #30 | VAD prob: 0.123 | Frames confirmed: 0/1 | Buffer: 0/15
```

**Fields explained:**
- `Chunk #X`: Total 512-sample chunks processed by interrupt monitor
- `VAD prob`: Current chunk's VAD probability
- `Frames confirmed`: Consecutive frames that passed (need 1 to trigger interrupt)
- `Buffer: X/15`: Current frame buffer size (resets to 0 after each 0.5s frame)

---

## Diagnosing Issues

### Issue: Speech not detected

**Look for:**
```
  [VAD CHUNK] Frame #50 | Chunk 1/3 | VAD prob: 0.234 | is_speech: False | speech_started: False | silence: 0.0s
  [VAD CHUNK] Frame #55 | Chunk 1/3 | VAD prob: 0.289 | is_speech: False | speech_started: False | silence: 0.0s
```

**Diagnosis:**
- VAD probabilities are < 0.5 (threshold)
- `is_speech: False` means not detecting speech

**Solutions:**
1. Speak louder or move closer to microphone
2. Lower threshold from 0.5 to 0.3:
   ```python
   self.vad_config = VADConfig(
       vad_threshold=0.3,  # More sensitive
       ...
   )
   ```
3. Check microphone permissions in browser
4. Check browser console for WebSocket errors

---

### Issue: Too many false speech detections

**Look for:**
```
🎤 Speech STARTED! (VAD avg: 0.534, max: 0.567) - Recording...
🔇 Silence: 0.5s (need 1.5s, VAD avg: 0.456, min: 0.423)
🔇 Silence: 1.0s (need 1.5s, VAD avg: 0.398, min: 0.367)
✅ Speech ENDED! (silence=1.5s, final VAD: 0.312) - Sending to STT...
```

**Diagnosis:**
- VAD probabilities hover around threshold (0.5)
- Recording triggered by background noise
- Silence detection also triggered by noise (VAD still 0.3-0.4)

**Solutions:**
1. Increase threshold from 0.5 to 0.6 or 0.7:
   ```python
   self.vad_config = VADConfig(
       vad_threshold=0.6,  # Less sensitive
       ...
   )
   ```
2. Reduce background noise
3. Use better microphone with noise cancellation

---

### Issue: Can't interrupt playback

**Look for:**
```
  [INTERRUPT MONITOR] Chunk #50 | VAD prob: 0.567 | Frames confirmed: 0/1 | Buffer: 5/15
  [INTERRUPT MONITOR] Chunk #60 | VAD prob: 0.678 | Frames confirmed: 0/1 | Buffer: 0/15

  [FRAME COMPLETE] Stats: avg=0.689, max=0.789, min=0.567, chunks>=0.95=0/15 (need 10)
  [FRAME FAILED] Not all chunks >= 0.95, resetting counter (was 0)
```

**Diagnosis:**
- VAD probabilities are high (0.6-0.7) but not >= 0.95
- No chunks passing the 0.95 threshold
- User is speaking but not loudly/clearly enough

**Solutions:**
1. Speak more clearly and loudly
2. Move closer to microphone
3. Lower interrupt threshold from 0.95 to 0.9:
   ```python
   self.vad_config = VADConfig(
       interrupt_threshold=0.9,  # Less strict
       ...
   )
   ```
4. Check if microphone input is working during playback

---

### Issue: Too many false interrupts

**Look for:**
```
  [FRAME COMPLETE] Stats: avg=0.923, max=0.967, min=0.889, chunks>=0.95=13/15 (need 10)
  [FRAME PASSED] 13/15 chunks >= 0.95 (67%+ rule)! Frames confirmed: 1/1
  [INTERRUPT CONFIRMED!] 1 frames at VAD >= 0.95

🚨 [VAD INTERRUPT] CONFIRMED! (VAD: 0.923, total chunks processed: 23)
```

**Diagnosis:**
- Interrupt triggered by non-speech (playback echo, background noise)
- VAD probabilities are very high (> 0.95) for non-speech audio

**Solutions:**
1. Already using very strict threshold (0.95)
2. Increase frames needed from 1 to 2:
   ```python
   self.vad_config = VADConfig(
       interrupt_frames_needed=2,  # Require 1.0 second of speech
       ...
   )
   ```
3. Check for audio feedback (speakers too loud, mic picks up playback)
4. Reduce speaker volume
5. Use headphones instead of speakers

---

### Issue: Recording cuts off too early

**Look for:**
```
🎤 Speech STARTED! (VAD avg: 0.876, max: 0.912) - Recording...
  [VAD CHUNK] Frame #25 | Chunk 1/3 | VAD prob: 0.789 | is_speech: True | speech_started: True | silence: 0.0s
  [VAD CHUNK] Frame #30 | Chunk 1/3 | VAD prob: 0.456 | is_speech: False | speech_started: True | silence: 0.0s
🔇 Silence: 0.5s (need 1.5s, VAD avg: 0.234, min: 0.123)
🔇 Silence: 1.0s (need 1.5s, VAD avg: 0.198, min: 0.089)
✅ Speech ENDED! (silence=1.5s, final VAD: 0.089) - Sending to STT...
   📊 Buffer size: 12 chunks = 1.02s of audio
```

**Diagnosis:**
- Very short recording (1.02 seconds)
- User paused mid-sentence, triggering silence detection

**Solutions:**
1. Speak continuously without long pauses
2. Increase silence timeout from 1.5s to 2.5s:
   ```python
   self.vad_config = VADConfig(
       silence_timeout=2.5,  # Allow longer pauses
       ...
   )
   ```

---

## Understanding VAD Probabilities

### Typical Values

**Silence/Background Noise:**
- 0.0 - 0.3: Confident silence
- 0.3 - 0.5: Ambiguous (might be very quiet speech or noise)

**Speech:**
- 0.5 - 0.7: Speech detected but not very confident
- 0.7 - 0.9: Confident speech
- 0.9 - 1.0: Very confident speech (clear, loud)

### Natural Speech Patterns

Speech is not constant! Expect VAD probabilities to fluctuate:

```
Vowels ("aaa", "ooo"):       0.9 - 1.0
Voiced consonants ("mmm"):    0.7 - 0.9
Unvoiced consonants ("sss"):  0.3 - 0.6
Plosives ("p", "t", "k"):     0.2 - 0.5 (brief dip)
Breathing between words:      0.1 - 0.3
```

This is why we use the **67% rule** for interrupts - allows for natural dips during consonants and breathing while still requiring most chunks to be high confidence.

---

## Tuning Parameters

### Current Settings (in server.py)

```python
self.vad_config = VADConfig(
    sample_rate=16000,              # Fixed (required by Silero)
    chunk_size=512,                 # Fixed (required by Silero)
    vad_threshold=0.5,              # Tune: 0.3 (sensitive) to 0.7 (strict)
    silence_timeout=1.5,            # Tune: 1.0 (fast) to 2.5 (slow speakers)
    interrupt_threshold=0.95,       # Tune: 0.85 (lenient) to 0.98 (strict)
    interrupt_frame_duration=0.5,   # Tune: 0.3 (fast) to 1.0 (slow)
    interrupt_frames_needed=1,      # Tune: 1 (fast) to 3 (strict)
    verbose=True                    # Set to False to disable verbose logs
)
```

### Recommended Tuning

**For noisy environment:**
```python
vad_threshold=0.7              # Less sensitive to noise
interrupt_threshold=0.98       # Very strict interrupt
```

**For quiet environment:**
```python
vad_threshold=0.3              # More sensitive
interrupt_threshold=0.9        # More lenient interrupt
```

**For slow speakers:**
```python
silence_timeout=2.5            # Allow longer pauses
```

**For fast/impatient users:**
```python
silence_timeout=1.0            # Quicker response
interrupt_frames_needed=1      # Faster interrupt (default)
```

---

## Disable Verbose Logging

To disable verbose logging (for production), change:

```python
self.vad_config = VADConfig(
    ...
    verbose=False  # Disable detailed logs
)
```

This will keep high-level logs (🎤 Speech STARTED, ✅ Speech ENDED, 🚨 INTERRUPT) but remove the detailed per-chunk logs.

---

## Example: Successful Detection & Interrupt

Full log sequence for a successful speech → interrupt cycle:

```
🎧 State → LISTENING (VAD-based speech detection active)
  [VAD CHUNK] Frame #5 | Chunk 1/3 | VAD prob: 0.234 | is_speech: False | speech_started: False | silence: 0.0s
  [VAD CHUNK] Frame #10 | Chunk 1/3 | VAD prob: 0.345 | is_speech: False | speech_started: False | silence: 0.0s
🎤 Speech STARTED! (VAD avg: 0.876, max: 0.912) - Recording...
  [VAD CHUNK] Frame #15 | Chunk 1/3 | VAD prob: 0.789 | is_speech: True | speech_started: True | silence: 0.0s
  [VAD CHUNK] Frame #20 | Chunk 1/3 | VAD prob: 0.234 | is_speech: False | speech_started: True | silence: 0.0s
🔇 Silence: 0.5s (need 1.5s, VAD avg: 0.123, min: 0.089)
🔇 Silence: 1.0s (need 1.5s, VAD avg: 0.098, min: 0.067)
✅ Speech ENDED! (silence=1.5s, final VAD: 0.089) - Sending to STT...
   📊 Buffer size: 34 chunks = 2.89s of audio
📝 State → TRANSCRIBING
✅ Transcription: 'Hello, can you hear me?'
🧠 Querying LLM...
✅ LLM Response: 'Yes, I can hear you! How can I help?'
🗣️ State → SPEAKING
[VAD] Interrupt monitor starting (threshold: 0.95, 67% rule, 0.5s frames)
  ▶ Playing chunk 1/3 on SERVER speakers
    📡 Audio chunk queued for VAD interrupt monitoring during speaking
  [INTERRUPT MONITOR] Chunk #10 | VAD prob: 0.234 | Frames confirmed: 0/1 | Buffer: 5/15
  [INTERRUPT CHECK] Chunk 5/15 | Current VAD: 0.978 | Frame Avg: 0.867 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 10/15 | Current VAD: 0.982 | Frame Avg: 0.923 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 15/15 | Current VAD: 0.989 | Frame Avg: 0.971 | Frames confirmed: 0/1

  [FRAME COMPLETE] Stats: avg=0.971, max=0.995, min=0.952, chunks>=0.95=14/15 (need 10)
  [FRAME PASSED] 14/15 chunks >= 0.95 (67%+ rule)! Frames confirmed: 1/1
  [INTERRUPT CONFIRMED!] 1 frames at VAD >= 0.95

🚨 [VAD INTERRUPT] CONFIRMED! (VAD: 0.971, total chunks processed: 42)
    🛑 Interrupt callback returned True - STOPPING playback
[VAD] Interrupt monitor stopping
[VAD] Monitor thread exiting (processed 42 chunks total)
🎧 State → LISTENING (ready for next utterance)
🎤 Speech STARTED! (VAD avg: 0.912, max: 0.934) - Recording...
```

Perfect cycle: Listen → Detect → Record → Transcribe → Respond → Speak → Interrupt → Listen again!
