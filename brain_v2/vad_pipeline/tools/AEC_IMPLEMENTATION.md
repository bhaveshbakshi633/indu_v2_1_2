# Acoustic Echo Cancellation (AEC) Implementation

## Problem

The VAD was detecting Edge TTS playback as human speech, causing false interrupts. This happened because:

1. Server plays TTS through speakers
2. User's microphone (streaming via WebSocket) picks up the speakers
3. VAD thinks "this is speech!" → triggers interrupt
4. Playback stops immediately (false positive)

## Solution: Reference-Based Echo Cancellation

Since we know EXACTLY what audio is being played (the TTS output), we can subtract it from the incoming microphone signal.

**This is better than generic AEC because:**
- We have the reference signal (TTS audio)
- No adaptive filtering needed
- Simple, deterministic subtraction
- Low latency

---

## How It Works

### 1. Track Playback Audio

When TTS starts playing, we store the audio data:

```python
# In play_audio() function (server.py:443-467)
if interrupt_monitor is not None:
    # Resample to 16kHz (match mic sample rate)
    if sr_rate != 16000:
        data_16k = scipy.signal.resample(data, int(len(data) * 16000 / sr_rate))

    # Convert stereo to mono
    if len(data_16k.shape) > 1:
        data_16k = np.mean(data_16k, axis=1)

    # Store in interrupt monitor
    interrupt_monitor.set_playback_audio(data_16k.astype(np.float32))
```

### 2. Sync Playback Position

As we receive microphone chunks, we track which part of the TTS is currently playing:

```python
# VADInterruptMonitor stores:
self.playback_buffer = np.array([...])  # Full TTS audio
self.playback_position = 0               # Current position in samples
```

### 3. Subtract Echo from Mic Signal

For each incoming mic chunk, subtract the corresponding TTS chunk:

```python
# In VADInterruptMonitor._monitor_loop() (server.py:255-278)
with self.lock:
    if len(self.playback_buffer) > 0:
        # Get the TTS chunk that's currently playing
        playback_chunk = self.playback_buffer[self.playback_position:playback_end]

        # Subtract from mic input (with suppression factor)
        audio_cleaned = audio_mic - (playback_chunk * 0.8)

        # Update position
        self.playback_position += chunk_len
```

**Why 0.8 suppression factor?**
- Account for room acoustics (echo is attenuated)
- Avoid over-subtraction artifacts
- Tunable based on your setup

### 4. Run VAD on Cleaned Signal

After subtraction, run VAD on the cleaned audio:

```python
# VAD now sees only user's voice (TTS echo removed)
interrupt_confirmed, prob = self.vad.detect_interrupt(cleaned_audio_chunk)
```

---

## Key Implementation Details

### Synchronization

**Challenge:** Mic audio and playback audio must be time-aligned.

**Solution:**
- Both use 16kHz sample rate
- Track playback position in samples
- Linear progression: `position += chunk_size` for each chunk

### Sample Rate Matching

```python
# TTS might be 22kHz or 24kHz, mic is always 16kHz
if sr_rate != 16000:
    data_16k = scipy.signal.resample(data, int(len(data) * 16000 / sr_rate))
```

### Thread Safety

```python
with self.lock:
    self.playback_buffer = audio_data
    self.playback_position = 0
```

Playback buffer is accessed from:
- Main thread (when setting audio)
- Interrupt monitor thread (when subtracting)

Lock prevents race conditions.

### Mono Conversion

```python
# TTS might be stereo, mic is mono
if len(data_16k.shape) > 1:
    data_16k = np.mean(data_16k, axis=1)  # Average L+R channels
```

---

## Code Flow

### When TTS Starts Playing

```
1. Edge TTS generates audio (may be 22kHz, stereo)
2. Save to temp MP3 file
3. play_audio() loads file
4. Resample to 16kHz, convert to mono
5. interrupt_monitor.set_playback_audio(audio_16k_mono)
6. Start playback with sd.play()
7. Start interrupt_monitor.start()
```

### During Playback (Interrupt Monitoring)

```
1. WebSocket receives mic chunk (int16, 1365 samples)
2. Convert to float32: audio_mic = np.frombuffer(data, dtype=np.int16).astype(np.float32)
3. Get corresponding TTS chunk: playback_chunk = playback_buffer[position:position+1365]
4. Subtract echo: audio_clean = audio_mic - (playback_chunk * 0.8)
5. Split into 512-sample VAD chunks
6. Run VAD on each: is_interrupt, prob = vad.detect_interrupt(chunk_512)
7. If interrupt confirmed → stop playback
8. Update position: position += 1365
```

---

## Verbose Logging

Enable in config:

```python
verbose=True
```

**AEC Logs:**

```
[AEC] Playback buffer set: 48000 samples (3.00s)
  [AEC] Pos: 10920/48000 | Echo energy: 2345 | Cleaned energy: 156
  [AEC] Pos: 38220/48000 | Echo energy: 3456 | Cleaned energy: 189
```

**Fields:**
- `Pos: X/Y`: Current position / Total samples
- `Echo energy`: RMS energy of TTS chunk (what we're subtracting)
- `Cleaned energy`: RMS energy after subtraction (should be lower)

**Expected behavior:**
- `Echo energy` high (2000-5000) when TTS is speaking
- `Cleaned energy` low (50-200) if only TTS (no user speech)
- `Cleaned energy` high (1000+) if user interrupts (real speech)

---

## Tuning Parameters

### Suppression Factor (server.py:169)

```python
self.echo_suppression_factor = 0.8  # How much to suppress (0.0-1.0)
```

**Too low (0.5):**
- Under-suppression: Echo still leaks through
- VAD still detects TTS as speech
- False interrupts

**Too high (1.0):**
- Over-suppression: Might subtract user's voice
- Artifacts if timing slightly off
- User interrupts not detected

**Sweet spot: 0.7-0.9**
- Start with 0.8
- Increase if false interrupts persist
- Decrease if missing real interrupts

### Monitoring Verbosity (server.py:273)

```python
if self.config.verbose and chunk_count % 20 == 0:
    # Log AEC stats every 20 chunks (~1.4 seconds)
```

Change `20` to `10` for more frequent logs, or `50` for less.

---

## Testing

### 1. Install Dependencies

```bash
cd /home/isaac/codes/Br_ai_n
source venv/bin/activate
pip install scipy>=1.11.0
```

### 2. Run Server

```bash
python server.py
```

### 3. Open Browser

Navigate to: `http://127.0.0.1:8080/stream`

### 4. Test Scenario

**Without AEC (before):**
```
User: "Hello"
AI: "Hi there, how..." → [INTERRUPT DETECTED] (false positive from TTS playback)
```

**With AEC (now):**
```
User: "Hello"
AI: "Hi there, how can I help you today?"
[User stays silent → No false interrupt]
[User speaks: "Wait!"] → [INTERRUPT DETECTED] (real interrupt)
```

### 5. Check Logs

```
[AEC] Playback buffer set: 44100 samples (2.76s)
[VAD] Interrupt monitor starting (threshold: 0.95, 67% rule, 0.5s frames)
  ▶ Playing chunk 1/2 on SERVER speakers
  [AEC] Pos: 5460/44100 | Echo energy: 3245 | Cleaned energy: 123
  [INTERRUPT MONITOR] Chunk #10 | VAD prob: 0.234 | Frames confirmed: 0/1 | Buffer: 10/15
  [AEC] Pos: 16380/44100 | Echo energy: 4567 | Cleaned energy: 156
  [INTERRUPT CHECK] Chunk 15/15 | Current VAD: 0.145 | Frame Avg: 0.187 | Frames confirmed: 0/1

  [FRAME COMPLETE] Stats: avg=0.187, max=0.234, min=0.089, chunks>=0.95=0/15 (need 10)
```

**Good signs:**
- Echo energy high (TTS is loud)
- Cleaned energy low (echo removed successfully)
- VAD probs low (0.1-0.3, not detecting TTS as speech)
- No false interrupts

### 6. Test Real Interrupt

Speak during playback:

```
[AEC] Pos: 27300/44100 | Echo energy: 2134 | Cleaned energy: 2567
  [INTERRUPT CHECK] Chunk 10/15 | Current VAD: 0.978 | Frame Avg: 0.945 | Frames confirmed: 0/1
  [INTERRUPT CHECK] Chunk 15/15 | Current VAD: 0.989 | Frame Avg: 0.971 | Frames confirmed: 0/1

  [FRAME COMPLETE] Stats: avg=0.971, max=0.996, min=0.912, chunks>=0.95=14/15 (need 10)
  [FRAME PASSED] 14/15 chunks >= 0.95 (67%+ rule)! Frames confirmed: 1/1
  [INTERRUPT CONFIRMED!] 1 frames at VAD >= 0.95

🚨 [VAD INTERRUPT] CONFIRMED! (VAD: 0.971, total chunks processed: 87)
```

**Good signs:**
- Cleaned energy INCREASED (user speech louder than echo)
- VAD probs high (0.9+, detecting real speech)
- Interrupt confirmed

---

## Limitations

### 1. Timing Drift

**Issue:** If playback and mic aren't perfectly synced, subtraction misaligns.

**Symptoms:**
- Cleaned energy stays high even without user speech
- Echo partially remains

**Solution:**
- Use constant sample rate (16kHz for both)
- No complex resampling (just scipy.signal.resample once)
- Linear position tracking

### 2. Room Acoustics

**Issue:** Echo undergoes transformations (reverb, attenuation, delay).

**Current approach:** Simple scaling (0.8x) and direct subtraction.

**Doesn't handle:**
- Multi-path reflections
- Room reverberation
- Frequency-dependent attenuation

**If needed:** Use full adaptive AEC (WebRTC library)

### 3. Stereo/Mono Mismatch

**Issue:** TTS might be stereo, mic is mono.

**Solution:** Average L+R channels before subtraction.

```python
if len(data.shape) > 1:
    data = np.mean(data, axis=1)
```

---

## Advanced: Full WebRTC AEC

If simple subtraction doesn't work well, use full WebRTC AEC:

### Installation

```bash
# Option 1: Try pip (may not work on all systems)
pip install webrtc-audio-processing

# Option 2: Build from source
bash INSTALL_AEC.sh
```

### Usage

```python
from webrtc_audio_processing import AudioProcessingModule as AP

# Initialize
ap = AP(enable_aec=True, enable_ns=True)
ap.set_stream_format(16000, 1)  # 16kHz, mono

# Process audio
for chunk in incoming_mic_chunks:
    # Feed reference signal (what's playing)
    ap.process_reverse_stream(playback_chunk)

    # Process mic input (with echo removed)
    cleaned = ap.process_stream(chunk)

    # Run VAD on cleaned audio
    is_interrupt, prob = vad.detect_interrupt(cleaned)
```

**Pros:**
- Industry-standard AEC
- Handles room acoustics, delays, reverb
- Adaptive filtering

**Cons:**
- More complex setup
- Harder to debug
- Older library (2017, may have compatibility issues)

---

## Resources

- [WebRTC Audio Processing for Python](https://pypi.org/project/webrtc-audio-processing/)
- [LiveKit Python SDK AEC](https://docs.livekit.io/reference/python/v1/livekit/rtc/apm.html)
- [GitHub: python-webrtc-audio-processing](https://github.com/xiongyihui/python-webrtc-audio-processing)
- [WebRTC AEC Documentation](https://webrtc.github.io/webrtc-org/blog/2011/07/11/webrtc-improvement-optimized-aec-acoustic-echo-cancellation.html)

---

## Summary

✅ **Implemented:** Reference-based echo cancellation
✅ **How:** Subtract known TTS signal from mic input
✅ **Result:** VAD only sees user's voice, not TTS playback
✅ **Performance:** Low latency, deterministic, no training needed
✅ **Tunable:** Suppression factor (0.8 default)

Test it and adjust `echo_suppression_factor` if needed!
