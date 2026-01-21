# Interrupt Detection Logic

## Overview

The VAD pipeline uses **two different detection modes**:

1. **Regular Speech Detection** (for LISTENING → RECORDING transition)
2. **Strict Interrupt Detection** (for interrupting during PLAYING state)

## Regular Speech Detection

**Used for:** Starting recordings when user begins speaking

**Requirements:**
- VAD probability >= 0.5 (configurable threshold)
- 3 consecutive chunks (32ms each = ~96ms total)
- More lenient to avoid missing speech

**Configuration:**
```python
vad_threshold: 0.5          # Default threshold
frames_needed: 3            # 3 consecutive chunks
```

## Strict Interrupt Detection

**Used for:** Interrupting playback (ensures it's really human speech)

**Requirements:**
- VAD probability == **1.0** (exactly 100% confidence, NOT threshold-based)
- Checked across **3 frames**
- Each frame is **0.5 seconds** (~15 chunks of 32ms each)
- **ALL chunks** within each frame must have VAD == 1.0
- **All 3 consecutive frames** must pass this test

**Configuration:**
```python
interrupt_threshold: 1.0            # Must be exactly 1.0 (100% confidence)
interrupt_frame_duration: 0.5       # Each frame is 0.5 seconds
interrupt_frames_needed: 3          # Need 3 consecutive frames
```

## Why This Approach?

### Problem with Threshold-Based Interrupts
- Regular threshold detection (>=0.5) gives **false positives**
- Background noise, playback echo, or other sounds can trigger interrupts
- Annoying user experience (playback keeps stopping)

### Solution: Exact Match (== 1.0)
- Requires **100% confidence** from Silero VAD
- Only triggers on clear, loud human speech
- Reduces false positives dramatically
- User must speak clearly and loudly to interrupt

## Timeline Example

```
Time: 0.0s to 0.5s (Frame 1)
├─ Chunk 1 (0-32ms):    VAD = 1.0 ✓
├─ Chunk 2 (32-64ms):   VAD = 1.0 ✓
├─ Chunk 3 (64-96ms):   VAD = 1.0 ✓
├─ ...
└─ Chunk 15 (448-480ms): VAD = 1.0 ✓
   Frame 1 PASSES (all chunks == 1.0)

Time: 0.5s to 1.0s (Frame 2)
├─ Chunk 16 (480-512ms): VAD = 1.0 ✓
├─ ...
└─ Chunk 30 (928-960ms): VAD = 1.0 ✓
   Frame 2 PASSES (all chunks == 1.0)

Time: 1.0s to 1.5s (Frame 3)
├─ Chunk 31 (960-992ms): VAD = 1.0 ✓
├─ ...
└─ Chunk 45 (1408-1440ms): VAD = 1.0 ✓
   Frame 3 PASSES (all chunks == 1.0)

INTERRUPT CONFIRMED! (3 consecutive frames at 1.0)
```

If **any single chunk** in a frame is < 1.0, that frame fails and the counter resets.

## Code Implementation

### VADProcessor.detect_interrupt()

```python
def detect_interrupt(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
    # Get VAD probability for this chunk
    is_speech, prob = self.is_speech(audio_chunk)

    # Add to current frame buffer
    self.interrupt_frame_buffer.append(prob)

    # Check if we've completed a frame (0.5 seconds worth of chunks)
    if len(self.interrupt_frame_buffer) >= self.chunks_per_interrupt_frame:
        # Check if ALL chunks in this frame have VAD == 1.0 (NOT >=)
        all_max_confidence = all(p == 1.0 for p in self.interrupt_frame_buffer)

        if all_max_confidence:
            self.interrupt_frames_confirmed += 1
        else:
            self.interrupt_frames_confirmed = 0  # Reset on failure

        self.interrupt_frame_buffer = []

        # Check if we have 3 consecutive frames at 1.0
        if self.interrupt_frames_confirmed >= 3:
            return True, 1.0

    return False, prob
```

### Pipeline Manager (PLAYING State)

```python
# Monitor for interrupts while playing
while self.player.is_active():
    chunk = self.audio_monitor.get_chunk(timeout=0.1)

    # Use strict interrupt detection (VAD == 1.0)
    interrupt_confirmed, prob = self.vad.detect_interrupt(chunk)

    if interrupt_confirmed:
        # Stop playback and start recording
        self.player.stop()
        self._transition_to_recording()
        return
```

## Tuning

If you're getting:

### Too Many False Interrupts
- System is already very strict (VAD == 1.0)
- Increase `interrupt_frames_needed` from 3 to 4 or 5
- Increase `interrupt_frame_duration` from 0.5s to 0.75s or 1.0s

### Missing Real Interrupts
- User needs to speak **louder and clearer**
- Silero VAD only returns 1.0 for very confident speech
- This is by design to prevent false positives
- **Do not lower the threshold below 1.0** (defeats the purpose)

## Summary

| Aspect | Regular Detection | Interrupt Detection |
|--------|------------------|---------------------|
| **Purpose** | Start recording | Interrupt playback |
| **Threshold** | >= 0.5 (configurable) | == 1.0 (exact match) |
| **Duration** | 3 chunks (~96ms) | 3 frames × 0.5s = 1.5s |
| **Strictness** | Lenient | Very strict |
| **False Positives** | Some acceptable | Must be minimized |
| **User Action** | Speak normally | Speak clearly & loudly |

This two-tier approach ensures:
- ✓ Recordings start easily (don't miss speech)
- ✓ Interrupts only on clear human speech (no false positives)
- ✓ Better user experience overall
