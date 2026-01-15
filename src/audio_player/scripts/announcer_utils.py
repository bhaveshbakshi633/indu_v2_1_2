#!/usr/bin/env python3
"""
Announcer Utils - Shared TTS functions for voice announcements
Chatterbox TTS primary, Edge TTS fallback
"""

import os
import time
import threading
import tempfile
import asyncio
import requests
import numpy as np

# TTS imports
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print('Warning: edge-tts not available')

try:
    import soundfile as sf
    from scipy import signal
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print('Warning: soundfile/scipy not available')


class AnnouncerTTS:
    """
    TTS Manager - Chatterbox primary, Edge TTS fallback
    Automatically checks for Chatterbox availability and switches back when restored
    """

    def __init__(self, chatterbox_host='192.168.123.169', chatterbox_port=8000,
                 audio_receiver_url='http://localhost:5050/play_audio',
                 edge_voice='en-IN-NeerjaExpressiveNeural', gain=4.0):
        self.chatterbox_host = chatterbox_host
        self.chatterbox_port = chatterbox_port
        self.chatterbox_url = f'http://{chatterbox_host}:{chatterbox_port}'
        self.audio_receiver_url = audio_receiver_url
        self.edge_voice = edge_voice
        self.gain = gain

        self.fallback_active = False
        self.checker_thread = None
        self._lock = threading.Lock()

        print(f'AnnouncerTTS initialized')
        print(f'  Chatterbox: {self.chatterbox_url}')
        print(f'  Audio receiver: {self.audio_receiver_url}')
        print(f'  Edge voice: {self.edge_voice}')
        print(f'  Gain: {self.gain}x')

    def announce(self, message, wait=True):
        """
        Announce message via G1 speaker
        Uses Chatterbox if available, falls back to Edge TTS
        """
        print(f'Announcing: {message}')

        try:
            # Generate audio (Chatterbox or Edge TTS)
            audio_data, sample_rate = self._generate_audio(message)

            if audio_data is None:
                print('  Failed to generate audio')
                return False

            # Convert to PCM and send to G1
            pcm_bytes = self._convert_to_pcm(audio_data, sample_rate)

            if pcm_bytes is None:
                print('  Failed to convert audio')
                return False

            # Send to G1 speaker
            success = self._send_to_g1(pcm_bytes)

            if success and wait:
                # Wait for playback duration
                duration = len(pcm_bytes) / 32000.0  # 16kHz, 16-bit
                time.sleep(duration + 0.3)  # Extra buffer

            return success

        except Exception as e:
            print(f'  Announce error: {e}')
            return False

    def _generate_audio(self, text):
        """Generate audio using Chatterbox or Edge TTS"""

        with self._lock:
            should_try_chatterbox = not self.fallback_active

        # Try Chatterbox first
        if should_try_chatterbox:
            try:
                print('  Trying Chatterbox TTS...')
                resp = requests.get(
                    f'{self.chatterbox_url}/tts',
                    params={'text': text},
                    timeout=30
                )
                if resp.status_code == 200:
                    # Save to temp file and load
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.write(resp.content)
                    temp_file.close()

                    audio_data, sample_rate = sf.read(temp_file.name)
                    os.unlink(temp_file.name)

                    print(f'  Chatterbox success: {len(audio_data)} samples at {sample_rate}Hz')
                    return audio_data, sample_rate
            except Exception as e:
                print(f'  Chatterbox failed: {e}')

            # Chatterbox failed - activate fallback
            with self._lock:
                if not self.fallback_active:
                    self.fallback_active = True
                    self._start_availability_checker()
                    print('  Activated Edge TTS fallback mode')

        # Use Edge TTS fallback
        return self._generate_edge_tts(text)

    def _generate_edge_tts(self, text):
        """Generate audio using Edge TTS"""
        if not EDGE_TTS_AVAILABLE:
            print('  Edge TTS not available!')
            return None, None

        try:
            print('  Using Edge TTS fallback...')
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Run async Edge TTS
            async def generate():
                communicate = edge_tts.Communicate(text, self.edge_voice)
                await communicate.save(temp_path)

            asyncio.run(generate())

            # Load audio
            audio_data, sample_rate = sf.read(temp_path)
            os.unlink(temp_path)

            print(f'  Edge TTS success: {len(audio_data)} samples at {sample_rate}Hz')
            return audio_data, sample_rate

        except Exception as e:
            print(f'  Edge TTS error: {e}')
            return None, None

    def _convert_to_pcm(self, audio_data, sample_rate):
        """Convert audio to 16kHz mono 16-bit PCM"""
        if not AUDIO_PROCESSING_AVAILABLE:
            print('  Audio processing not available!')
            return None

        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)

            # Apply gain
            audio_data = audio_data * self.gain

            # Clip and convert to 16-bit
            audio_data = np.clip(audio_data, -1.0, 1.0)
            pcm_data = (audio_data * 32767).astype(np.int16)

            return pcm_data.tobytes()

        except Exception as e:
            print(f'  PCM conversion error: {e}')
            return None

    def _send_to_g1(self, pcm_bytes):
        """Send PCM audio to G1 speaker via HTTP POST"""
        try:
            resp = requests.post(
                self.audio_receiver_url,
                data=pcm_bytes,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=10
            )
            if resp.status_code == 200:
                result = resp.json()
                print(f'  Sent to G1: {result.get("bytes_received", 0)} bytes')
                return True
            else:
                print(f'  G1 error: {resp.status_code}')
                return False
        except Exception as e:
            print(f'  G1 send error: {e}')
            return False

    def _start_availability_checker(self):
        """Start background thread to check Chatterbox availability"""
        if self.checker_thread is None or not self.checker_thread.is_alive():
            self.checker_thread = threading.Thread(
                target=self._check_chatterbox_availability,
                daemon=True
            )
            self.checker_thread.start()
            print('  Started Chatterbox availability checker')

    def _check_chatterbox_availability(self):
        """Background thread: check if Chatterbox is back online"""
        while True:
            with self._lock:
                if not self.fallback_active:
                    print('  Chatterbox checker: fallback no longer active, stopping')
                    return

            time.sleep(10)

            try:
                resp = requests.get(
                    f'{self.chatterbox_url}/health',
                    timeout=5
                )
                if resp.status_code == 200:
                    with self._lock:
                        self.fallback_active = False
                    print('  Chatterbox is back online!')

                    # Announce restoration (using Edge TTS one last time)
                    self._announce_with_edge('Chatterbox TTS restored')
                    return
            except:
                pass

    def _announce_with_edge(self, message):
        """Force announce using Edge TTS (for restoration message)"""
        audio_data, sample_rate = self._generate_edge_tts(message)
        if audio_data is not None:
            pcm_bytes = self._convert_to_pcm(audio_data, sample_rate)
            if pcm_bytes is not None:
                self._send_to_g1(pcm_bytes)
                duration = len(pcm_bytes) / 32000.0
                time.sleep(duration + 0.3)


# Global announcer instance
_announcer = None

def get_announcer(chatterbox_host='192.168.123.169', chatterbox_port=8000, gain=None):
    """Get or create global announcer instance"""
    global _announcer
    if _announcer is None:
        # Environment variable se gain lo, default 0.5
        if gain is None:
            gain = float(os.environ.get('GAIN', '0.5'))
            gain = max(0.1, min(6.0, gain))
        _announcer = AnnouncerTTS(
            chatterbox_host=chatterbox_host,
            chatterbox_port=chatterbox_port,
            gain=gain
        )
    return _announcer

def announce(message, wait=True):
    """Convenience function to announce message"""
    announcer = get_announcer()
    return announcer.announce(message, wait=wait)


if __name__ == '__main__':
    # Test
    print('Testing AnnouncerTTS...')
    announce('Hello! This is a test announcement.')
    print('Test complete.')
