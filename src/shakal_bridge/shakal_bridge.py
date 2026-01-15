#!/usr/bin/env python3
"""
Shakal Bridge - Face Detection to Greeting Integration
========================================================
- Reads shakal headless output (stdout)
- Triggers personalized greeting via Ollama + TTS
- One-time greeting per session
- Queue greeting if TTS busy
"""

import subprocess
import requests
import threading
import time
import json
import os
from collections import deque

class ShakalBridge:
    def __init__(self):
        self.greeted_faces = set()  # Session memory - reset on restart
        self.tts_busy = False
        self.pending_greetings = deque()
        self.ollama_url = "http://192.168.123.230:11434/api/generate"
        self.tts_url = "http://192.168.123.169:8080/api/speak"  # brain_v2
        self.lock = threading.Lock()
        
        # TTS busy check thread
        self.running = True
        self.pending_thread = threading.Thread(target=self._process_pending, daemon=True)
        self.pending_thread.start()
        
        print("🎭 ShakalBridge initialized")
        print(f"   Ollama: {self.ollama_url}")
        print(f"   TTS: {self.tts_url}")

    def on_face_detected(self, name):
        """Face detect hone par call hota hai"""
        if name == "Unknown":
            return
            
        with self.lock:
            if name in self.greeted_faces:
                return  # Already greeted this session
            
            self.greeted_faces.add(name)
            
            if self.tts_busy:
                # Queue with alternate message
                self.pending_greetings.append({
                    "name": name,
                    "delayed": True
                })
                print(f"📝 Queued greeting for {name} (TTS busy)")
            else:
                # Immediate greeting
                self._greet(name, delayed=False)

    def _greet(self, name, delayed=False):
        """Ollama se greeting generate karke TTS se bolao"""
        try:
            self.tts_busy = True
            
            if delayed:
                prompt = f"You are INDU, a friendly robot. You saw {name} a moment ago but were busy talking. Now greet them warmly and apologize for the delay. Keep it short and natural, 1-2 sentences max. Just the greeting, no explanation."
            else:
                prompt = f"You are INDU, a friendly robot. You just noticed {name} in front of you. Greet them warmly by name. Keep it short and natural, 1-2 sentences max. Just the greeting, no explanation."
            
            print(f"🤖 Generating greeting for {name}...")
            
            # Call Ollama
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 50
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                greeting = response.json().get("response", "").strip()
                if greeting:
                    print(f"💬 Greeting: {greeting}")
                    self._speak(greeting)
                else:
                    print(f"⚠️ Empty greeting for {name}")
            else:
                print(f"❌ Ollama error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Greeting error for {name}: {e}")
        finally:
            self.tts_busy = False

    def _speak(self, text):
        """TTS se bolao via brain_v2 API"""
        try:
            response = requests.post(
                self.tts_url,
                json={"text": text},
                timeout=60
            )
            if response.status_code != 200:
                print(f"⚠️ TTS error: {response.status_code}")
        except Exception as e:
            print(f"❌ TTS request failed: {e}")

    def _process_pending(self):
        """Pending greetings process karo jab TTS free ho"""
        while self.running:
            time.sleep(1)
            
            if not self.tts_busy and self.pending_greetings:
                with self.lock:
                    if self.pending_greetings:
                        item = self.pending_greetings.popleft()
                        self._greet(item["name"], delayed=True)

    def stop(self):
        self.running = False


def main():
    """Shakal ka stdout read karke bridge ko feed karo"""
    bridge = ShakalBridge()
    
    # Shakal process run karo
    shakal_cmd = ["./shakal", "-c", "../config/config.yaml", "--headless"]
    
    print("🎬 Starting Shakal face detection...")
    
    try:
        process = subprocess.Popen(
            shakal_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/shakal/build",
            text=True,
            bufsize=1
        )
        
        # Read stdout line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                # Parse comma-separated names
                names = [n.strip() for n in line.split(",")]
                for name in names:
                    if name:
                        bridge.on_face_detected(name)
                        
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        bridge.stop()
        process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")
        bridge.stop()


if __name__ == "__main__":
    main()
