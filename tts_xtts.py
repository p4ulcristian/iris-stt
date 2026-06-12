"""XTTS-v2 TTS via Docker API."""

import io
import os
import requests

XTTS_API_URL = os.environ.get("XTTS_API_URL", "http://localhost:8020")

class XTTSStreaming:
    def __init__(self):
        print(f"Connecting to XTTS Docker API at {XTTS_API_URL}...", flush=True)
        self.api_url = XTTS_API_URL
        self.speaker = "sample"  # Default speaker
        self.sample_rate = 24000
        
        # Check connection
        try:
            resp = requests.get(f"{self.api_url}/languages", timeout=5)
            if resp.status_code == 200:
                print(f"XTTS Docker API ready", flush=True)
            else:
                print(f"XTTS API returned {resp.status_code}", flush=True)
        except Exception as e:
            print(f"Warning: XTTS API not reachable: {e}", flush=True)
    
    def synthesize_streaming(self, text, language="en"):
        """Generate audio and yield as single chunk (streaming API has bug)."""
        wav_bytes = self.synthesize(text, language)
        if wav_bytes:
            yield wav_bytes
    
    def synthesize(self, text, language="en"):
        """Synthesize complete audio. Returns WAV bytes."""
        try:
            # Call Docker TTS API
            resp = requests.post(
                f"{self.api_url}/tts_to_audio",
                json={
                    "text": text,
                    "language": language,
                    "speaker_wav": self.speaker
                },
                timeout=60
            )
            
            if resp.status_code != 200:
                print(f"TTS API error: {resp.status_code} - {resp.text[:200]}", flush=True)
                return None
            
            # Get the audio URL from response
            result = resp.json()
            audio_url = result.get("url", "")
            
            if not audio_url:
                print(f"TTS API returned no URL: {result}", flush=True)
                return None
            
            # Fetch the actual audio file
            # URL comes as localhost, replace with our API URL
            audio_path = audio_url.split("/")[-1]
            audio_resp = requests.get(f"{self.api_url}/output/{audio_path}", timeout=30)
            
            if audio_resp.status_code == 200:
                return audio_resp.content
            else:
                print(f"Failed to fetch audio: {audio_resp.status_code}", flush=True)
                return None
                
        except Exception as e:
            print(f"TTS error: {e}", flush=True)
            return None


if __name__ == "__main__":
    import time
    xtts = XTTSStreaming()
    
    print("\nTesting synthesis:")
    text = "Hello, this is a test of the Docker TTS API."
    start = time.time()
    audio = xtts.synthesize(text)
    elapsed = time.time() - start
    if audio:
        print(f"Generated {len(audio)} bytes in {elapsed:.2f}s")
    else:
        print("Failed to generate audio")
