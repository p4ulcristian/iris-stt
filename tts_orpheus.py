"""TTS via Orpheus-FastAPI with streaming and request cancellation."""

import os
import time
import threading

import numpy as np
import requests

ORPHEUS_API_URL = os.environ.get("ORPHEUS_API_URL", "http://localhost:8020")
ORPHEUS_VOICE = os.environ.get("ORPHEUS_VOICE", "tara")


class OrpheusTTS:
    def __init__(self):
        self.api_url = ORPHEUS_API_URL
        self.voice = ORPHEUS_VOICE
        self.sample_rate = 24000
        self._current_response = None
        self._lock = threading.Lock()

        print(f"Connecting to Orpheus TTS at {self.api_url}...", flush=True)
        try:
            resp = requests.get(f"{self.api_url}/v1/audio/voices", timeout=5)
            if resp.status_code == 200:
                voices = resp.json().get("voices", [])
                print(f"Orpheus TTS ready — voices: {voices}", flush=True)
        except Exception as e:
            print(f"Warning: Orpheus API not reachable: {e}", flush=True)

    def set_speaker(self, voice: str):
        self.voice = voice

    def cancel_current(self):
        """Cancel any in-progress TTS request."""
        with self._lock:
            if self._current_response:
                try:
                    self._current_response.close()
                    print("[TTS] Cancelled previous request", flush=True)
                except:
                    pass
                self._current_response = None

    def synthesize_stream(self, text: str, first_chunk_bytes: int = 2400, min_chunk_bytes: int = 4800, **kwargs):
        """Yield audio chunks as they stream from Orpheus."""
        if not text or not text.strip():
            return

        # Cancel any in-progress request
        self.cancel_current()

        start_time = time.time()
        print(f"[TTS-Stream] Starting for: {text[:50]}...", flush=True)

        try:
            resp = requests.post(
                f"{self.api_url}/v1/audio/speech/stream",
                json={"input": text, "voice": self.voice},
                timeout=120,
                stream=True,
            )
            
            with self._lock:
                self._current_response = resp
                
            if resp.status_code != 200:
                return

            header_skipped = False
            buffer = bytearray()
            first_chunk_sent = False
            chunk_count = 0

            for chunk in resp.iter_content(chunk_size=2048):
                # Check if we've been cancelled
                with self._lock:
                    if self._current_response is not resp:
                        print("[TTS-Stream] Cancelled, stopping", flush=True)
                        return
                        
                if not chunk:
                    continue
                buffer.extend(chunk)

                # Skip WAV header
                if not header_skipped and len(buffer) > 44:
                    buffer = buffer[44:]
                    header_skipped = True

                threshold = first_chunk_bytes if not first_chunk_sent else min_chunk_bytes

                if header_skipped and len(buffer) >= threshold:
                    usable = (len(buffer) // 2) * 2
                    chunk_count += 1
                    elapsed = time.time() - start_time
                    print(f"[TTS-Stream] Chunk {chunk_count} ({usable} bytes) at {elapsed:.3f}s", flush=True)
                    yield self._bytes_to_float32(bytes(buffer[:usable]))
                    buffer = buffer[usable:]
                    first_chunk_sent = True

            # Yield remaining
            if buffer and header_skipped:
                usable = (len(buffer) // 2) * 2
                if usable > 0:
                    chunk_count += 1
                    yield self._bytes_to_float32(bytes(buffer[:usable]))

            print(f"[TTS-Stream] Done: {chunk_count} chunks in {time.time()-start_time:.3f}s", flush=True)

        except Exception as e:
            print(f"[Orpheus] Stream error: {e}", flush=True)
        finally:
            with self._lock:
                if self._current_response is resp:
                    self._current_response = None

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        if len(audio_bytes) == 0:
            return np.zeros(0, dtype=np.float32)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0
