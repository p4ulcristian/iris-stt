"""TTS via Docker XTTS API with streaming support."""

import os
import io
import struct
import numpy as np
import requests
import wave
from urllib.parse import quote

TTS_API_URL = os.environ.get("XTTS_API_URL", "http://localhost:8020")


class TextToSpeech:
    """TTS wrapper using Docker XTTS API with streaming."""

    def __init__(self):
        print(f"Connecting to TTS Docker API at {TTS_API_URL}...", flush=True)
        self.api_url = TTS_API_URL
        self.speaker = "sample"
        self.sample_rate = 24000
        
        # Check connection
        try:
            resp = requests.get(f"{self.api_url}/languages", timeout=5)
            if resp.status_code == 200:
                print(f"TTS Docker API ready (streaming enabled)", flush=True)
            else:
                print(f"TTS API returned {resp.status_code}", flush=True)
        except Exception as e:
            print(f"Warning: TTS API not reachable: {e}", flush=True)

    def set_speaker(self, audio_path: str):
        """Set speaker (not implemented for Docker API yet)."""
        print(f"set_speaker not implemented for Docker API", flush=True)

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize text to a mono float32 waveform using streaming endpoint."""
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)
        
        try:
            # Use streaming endpoint for faster response
            encoded_text = quote(text)
            url = f"{self.api_url}/tts_stream?text={encoded_text}&speaker_wav={self.speaker}&language=en"
            
            resp = requests.get(url, timeout=120)
            
            if resp.status_code != 200:
                print(f"[TTS] Stream error: {resp.status_code}, falling back to non-stream", flush=True)
                return self._synthesize_fallback(text, audio_prompt_path, **kwargs)
            
            # Fix WAV header and convert to float32
            wav_bytes = self._fix_wav_header(resp.content)
            return self._wav_to_float32(wav_bytes)
            
        except Exception as e:
            print(f"[TTS] Error: {e}, falling back to non-stream", flush=True)
            return self._synthesize_fallback(text, audio_prompt_path, **kwargs)

    def _synthesize_fallback(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Fallback to non-streaming endpoint."""
        try:
            resp = requests.post(
                f"{self.api_url}/tts_to_audio",
                json={
                    "text": text,
                    "language": "en",
                    "speaker_wav": self.speaker
                },
                timeout=60
            )
            
            if resp.status_code != 200:
                print(f"[TTS] Fallback error: {resp.status_code}", flush=True)
                return np.zeros(0, dtype=np.float32)
            
            result = resp.json()
            audio_url = result.get("url", "")
            if not audio_url:
                return np.zeros(0, dtype=np.float32)
            
            audio_path = audio_url.split("/")[-1]
            audio_resp = requests.get(f"{self.api_url}/output/{audio_path}", timeout=30)
            
            if audio_resp.status_code != 200:
                return np.zeros(0, dtype=np.float32)
            
            return self._wav_to_float32(audio_resp.content)
            
        except Exception as e:
            print(f"[TTS] Fallback error: {e}", flush=True)
            return np.zeros(0, dtype=np.float32)

    def synthesize_stream(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        chunk_size: int = 8192,
        **kwargs,
    ):
        """Stream audio chunks as they're generated.
        
        Uses the /tts_stream endpoint for real-time streaming.
        Yields numpy float32 arrays as audio chunks arrive.
        """
        if not text or not text.strip():
            return
        
        try:
            encoded_text = quote(text)
            url = f"{self.api_url}/tts_stream?text={encoded_text}&speaker_wav={self.speaker}&language=en"
            
            print(f"[TTS] Starting stream: {text[:50]}...", flush=True)
            
            resp = requests.get(url, stream=True, timeout=120)
            
            if resp.status_code != 200:
                print(f"[TTS] Stream error: {resp.status_code}", flush=True)
                return
            
            # Buffer for accumulating audio data
            audio_buffer = bytearray()
            header_size = 44  # Standard WAV header
            header_received = False
            
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                
                audio_buffer.extend(chunk)
                
                # Skip the WAV header, yield audio data
                if not header_received and len(audio_buffer) >= header_size:
                    header_received = True
                    audio_data = bytes(audio_buffer[header_size:])
                    audio_buffer = bytearray()
                    
                    if len(audio_data) > 0:
                        yield self._bytes_to_float32(audio_data)
                elif header_received and len(audio_buffer) >= chunk_size:
                    audio_data = bytes(audio_buffer)
                    audio_buffer = bytearray()
                    yield self._bytes_to_float32(audio_data)
            
            # Yield any remaining data
            if len(audio_buffer) > 0:
                yield self._bytes_to_float32(bytes(audio_buffer))
            
            print(f"[TTS] Stream complete", flush=True)
            
        except Exception as e:
            print(f"[TTS] Stream error: {e}", flush=True)

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw PCM bytes (16-bit) to float32 numpy array."""
        if len(audio_bytes) == 0:
            return np.zeros(0, dtype=np.float32)
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0

    def _wav_to_float32(self, wav_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes to float32 numpy array."""
        try:
            buffer = io.BytesIO(wav_bytes)
            with wave.open(buffer, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                return audio.astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[TTS] WAV parse error: {e}", flush=True)
            # Try extracting raw PCM if header is bad
            if len(wav_bytes) > 44:
                return self._bytes_to_float32(wav_bytes[44:])
            return np.zeros(0, dtype=np.float32)

    def _fix_wav_header(self, wav_bytes: bytes) -> bytes:
        """Fix WAV header with correct file/data sizes."""
        if len(wav_bytes) < 44:
            return wav_bytes
        
        data = bytearray(wav_bytes)
        
        # Fix RIFF chunk size (file size - 8)
        file_size = len(data) - 8
        data[4:8] = struct.pack('<I', file_size)
        
        # Find and fix data chunk size
        data_pos = wav_bytes.find(b'data')
        if data_pos > 0 and data_pos + 8 <= len(data):
            data_chunk_size = len(data) - data_pos - 8
            data[data_pos+4:data_pos+8] = struct.pack('<I', data_chunk_size)
        
        return bytes(data)
