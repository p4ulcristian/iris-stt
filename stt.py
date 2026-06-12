"""Speech-to-Text via NVIDIA Canary 1B NIM API."""

import os
import io
import tempfile
import numpy as np
import soundfile as sf
import requests

SAMPLE_RATE = 16000
STT_API_URL = os.environ.get("STT_API_URL", "http://localhost:8000")


class SpeechToText:
    """STT via NVIDIA Canary 1B NIM API."""

    def __init__(self):
        print(f"Connecting to STT Docker API at {STT_API_URL}...", flush=True)
        self.api_url = STT_API_URL
        
        # Check connection
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"STT Docker API ready", flush=True)
            else:
                print(f"STT API returned {resp.status_code}", flush=True)
        except Exception as e:
            print(f"Warning: STT API not reachable: {e}", flush=True)

    def transcribe(self, audio: np.ndarray, language: str = None) -> tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio: numpy array of audio samples at 16kHz
            language: language hint (optional)

        Returns:
            Tuple of (transcribed text, language code)
        """
        if audio is None or len(audio) == 0:
            return "", ""

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            temp_path = f.name

        try:
            # Call Docker STT API (faster-whisper uses ISO 639-1 two-letter codes, e.g. "en")
            with open(temp_path, "rb") as audio_file:
                lang_code = language or "en"
                # Strip BCP-47 region suffix if present (e.g. "en-US" → "en")
                lang_code = lang_code.split("-")[0].split("_")[0]
                resp = requests.post(
                    f"{self.api_url}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                    data={"language": lang_code},
                    timeout=60
                )
            
            if resp.status_code == 200:
                result = resp.json()
                text = result.get("text", "")
                return text.strip(), language or "en"
            else:
                print(f"STT API error: {resp.status_code} - {resp.text[:200]}", flush=True)
                return "", ""
                
        except Exception as e:
            print(f"STT error: {e}", flush=True)
            return "", ""
        finally:
            os.unlink(temp_path)
