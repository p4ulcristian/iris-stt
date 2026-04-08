"""NVIDIA Canary 1B Flash STT - English speech recognition."""

import os
import logging

# Must set before any torch imports
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Suppress warnings
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tempfile
import soundfile as sf

SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/canary-180m-flash"


class SpeechToText:
    """NVIDIA Canary 1B Flash for English speech recognition."""

    def __init__(self):
        print(f"Loading STT model ({MODEL_NAME})...", flush=True)

        import torch
        torch.set_float32_matmul_precision("high")
        from nemo.collections.asr.models import ASRModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.model = ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.to(self.torch.bfloat16)
        self.model.eval()

        print(f"STT model ready ({MODEL_NAME}) on {self.device}", flush=True)

    def transcribe(self, audio: np.ndarray, language: str = None) -> tuple[str, str]:
        """Transcribe audio to text (English only).

        Args:
            audio: numpy array of audio samples at 16kHz
            language: ignored, always transcribes as English

        Returns:
            Tuple of (transcribed text, language code)
        """
        if audio is None or len(audio) == 0:
            return "", ""

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            temp_path = f.name

        try:
            with self.torch.autocast(device_type=self.device, dtype=self.torch.bfloat16, enabled=self.device == "cuda"):
                output = self.model.transcribe(
                    [temp_path], source_lang="en", target_lang="en"
                )
            text = output[0].text if output else ""
            return text.strip() if text else "", "en"
        finally:
            os.unlink(temp_path)
