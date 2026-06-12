"""Speech-to-Text via NVIDIA Canary-1B-v2 (NeMo, in-process).

Multilingual ASR over 25 European languages. The model runs directly on the
GPU in this process — there is no external STT service. Canary requires the
source language to be specified (it does not auto-detect); an omitted or
unsupported hint falls back to STT_DEFAULT_LANG (default "en").
"""

import os
import logging

# Must be set before any torch import.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tempfile
import soundfile as sf

SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/canary-1b-v2"
DEFAULT_LANG = os.environ.get("STT_DEFAULT_LANG", "en")

# canary-1b-v2 source languages (ISO 639-1).
SUPPORTED_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}


class SpeechToText:
    """NVIDIA Canary-1B-v2 multilingual speech recognition (in-process)."""

    model_name = MODEL_NAME

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
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        print(f"STT model ready ({MODEL_NAME}) on {self.device}", flush=True)

    def _resolve_lang(self, language):
        """Map a requested language hint to a supported source code."""
        if language:
            code = language.split("-")[0].split("_")[0].lower()
            if code in SUPPORTED_LANGS:
                return code
        return DEFAULT_LANG

    def transcribe(self, audio: np.ndarray, language: str = None) -> tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio: numpy array of audio samples at 16kHz
            language: source-language hint (e.g. "en", "hu"); falls back to
                STT_DEFAULT_LANG when omitted or unsupported

        Returns:
            Tuple of (transcribed text, language code used)
        """
        if audio is None or len(audio) == 0:
            return "", ""

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        lang = self._resolve_lang(language)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            temp_path = f.name

        try:
            with self.torch.autocast(
                device_type=self.device,
                dtype=self.torch.bfloat16,
                enabled=self.device == "cuda",
            ):
                output = self.model.transcribe(
                    [temp_path], source_lang=lang, target_lang=lang
                )
            text = output[0].text if output else ""
            return (text.strip() if text else ""), lang
        finally:
            os.unlink(temp_path)
