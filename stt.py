"""Speech-to-Text via NVIDIA Parakeet-TDT-0.6B-v3 (NeMo, in-process).

Multilingual ASR over 25 European languages with **automatic language
detection** — the source language does not need to be specified, so mixed
Hungarian/English dictation just works. The model runs directly on the GPU in
this process; there is no external STT service.
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
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def _detected_lang(hyp):
    """Best-effort read of the auto-detected language off a NeMo hypothesis."""
    for attr in ("lang", "language", "langs"):
        val = getattr(hyp, attr, None)
        if val:
            return val[0] if isinstance(val, (list, tuple)) else val
    return "auto"


class SpeechToText:
    """NVIDIA Parakeet-TDT-0.6B-v3 multilingual ASR (in-process, auto-detect)."""

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
        # NB: keep float32. Parakeet's TDT decoder (CUDA-graph label looping)
        # mixes float32 tensors internally, so a manual bfloat16 cast or
        # autocast triggers "mat1 and mat2 must have the same dtype". The 0.6B
        # model is small enough that float32 inference is still well sub-realtime.
        self.model.eval()

        print(f"STT model ready ({MODEL_NAME}) on {self.device}", flush=True)

    def transcribe(self, audio: np.ndarray, language: str = None) -> tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio: numpy array of audio samples at 16kHz
            language: ignored — parakeet-v3 auto-detects the language

        Returns:
            Tuple of (transcribed text, detected language code or "auto")
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
            output = self.model.transcribe([temp_path])
            if not output:
                return "", "auto"
            hyp = output[0]
            text = (hyp.text or "").strip()
            return text, _detected_lang(hyp)
        finally:
            os.unlink(temp_path)
