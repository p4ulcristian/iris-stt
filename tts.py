"""Chatterbox TTS — text-to-speech with optional zero-shot voice cloning.

Uses Resemble AI's Chatterbox model (chatterbox-tts on PyPI). Output is
mono float32 audio at the model's native sample rate (24 kHz).

Zero-shot voice cloning: pass a short reference audio file (3–10 s of
clean speech) to `synthesize(..., audio_prompt_path=...)` and the output
will match that speaker. Without a prompt, the model uses its built-in
default voice.
"""

import os
import logging

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

MODEL_NAME = "ResembleAI/chatterbox"


class TextToSpeech:
    """Chatterbox TTS wrapper.

    `exaggeration` controls emotion intensity (0 = flat, 0.5 = default,
    higher = more expressive). `cfg_weight` controls how strictly the
    model follows the text vs. the reference style — lower values
    (~0.3) produce faster, more natural pacing for expressive speech;
    higher values (~0.7) stay closer to the text.
    """

    def __init__(self):
        print(f"Loading TTS model ({MODEL_NAME})...", flush=True)

        import torch
        from chatterbox.tts import ChatterboxTTS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = int(self.model.sr)

        print(f"TTS model ready ({MODEL_NAME}) on {self.device} @ {self.sample_rate} Hz", flush=True)

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> np.ndarray:
        """Synthesize text to a single mono float32 waveform at self.sample_rate."""
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)

        with self.torch.inference_mode():
            wav = self.model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        # ChatterboxTTS returns a (1, N) or (N,) torch tensor; normalise to 1-D float32 numpy.
        if hasattr(wav, "detach"):
            wav = wav.detach().to("cpu", dtype=self.torch.float32).numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        return wav
