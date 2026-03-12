"""Faster-Whisper large-v3 STT - Multilingual speech recognition."""

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

SAMPLE_RATE = 16000


class SpeechToText:
    """Faster-Whisper large-v3 for multilingual speech recognition.

    Uses CTranslate2 backend for 4x speedup compared to vanilla Whisper.
    Preserves original language (English stays English, Hungarian stays Hungarian).
    """

    def __init__(self):
        print("Loading STT model (Faster-Whisper large-v3)...", flush=True)

        from faster_whisper import WhisperModel

        model_path = "Systran/faster-whisper-large-v3"
        print(f"Using multilingual model: {model_path}", flush=True)

        import torch
        if torch.cuda.is_available():
            self.model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16"
            )
        else:
            self.model = WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8"
            )

        print("STT model ready (Faster-Whisper large-v3)", flush=True)

    def transcribe(self, audio: np.ndarray, language: str = None) -> tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio: numpy array of audio samples at 16kHz
            language: optional language code ("en", "hu", etc.). Auto-detects if None.

        Returns:
            Tuple of (transcribed text, detected language code)
        """
        if audio is None or len(audio) == 0:
            return "", ""

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Detect or use provided language
        if language and language != "auto":
            detected_lang = language
        else:
            lang, lang_prob, all_probs = self.model.detect_language(audio)
            probs_dict = dict(all_probs) if all_probs else {}
            hu_prob = probs_dict.get("hu", 0)
            en_prob = probs_dict.get("en", 0)
            detected_lang = "hu" if hu_prob > en_prob else "en"

        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=detected_lang,
            task="transcribe",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            )
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        text = " ".join(text_parts).strip()
        return text if text else "", detected_lang
