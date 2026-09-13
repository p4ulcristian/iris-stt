"""Text-to-Speech via Resemble AI Chatterbox-Turbo (in-process, voice cloned).

Turbo is a 350M English model that clones a voice from a short reference clip.
It generates whole utterances, not token-level audio, so ``synthesize_stream``
streams sentence by sentence: the first sentence plays while the rest render.
"""

import os
import re
import threading
import time
import warnings
import logging
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

MODEL_NAME = "ResembleAI/chatterbox-turbo"
DEFAULT_VOICE = Path(__file__).parent / "voices" / "iris-female-ref.wav"
VOICE_PATH = Path(os.environ.get("IRIS_VOICE", DEFAULT_VOICE))

# Longest text handed to one generate() call; longer sentences are split on
# clause punctuation. Very long generations drift.
MAX_CHUNK_CHARS = 280
MIN_CHUNK_CHARS = 8

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def split_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-sized chunks, merging short sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    pieces = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        # Over-long sentence: break on clause punctuation, then on words.
        current = ""
        for part in _CLAUSE_END.split(sentence):
            for word in part.split(" "):
                if current and len(current) + 1 + len(word) > max_chars:
                    pieces.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            if len(current) >= max_chars // 2:
                pieces.append(current)
                current = ""
        if current:
            pieces.append(current)

    # One chunk per sentence. Generation runs ~3x faster than realtime, so each
    # sentence is ready before the previous one finishes playing; merging
    # sentences would leave audible gaps for live players. Only fragments
    # shorter than MIN_CHUNK_CHARS are glued onto the following sentence.
    chunks, current = [], ""
    for piece in pieces:
        current = f"{current} {piece}".strip()
        if len(current) >= MIN_CHUNK_CHARS:
            chunks.append(current)
            current = ""
    if current:
        if chunks and len(chunks[-1]) + 1 + len(current) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


class ChatterboxTTS:
    """Chatterbox-Turbo with the Iris voice prepared once at startup."""

    model_name = MODEL_NAME

    def __init__(self, voice_path: Path = VOICE_PATH):
        print(f"Loading TTS model ({MODEL_NAME})...", flush=True)
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        class _Turbo(ChatterboxTurboTTS):
            def norm_loudness(self, wav, sr, target_lufs=-27):
                # Upstream multiplies float32 audio by a float64 gain. Under
                # NumPy 2 that promotes to float64, and the S3 tokenizer then
                # fails with "expected scalar type Float but found Double".
                out = super().norm_loudness(wav, sr, target_lufs)
                return np.asarray(out, dtype=np.float32)

        # Upstream logs a warning on every call about ignored CFG params.
        logging.getLogger("chatterbox").setLevel(logging.ERROR)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _Turbo.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr
        self._lock = threading.Lock()

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice reference not found: {voice_path}")
        self.model.prepare_conditionals(str(voice_path))
        self.voice = voice_path.name

        # Warm up CUDA kernels so the first real request is not slow.
        self._generate("Ready.")
        print(f"TTS model ready ({MODEL_NAME}, voice {self.voice}) on {self.device}", flush=True)

    def _generate(self, text: str) -> np.ndarray:
        with self._lock:
            wav = self.model.generate(text)
        return wav.squeeze(0).numpy().astype(np.float32)

    def synthesize_stream(self, text: str):
        """Yield mono float32 audio chunks at ``self.sample_rate``."""
        chunks = split_for_tts(text)
        start = time.time()
        for i, chunk in enumerate(chunks):
            audio = self._generate(chunk)
            print(f"[TTS] chunk {i + 1}/{len(chunks)} {len(audio) / self.sample_rate:.2f}s "
                  f"audio at {time.time() - start:.2f}s", flush=True)
            yield audio
