"""
openWakeWord server-side detection for Iris.

Each client arm() call generates a fresh session ID. The server keeps one
AudioFeatures + ORT classifier instance per session so mel-feature history
is isolated between clients and across re-arms.

Sessions are cleaned up after SESSION_TIMEOUT_S seconds of inactivity.
The DELETE /wake-word/session endpoint allows eager teardown on disarm.
"""

import os
import threading
import time
import warnings

import numpy as np
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

warnings.filterwarnings("ignore")

WAKE_WORD_MODEL_PATH = os.environ.get(
    "IRIS_WAKE_WORD_MODEL",
    os.path.join(
        os.path.dirname(__file__),
        "../iris-os-beta/training/hey-iris/model/hey_iris.onnx",
    ),
)

TRIGGER_THRESHOLD = 0.5
SESSION_TIMEOUT_S = 30
# Silence embeddings score ~0.014 confidence (well below 0.5 threshold),
# so no warmup suppression is needed — the keyword is detectable from
# the very first chunk.
WARMUP_CHUNKS = 0

_sessions: dict = {}
_lock = threading.Lock()


def _make_session(model_path: str) -> dict:
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    clf = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    return {
        "audio_features": AudioFeatures(ncpu=1),
        "clf":            clf,
        "last_used":      time.monotonic(),
        "chunks_seen":    0,
    }


def _get_session(session_id: str) -> dict:
    now = time.monotonic()
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = _make_session(WAKE_WORD_MODEL_PATH)
            print(f"[wake-word] new session {session_id[:8]}… "
                  f"(total: {len(_sessions)})")
        else:
            _sessions[session_id]["last_used"] = now
        return _sessions[session_id]


def remove_session(session_id: str) -> None:
    with _lock:
        if session_id in _sessions:
            del _sessions[session_id]
            print(f"[wake-word] removed session {session_id[:8]}…")


def _cleanup_loop() -> None:
    while True:
        time.sleep(10)
        cutoff = time.monotonic() - SESSION_TIMEOUT_S
        with _lock:
            stale = [k for k, v in _sessions.items() if v["last_used"] < cutoff]
            for k in stale:
                del _sessions[k]
        if stale:
            print(f"[wake-word] cleaned {len(stale)} stale session(s)")


def detect(session_id: str, audio_bytes: bytes) -> dict:
    """
    Run one detection step for session_id.

    audio_bytes must be raw little-endian float32 PCM at 16 kHz mono.
    Any length >= 1280 samples (80 ms) is accepted; 4096 (256 ms) is typical.

    Returns {"confidence": float, "detected": bool}.
    """
    audio = np.frombuffer(audio_bytes, dtype="<f4").copy()

    if len(audio) < 400:
        return {"confidence": 0.0, "detected": False}

    # Guard against clipping artefacts from the browser ScriptProcessor
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak

    # AudioFeatures expects int16 PCM; convert float32 [-1, 1] → int16 [-32768, 32767]
    audio_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    session = _get_session(session_id)
    af  = session["audio_features"]
    clf = session["clf"]

    # Each 1280-sample group produces one new embedding frame.
    # We query all of them so we don't miss a detection that falls in the
    # middle of a chunk and gets pushed out by trailing silence.
    n_new = max(1, len(audio) // 1280)

    af(audio_i16)
    session["chunks_seen"] += 1

    # Skip the first WARMUP_CHUNKS — the mel buffer is pre-filled with a
    # silence embedding, so early frames yield a spurious baseline confidence.
    if session["chunks_seen"] <= WARMUP_CHUNKS:
        return {"confidence": 0.0, "detected": False}

    # get_features(n, -1) → feature_buffer[-n:] → shape (1, n_new, 96).
    # Run the classifier on every new frame as a batch and take the max score.
    feats = af.get_features(n_feature_frames=n_new, start_ndx=-1)  # (1, n_new, 96)
    batch = feats[0]                                                 # (n_new, 96)

    _, proba = clf.run(None, {"float_input": batch})
    confidence = float(proba[:, 1].max())
    detected   = confidence >= TRIGGER_THRESHOLD

    return {"confidence": round(confidence, 4), "detected": detected}


# ── Startup validation ────────────────────────────────────────────────────────

def _validate_model_path() -> None:
    resolved = os.path.realpath(WAKE_WORD_MODEL_PATH)
    if not os.path.exists(resolved):
        print(f"[wake-word] WARNING: model not found at {resolved}")
        print("[wake-word]   set IRIS_WAKE_WORD_MODEL env var to the correct path")
    else:
        print(f"[wake-word] model OK: {resolved}")


_validate_model_path()
threading.Thread(target=_cleanup_loop, daemon=True).start()
