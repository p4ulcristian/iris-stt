#!/usr/bin/env python3
"""
Iris STT Server - Speech-to-Text HTTP API

Endpoints:
  GET    /health               - Health check
  POST   /transcribe           - Upload audio file, get text back
  POST   /synthesize           - Send text, get WAV audio back (optional voice clone)
  POST   /detect-wake-word     - Stream raw float32 PCM chunk, get confidence score
  DELETE /wake-word/session    - Explicit session teardown on disarm
"""

import io
import os
from pathlib import Path
import tempfile
import threading

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

import numpy as np
import soundfile as sf
import resampy
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from stt import SpeechToText, SAMPLE_RATE
from tts import TextToSpeech
from wake_word import detect as ww_detect, remove_session as ww_remove_session

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# State
stt_model = None
tts_model = None
stt_ready = False
tts_ready = False

# Auth - required
API_KEY = os.environ.get("IRIS_STT_API_KEY")


def require_api_key(f):
    """Decorator to require API key."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        provided = request.headers.get("X-API-Key")
        if provided != API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)

    return decorated


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "ready": stt_ready and tts_ready,
        "stt_ready": stt_ready,
        "tts_ready": tts_ready,
        "stt_model": "nvidia/canary-180m-flash",
        "tts_model": "ResembleAI/chatterbox",
    })


@app.route('/transcribe', methods=['POST'])
@require_api_key
def transcribe():
    if not stt_ready:
        return jsonify({"error": "STT model not ready yet"}), 503

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided. Send as multipart with field name 'audio'"}), 400

    audio_file = request.files['audio']
    language = request.form.get('language', None)

    # Save to temp file for soundfile to read
    suffix = os.path.splitext(audio_file.filename or '.wav')[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        audio_file.save(f.name)
        temp_path = f.name

    try:
        audio, sr = sf.read(temp_path)

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            audio = resampy.resample(audio.astype(np.float32), sr, SAMPLE_RATE)

        audio = audio.astype(np.float32)
        duration = len(audio) / SAMPLE_RATE

        text, detected_lang = stt_model.transcribe(audio, language=language)

        return jsonify({
            "text": text,
            "language": detected_lang,
            "duration": round(duration, 2)
        })
    finally:
        os.unlink(temp_path)


@app.route('/synthesize', methods=['POST'])
@require_api_key
def synthesize():
    if not tts_ready:
        return jsonify({"error": "TTS model not ready yet"}), 503

    # Accept JSON or multipart (multipart needed when uploading a voice reference).
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        exaggeration = float(payload.get("exaggeration", 0.5))
        cfg_weight = float(payload.get("cfg_weight", 0.5))
    else:
        text = request.form.get("text", "") or request.values.get("text", "")
        exaggeration = float(request.form.get("exaggeration", 0.5))
        cfg_weight = float(request.form.get("cfg_weight", 0.5))

    if not text or not text.strip():
        return jsonify({"error": "Missing 'text'"}), 400
    if len(text) > 5000:
        return jsonify({"error": "text too long (max 5000 chars)"}), 400
    if not (0.0 <= exaggeration <= 2.0):
        return jsonify({"error": "exaggeration must be between 0.0 and 2.0"}), 400
    if not (0.0 <= cfg_weight <= 1.0):
        return jsonify({"error": "cfg_weight must be between 0.0 and 1.0"}), 400

    # Optional zero-shot voice cloning: 3–10 s of clean reference audio.
    voice_path = None
    if 'voice' in request.files:
        voice_file = request.files['voice']
        suffix = os.path.splitext(voice_file.filename or '.wav')[1] or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            voice_file.save(f.name)
            voice_path = f.name

    try:
        audio = tts_model.synthesize(
            text,
            audio_prompt_path=voice_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        if audio.size == 0:
            return jsonify({"error": "Synthesis produced no audio"}), 500

        buf = io.BytesIO()
        sf.write(buf, audio, tts_model.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return send_file(buf, mimetype="audio/wav", as_attachment=False, download_name="speech.wav")
    finally:
        if voice_path:
            os.unlink(voice_path)


@app.route('/detect-wake-word', methods=['POST'])
@require_api_key
def detect_wake_word():
    session_id = request.headers.get("X-Wake-Session")
    if not session_id:
        return jsonify({"error": "X-Wake-Session header required"}), 400

    audio_bytes = request.get_data()

    min_bytes = 1280 * 4  # 1280 float32 samples = 80 ms
    if len(audio_bytes) < min_bytes:
        return jsonify({"error": f"Audio too short (got {len(audio_bytes)} bytes, need >= {min_bytes})"}), 400

    if len(audio_bytes) % 4 != 0:
        return jsonify({"error": "Body must be raw float32 PCM (byte length must be a multiple of 4)"}), 400

    result = ww_detect(session_id, audio_bytes)
    return jsonify(result)


@app.route('/wake-word/session', methods=['DELETE'])
@require_api_key
def delete_wake_session():
    session_id = request.headers.get("X-Wake-Session")
    if session_id:
        ww_remove_session(session_id)
    return jsonify({"ok": True})


def load_stt():
    global stt_model, stt_ready
    stt_model = SpeechToText()
    stt_ready = True


def load_tts():
    global tts_model, tts_ready
    tts_model = TextToSpeech()
    tts_ready = True


if __name__ == '__main__':
    if not API_KEY:
        print("ERROR: Set IRIS_STT_API_KEY environment variable before starting.")
        print("  export IRIS_STT_API_KEY=your-secret-key")
        exit(1)

    port = int(os.environ.get('PORT', 4260))
    print(f"Loading STT + TTS models in background...")
    threading.Thread(target=load_stt, daemon=True).start()
    threading.Thread(target=load_tts, daemon=True).start()
    print(f"Starting server on port {port} (API key required)...")
    app.run(host='0.0.0.0', port=port, threaded=True)
