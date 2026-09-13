#!/usr/bin/env python3
"""
Iris Comms Server — Speech-to-Text + Text-to-Speech HTTP API.

Endpoints:
  GET    /health             - Health check
  POST   /stt/transcribe     - Upload audio file, get text back
  POST   /tts/stream         - Stream TTS as Server-Sent Events
"""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import subprocess

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
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from stt import SpeechToText, SAMPLE_RATE, MODEL_NAME as STT_MODEL_NAME
from tts_chatterbox import ChatterboxTTS, MODEL_NAME as TTS_MODEL_NAME
from normalize import normalize_for_tts

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# State
stt_model = None
tts_model = None
stt_ready = False
tts_ready = False

# Auth — required.
API_KEY = os.environ.get("IRIS_COMMS_API_KEY")


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
        "stt_model": STT_MODEL_NAME,
        "tts_model": TTS_MODEL_NAME,
    })


def convert_to_wav(input_path):
    """Convert any audio format to WAV using ffmpeg."""
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", output_path
        ], capture_output=True, timeout=30)
        if result.returncode == 0:
            return output_path
        print(f"ffmpeg conversion failed: {result.stderr.decode()[:200]}", flush=True)
        return None
    except Exception as e:
        print(f"ffmpeg error: {e}", flush=True)
        return None


@app.route('/stt/transcribe', methods=['POST'])
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
        # Try to read directly, convert if needed
        converted_path = None
        try:
            audio, sr = sf.read(temp_path)
        except Exception as read_err:
            print(f"Direct read failed ({read_err}), converting with ffmpeg...", flush=True)
            converted_path = convert_to_wav(temp_path)
            if not converted_path:
                return jsonify({"error": "Audio format not supported"}), 400
            audio, sr = sf.read(converted_path)

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
        if converted_path and os.path.exists(converted_path):
            os.unlink(converted_path)


@app.route('/tts/stream', methods=['POST'])
@require_api_key
def tts_stream():
    """Stream synthesized speech as Server-Sent Events.

    Emits:
      event: chunk
      data: {"seq": <int>, "text": "...", "sample_rate": 24000, "audio_b64": "<base64 WAV>"}
      ...
      event: done
      data: {"chunks": <int>}
      ...
      event: error
      data: {"error": "..."}
    """
    if not tts_ready:
        return jsonify({"error": "TTS model not ready yet"}), 503

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
    else:
        text = request.form.get("text", "") or request.values.get("text", "")

    if not text or not text.strip():
        return jsonify({"error": "Missing 'text'"}), 400
    if len(text) > 5000:
        return jsonify({"error": "text too long (max 5000 chars)"}), 400

    clean_text = normalize_for_tts(text)

    sr = tts_model.sample_rate

    def gen():
        try:
            chunk_seq = 0
            first_chunk = True
            for audio_chunk in tts_model.synthesize_stream(clean_text):
                if audio_chunk is None or len(audio_chunk) == 0:
                    continue
                buf = io.BytesIO()
                sf.write(buf, audio_chunk, sr, format="WAV", subtype="PCM_16")
                payload = {
                    "seq": chunk_seq,
                    "sample_rate": sr,
                    "audio_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                }
                if first_chunk:
                    payload["text"] = text
                    first_chunk = False
                yield f"event: chunk\ndata: {json.dumps(payload)}\n\n"
                chunk_seq += 1
            yield f"event: done\ndata: {json.dumps({'chunks': chunk_seq})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def load_stt():
    global stt_model, stt_ready
    stt_model = SpeechToText()
    stt_ready = True


def load_tts():
    global tts_model, tts_ready
    tts_model = ChatterboxTTS()
    tts_ready = True


if __name__ == '__main__':
    if not API_KEY:
        print("ERROR: Set IRIS_COMMS_API_KEY environment variable before starting.")
        print("  export IRIS_COMMS_API_KEY=your-secret-key")
        exit(1)

    port = int(os.environ.get('PORT', 4260))

    def warmup():
        for name, loader in (("STT", load_stt), ("TTS", load_tts)):
            try:
                loader()
            except Exception:
                import traceback
                print(f"ERROR: {name} model failed to load; exiting so systemd restarts us", flush=True)
                traceback.print_exc()
                # sys.exit() from a thread would only end the thread.
                os._exit(1)

    print(f"Loading STT then TTS models in background...")
    threading.Thread(target=warmup, daemon=True).start()
    print(f"Starting server on port {port} (API key required)...")
    app.run(host='0.0.0.0', port=port, threaded=True)
