# Iris Comms

HTTP API for speech-to-text and text-to-speech.

- **STT** — [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): multilingual ASR over 25 European languages with automatic language detection, run in-process via NeMo.
- **TTS** — [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo): 350M English model, cloned from `voices/iris-female-ref.wav`, streamed sentence by sentence as Server-Sent Events.

Both models run in-process on the local GPU (CPU fallback). There is no external STT/TTS service.

## Setup

Dependencies are managed with `uv` against a Python 3.11 venv:

```bash
uv sync
```

The venv must use a **uv-managed** CPython (pinned in `pyproject.toml`). A
Nix-built python links its own glibc and cannot load the system NVIDIA driver,
which shows up as `CUDA error: unknown error` while loading the STT model.

`chatterbox-tts` hard-pins `torch==2.6.0` and `transformers==5.2.0`. The
`override-dependencies` in `pyproject.toml` replace those pins, so `uv sync`
installs it cleanly next to NeMo.

### Note for RTX 50-series / Blackwell (sm_120)

Blackwell GPUs need a CUDA 12.8+ / 13.x build of torch with sm_120 kernels —
older wheels fail with `no kernel image is available for execution on the
device`. NeMo and torch are pinned accordingly in the project; if you install
torch manually, use a recent cu12x/cu13x index, e.g.:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Usage

```bash
export IRIS_COMMS_API_KEY=your-secret-key
uv run --python 3.11 server.py
```

Server starts on port `4260` (override with `PORT` env var). Models load in the
background after startup; `/health` reports when they are ready.

Public URL: `https://iris-comms.irisdoes.work` (preferred for in-cluster use:
the WireGuard-direct address of the host).

## API

### `GET /health`

Check if the models are loaded. No auth required.

```bash
curl https://iris-comms.irisdoes.work/health
```

```json
{
  "ready": true,
  "stt_ready": true,
  "tts_ready": true,
  "stt_model": "nvidia/parakeet-tdt-0.6b-v3",
  "tts_model": "ResembleAI/chatterbox-turbo"
}
```

### `POST /stt/transcribe`

Upload an audio file, get transcribed text back. Requires `X-API-Key` header.

**Parameters:**
- `audio` (file, required) — audio file (wav, mp3, flac, ogg, etc.)
- `language` (form field, optional) — **ignored**; parakeet-v3 auto-detects the
  language. Detected code is returned in the `language` field of the response
  (`"auto"` if the model does not expose it). Supported languages: bg, hr, cs,
  da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es,
  sv, ru, uk.

```bash
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "audio=@recording.wav" \
  https://iris-comms.irisdoes.work/stt/transcribe
```

```json
{"text": "Hello, how are you?", "language": "en", "duration": 3.21}
```

With explicit language:

```bash
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "audio=@recording.wav" \
  -F "language=hu" \
  https://iris-comms.irisdoes.work/stt/transcribe
```

### `POST /tts/stream`

Send text, get synthesized speech streamed as Server-Sent Events. Requires
`X-API-Key`. Accepts a JSON body or form field.

**Body:**
- `text` (required) — string, max 5000 chars

**Event stream:**
```
event: chunk
data: {"seq": 0, "sample_rate": 24000, "audio_b64": "<base64 WAV>", "text": "..."}
...
event: done
data: {"chunks": <int>}

event: error
data: {"error": "..."}
```

Each `chunk` carries a base64-encoded WAV (mono PCM16) fragment; `text` is
included only on the first chunk. `sample_rate` is reported per chunk (24000).
Each sentence is sent as its own chunk as soon as it is rendered, so playback
starts in about half a second and stays ahead of realtime. English only.
Output carries Resemble's inaudible Perth watermark.

```bash
curl -N -X POST \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Iris."}' \
  https://iris-comms.irisdoes.work/tts/stream
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IRIS_COMMS_API_KEY` | yes | — | API key for authentication |
| `PORT` | no | `4260` | Server port |
| `CUDA_VISIBLE_DEVICES` | no | `0` | GPU device index |
| `IRIS_VOICE` | no | `voices/iris-female-ref.wav` | Reference clip (>5 s) for the cloned voice |

## Models

- **STT** — [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) via NeMo, in-process. 25 European languages with automatic language detection, GPU (bfloat16) when CUDA is available.
- **TTS** — [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), in-process, streamed over SSE.

Max upload size: 50MB.
