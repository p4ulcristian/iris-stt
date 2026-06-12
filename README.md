# Iris Comms

HTTP API for speech-to-text and text-to-speech.

- **STT** — [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): multilingual ASR over 25 European languages with automatic language detection, run in-process via NeMo.
- **TTS** — Orpheus (`orpheus-3b`, Q8_0), streamed as Server-Sent Events.

Both models run on the local GPU (bfloat16 when CUDA is available, otherwise CPU). There is no external STT/TTS service.

## Setup

Dependencies are managed with `uv` against a Python 3.11 venv:

```bash
uv sync
```

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
  "tts_model": "orpheus-3b (Q8_0)"
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
included only on the first chunk. `sample_rate` is reported per chunk.

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

## Models

- **STT** — [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) via NeMo, in-process. 25 European languages with automatic language detection, GPU (bfloat16) when CUDA is available.
- **TTS** — Orpheus (`orpheus-3b`, Q8_0), in-process, streamed over SSE.

Max upload size: 50MB.
