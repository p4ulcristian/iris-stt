# Iris Comms

HTTP API for speech-to-text and text-to-speech.

- **STT** — NVIDIA Canary 180m-flash (English).
- **TTS** — Resemble AI's Chatterbox, with optional zero-shot voice cloning.

## Setup

```bash
pip install -r requirements.txt
pip install --no-deps -r requirements-tts.txt
```

### Note for RTX 50-series / Blackwell (sm_120)

`chatterbox-tts` hard-pins `torch==2.6.0`, which has **no sm_120 kernels** and
will fail on Blackwell GPUs with `no kernel image is available for execution on
the device`. The two-step install above sidesteps this:

1. `requirements.txt` installs everything *except* chatterbox-tts (including
   chatterbox's transitive deps, resolved against your existing torch).
2. `requirements-tts.txt` installs chatterbox-tts itself with `--no-deps` so
   pip doesn't try to downgrade torch.

Make sure your torch is a cu128 build with sm_120 (Blackwell) kernels — e.g.
`torch>=2.9.1+cu128` from the PyTorch cu128 index:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
export IRIS_COMMS_API_KEY=your-secret-key
python server.py
```

Server starts on port `4260` (override with `PORT` env var).

Public URL: `https://iris-comms.irisdoes.work`

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
  "stt_model": "nvidia/canary-180m-flash",
  "tts_model": "ResembleAI/chatterbox"
}
```

### `POST /stt/transcribe`

Upload an audio file, get transcribed text back. Requires `X-API-Key` header.

**Parameters:**
- `audio` (file, required) — audio file (wav, mp3, flac, ogg, etc.)
- `language` (form field, optional) — language code (`"en"`, `"hu"`, etc.). Auto-detects if omitted.

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

### `POST /tts/synthesize`

Send text, get a WAV (24 kHz mono PCM16). Requires `X-API-Key`.

**JSON body:**
- `text` (required) — string, max 5000 chars
- `exaggeration` (optional, default `0.5`) — emotion intensity, `0.0`–`2.0`. Higher = more expressive
- `cfg_weight` (optional, default `0.5`) — `0.0`–`1.0`. Lower (~0.3) = faster pacing, higher = stays closer to text

```bash
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Iris."}' \
  --output speech.wav \
  https://iris-comms.irisdoes.work/tts/synthesize
```

**Zero-shot voice cloning** — upload a 3–10 s clean reference clip as multipart `voice`:

```bash
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "text=Hello from Iris." \
  -F "voice=@reference.wav" \
  --output speech.wav \
  https://iris-comms.irisdoes.work/tts/synthesize
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IRIS_COMMS_API_KEY` | yes | — | API key for authentication |
| `PORT` | no | `4260` | Server port |
| `CUDA_VISIBLE_DEVICES` | no | `0` | GPU device index |

## Models

- **STT** — [nvidia/canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash) via NeMo. GPU (bfloat16) if CUDA is available, otherwise CPU.
- **TTS** — [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) via `chatterbox-tts`. GPU when available. Output is 24 kHz mono.

Max upload size: 50MB.
