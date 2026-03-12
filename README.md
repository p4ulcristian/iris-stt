# Iris STT

Speech-to-Text HTTP API powered by Faster-Whisper large-v3. Send an audio file, get text back.

Supports English and Hungarian with automatic language detection.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
export IRIS_STT_API_KEY=your-secret-key
python server.py
```

Server starts on port `4260` (override with `PORT` env var).

Public URL: `https://iris-stt.irisdoes.work`

## API

### `GET /health`

Check if the model is loaded. No auth required.

```bash
curl https://iris-stt.irisdoes.work/health
```

```json
{"ready": true, "model": "faster-whisper-large-v3"}
```

### `POST /transcribe`

Upload an audio file, get transcribed text back. Requires `X-API-Key` header.

**Parameters:**
- `audio` (file, required) — audio file (wav, mp3, flac, ogg, etc.)
- `language` (form field, optional) — language code (`"en"`, `"hu"`, etc.). Auto-detects if omitted.

```bash
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "audio=@recording.wav" \
  https://iris-stt.irisdoes.work/transcribe
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
  https://iris-stt.irisdoes.work/transcribe
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IRIS_STT_API_KEY` | yes | — | API key for authentication |
| `PORT` | no | `4260` | Server port |
| `CUDA_VISIBLE_DEVICES` | no | `0` | GPU device index |

## Model

Uses [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) via CTranslate2 backend. Runs on GPU (float16) if CUDA is available, otherwise CPU (int8).

Max upload size: 50MB.
