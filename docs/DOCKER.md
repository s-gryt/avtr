# Running with Docker

## Quick start (fresh Mac)

1. Install Ollama natively: https://ollama.com/download (or `brew install ollama`).
2. Pull the chat/audio model: `ollama pull gemma4:e4b-it-q4_K_M`. This installed Gemma tag supports tool calling and audio input.
3. Start the app: `docker compose up --build` (or `make up`). First build downloads ~150 MB of models into a cached layer.
4. Open http://localhost:8000 in Chrome.

The port is published on `127.0.0.1` only: the API has no authentication and the page holds camera and microphone permission. Set `BIND=0.0.0.0` in `.env` only on a trusted network, e.g. to open it from another device.

Chrome only allows camera/mic on `https://` or `localhost`. Use `http://localhost:8000`, not a LAN IP, unless you add TLS.

## How the container reaches Ollama

Ollama runs on the host. The container calls it via `host.docker.internal:11434` (set by default in `OLLAMA`). Docker Desktop provides that name on Mac/Windows. On Linux, compose adds it through `extra_hosts: host.docker.internal:host-gateway`. Linux Ollama must listen on an address the bridge can reach: `OLLAMA_HOST=0.0.0.0 ollama serve`.

Ollama is installed natively on the host (Metal GPU on Apple Silicon). It is not run in Docker: Docker on Mac has no Metal GPU access, so in-container inference would be CPU-only and far slower. The container only maps to the host's Ollama port.

## Switching models

Set `MODEL` in `.env` (copy `.env.example`) or inline: `MODEL=qwen3:4b-instruct docker compose up`. `VOICE` picks the Kokoro voice (default `af_heart`).

| Model | Size | Notes |
|---|---|---|
| `gemma4:e4b-it-q4_K_M` | 6.6 GB | Default chat and audio transcription model. Native tool calling; supports audio input. |
| `gemma4:e4b` | 9.5 GB | Full-precision variant of the same model |
| `qwen3:4b-instruct` | 2.5 GB | Light fallback for low RAM |
| `gemma4:12b` | larger | Upgrade candidate for a 48 GB MacBook |
| `qwen3:14b` | larger | Upgrade candidate for a 48 GB MacBook |

Larger models fit on a 48 GB MacBook, but verify tool-calling support for each: `ollama show <model>` must list the `tools` capability. Confirm the tag exists in the Ollama library before pulling.

## Model roles

`MODEL` and `STT_MODEL` are independent settings in `.env`. Both default to `gemma4:e4b-it-q4_K_M`; set `STT_MODEL` to another locally pulled Ollama model if it supports audio input. The Docker app forwards captured mic audio to the host Ollama `/v1/audio/transcriptions` endpoint.

- Chat/tool selection: Ollama `MODEL` on the host.
- Speech-to-text: Ollama `STT_MODEL` on the host.
- Text-to-speech: Kokoro ONNX, runs inside Docker on CPU; no Ollama model is needed.
- Face detection/recognition: OpenCV YuNet + SFace, runs inside Docker on CPU; no Ollama model is needed.
- Avatar animation: WebGL portrait mesh and audio-derived lip features in Chrome; no Ollama model is needed.

## Local data

Speech audio is sent to the host Ollama service, not to Google. Face embeddings and preferences exist only in the app process memory and are cleared when the container restarts. The app container has no device credentials; home actions are simulated.
