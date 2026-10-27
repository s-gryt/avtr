# Running with Docker

## Quick start (fresh Mac)

1. Install Ollama natively: https://ollama.com/download (or `brew install ollama`).
2. Pull the model: `ollama pull gemma4:e4b-it-q4_K_M` (or `make pull-model`).
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
| `gemma4:e4b-it-q4_K_M` | 6.6 GB | Default. Native tool calling. ~8-10 tok/s on M1 16 GB |
| `gemma4:e4b` | 9.5 GB | Full-precision variant of the same model |
| `qwen3:4b-instruct` | 2.5 GB | Light fallback for low RAM |
| `gemma4:12b` | larger | Upgrade candidate for a 48 GB MacBook |
| `qwen3:14b` | larger | Upgrade candidate for a 48 GB MacBook |

Larger models fit on a 48 GB MacBook, but verify tool-calling support for each: `ollama show <model>` must list the `tools` capability. Confirm the tag exists in the Ollama library before pulling.

## Do you need multiple models?

No. One Ollama chat model with tool support is enough.

- TTS: Kokoro ONNX, runs inside the container on CPU.
- Vision: OpenCV YuNet (detection) + SFace (recognition), inside the container on CPU.
- STT: Chrome Web Speech API, in the browser.

## What leaves your machine

Only speech-to-text audio, sent to Google by Chrome's Web Speech API. LLM, TTS, and vision all run locally.
