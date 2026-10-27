# Aria — mirror concierge POC

A foyer-mirror concierge, fully local. One Python file (`main.py`) + one HTML page (`static/index.html`).
The portrait **is** the avatar: one deformable face+head mesh warped in WebGL, so mouth, jaw, eyes, brows,
head and hair move as part of the picture.

```
camera → OpenCV YuNet (detect) + SFace (recognize) → anonymous runtime registry (User 1, User 2, …) → presence/session state machine
Chrome mic → Web Speech API (STT) → POST /chat → Ollama (native tool calling, reasoning)
→ mock Home API (/home/api/v1/actions) mutates device state + logs the Home Assistant request it would send
→ Kokoro TTS (ONNX, CPU) → per-20 ms lip features → NDJSON stream → WebGL mesh avatar, lips driven by the audio clock
```

Run it: see [docs/DOCKER.md](../docs/DOCKER.md) (`docker compose up --build` + a native Ollama) or, for development,
`cd app && uv sync && uv run uvicorn main:app --port 8000`, then open http://localhost:8000 in **Chrome**.

## What the testing page does

- **Start mirror** asks for camera + mic. A dim "mirror" veil lifts when someone approaches. She greets once per visit
  (one sentence + one offer as a question), then listens. She goes back to a mirror when nobody is there.
- **Users are anonymous.** There are no names. A face that engages the mirror is remembered as **User 1**, the next
  new face as **User 2**, and so on (auto-remember, switchable). **Snapshot my face** registers explicitly. Only SFace
  embeddings are kept, in process memory, until **Forget all users** or a restart. Everyone else is a *guest* (default
  user): general welcome, no label, no private household info.
- **Scenarios** are deterministic conversation flows. Each step shows what to say, what tool call is expected, what the
  model actually called, what she said, and the **Home Assistant request** the mock bridge would send
  (`POST /api/services/media_player/play_media …`). Home Assistant itself is out of scope: requests are logged, not sent.
- **Identity dropdown + Simulate approach** work without a camera (forces guest / User N through the same session logic).
- **Interrupt**: speaking into the mic, typing, or the button stops audio in ~1 ms and the face stops with it.
- **Event stream** is everything the backend emits, as a Home-API bridge would see it.

## Avatar rig (what moves)

Built once by `python main.py build-mesh` from `static/avatar.png` + `static/avatar_landmarks.json` (478 MediaPipe
landmarks; re-run with the `mediapipe` extra for a new portrait). Output `static/avatar_mesh.json`:
face tessellation + dense grid over head and hair + coarse grid over body, with a baked **head weight** per point
(1 = rotates rigidly with the head incl. hair, 0 = pinned: shoulders, background, image border) and depth.

In the browser, every frame: lips from the audio clock (fast attack, slower release), jaw drop with the whole lower
face, upper-lip lift, corner width from spectral brightness, smile/cheeks, brows, blinks (150–220 ms, 2–7 s apart,
occasional double), micro-saccades, gaze that tracks the detected face and compensates head turns, breathing, a
slow drift so she is never frozen, listening tilt/nod, a thinking glance away, speech-energy head emphasis. All targets
are smoothed with time-based exponential filters, so motion never jumps. The oral cavity (shaded individual teeth, lip
shadow, tongue, corner darkening) is drawn under the warped lips on a 2D canvas.

## Models

| Role | Default | Notes |
|---|---|---|
| Brain | Ollama `gemma4:e4b-it-q4_K_M` | native tool calling, `think:false`; `MODEL=` to switch, see docs/DOCKER.md |
| Voice | Kokoro int8 ONNX, `af_heart` | CPU, RTF ≈ 0.4 on M1; `VOICE=af_bella|bf_emma|…` |
| Vision | OpenCV Zoo YuNet + SFace | CPU, 40–200 ms per frame |
| STT | Chrome Web Speech API | the only thing that leaves the machine (audio → Google); swap for local Whisper later |

## Endpoints

- `POST /chat` NDJSON: `token` · `speech` (pcm16 + lip features) · `tool_call` (with reasoning) · `tool_result` (with `ha` request) · `done` (metrics)
- `POST /vision/frame` · `POST /vision/simulate` · `POST /users/snapshot` · `GET /users` · `POST /users/reset` · `POST /users/forget` · `POST /users/auto`
- `POST /home/api/v1/actions` (SIMULATED) · `GET /home/api/v1/state` · `GET /home/api/v1/events` (SSE) · `GET /metrics` · `GET /scenarios`
- Events: presence.detected/left, session.started/ended, identity.confirmed/unknown, user.registered/forgotten, users.reset, greeting.suppressed,
  turn.user, llm.first_token, speech.filler, tool_call.requested, **ha.request**, device.state_changed, speech.interrupted, turn.cancelled,
  policy.unverified_claim/missing_tool_call/repaired, turn.assistant, scenario.start/pass/fail/end

## Not in scope (POC)

All device actions are simulated and labelled. No Sonos/Lutron/Home Assistant traffic. No persistence of faces or
preferences across restarts (by design for the POC). The physical mirror, glass, microphone array and arrival awareness
are later phases.
