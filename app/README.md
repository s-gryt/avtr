# Aria — mirror concierge POC

A foyer-mirror concierge, fully local. One Python file (`main.py`) + one HTML page (`static/index.html`).
The portrait **is** the avatar: one deformable face+head mesh warped in WebGL, so mouth, jaw, eyes, brows,
head and hair move as part of the picture.

```
camera → OpenCV YuNet (detect) + SFace (recognize) → anonymous runtime registry (User 1, User 2, …) → presence/session state machine
Chrome mic (always on, echo-cancelled) → AudioWorklet VAD → 16 kHz WAV → POST /transcribe → host Ollama audio model → POST /chat → host Ollama chat model (native tool calling)
→ mock `home_action(device, action, value)`: nothing is controlled, the action is only logged (server log, event stream, browser console)
→ Kokoro TTS (ONNX, CPU) → per-20 ms lip features → NDJSON stream → WebGL mesh avatar, lips driven by the audio clock
```

Run it only in Docker: see [docs/DOCKER.md](../docs/DOCKER.md). Ollama runs natively on the host and serves the chat and audio models; open http://localhost:8000 in **Chrome**.

## What the testing page does

- **Start mirror** asks for camera + mic. A dim "mirror" veil lifts when someone approaches. She greets once per visit
  (one sentence + one offer as a question), then waits for one visitor turn before responding. She goes back to a mirror when nobody is there.
- **Users are anonymous.** There are no names. New faces stay guests by default; **Snapshot my face** explicitly registers
  a household user as **User 1**, the next registered person as **User 2**, and so on. Optional auto-remember can be enabled.
  Only SFace
  embeddings are kept, in process memory, until **Forget all users** or a restart. Everyone else is a *guest* (default
  user): general welcome, no label, no private household info.
- **Conversation**: just talk. She answers general questions (facts, advice, jokes) and acts on requests ("play Frank
  Sinatra", "pause the music", "dim the lights") by calling the mock `home_action` tool. Mock actions show up as
  `[mock action]` in the browser console, as `action.mock` in the event stream and as ⚙ lines in the chat.
- **Identity dropdown + Simulate approach** work without a camera (forces guest / User N through the same session logic).
- **Interrupt**: speak over her (voice barge-in), type, or press the button — audio stops immediately and she listens.
  The mic stays open while she speaks; Chrome echo cancellation plus an adaptive echo threshold keep her own voice from
  triggering it (use headphones if the room is very reverberant). Only one tab runs the mirror at a time.
- **Event stream** is everything the backend emits.

## Avatar rig (what moves)

Built once by `python main.py build-mesh` from `static/avatar.png` + `static/avatar_landmarks.json` (478 MediaPipe
landmarks; re-run with the `mediapipe` extra for a new portrait). Output `static/avatar_mesh.json`:
face tessellation + dense grid over head and hair + coarse grid over body, with a baked **head weight** per point
(1 = rotates rigidly with the head incl. hair, 0 = pinned: shoulders, background, image border) and depth.

In the browser, every frame: lips from the audio clock (fast attack, slower release), jaw drop with the whole lower
face, upper-lip lift, corner width from spectral brightness, smile/cheeks, brows, blinks (150–220 ms, 2–7 s apart,
occasional double), micro-saccades, gaze that tracks the detected face and compensates head turns, breathing, a
slow drift so she is never frozen, listening tilt/nod, a thinking glance away, speech-energy head emphasis. All targets
are smoothed with time-based exponential filters, so motion never jumps. Head rotation uses compressed depth and small
clamped angles so face, hair and skull move as one rigid piece. The lips part as a lens (full at the center, closed at
the corners) and the whole lower lip rides the jaw, so it keeps its shape. The mouth interior (dark cavity, a soft
upper-teeth hint under the lip, tongue, lip shadow) is drawn anti-aliased under the warped lips on a 2D canvas; at rest
the untouched portrait shows through.

## Models

| Role | Default | Notes |
|---|---|---|
| Brain | Ollama `gemma4:e4b-it-q4_K_M` | native tool calling; `MODEL=` to switch |
| Speech input | Ollama `gemma4:e4b-it-q4_K_M` | audio-capable model; `STT_MODEL=` switches independently |
| Voice | Kokoro int8 ONNX, `af_heart` | CPU, RTF ≈ 0.4 on M1; `VOICE=af_bella|bf_emma|…` |
| Vision | OpenCV Zoo YuNet + SFace | CPU, 40–200 ms per frame |
| Avatar | WebGL portrait mesh | no Ollama model; landmarks drive face/head deformation and audio-clock lip motion |

## Endpoints

- `POST /transcribe` WAV bytes → host Ollama `/v1/audio/transcriptions` → `{"text"}`; `POST /chat` NDJSON: `token` · `speech` (pcm16 + lip features) · `tool_call` · `tool_result` · `done` (metrics)
- `POST /interrupt` · `POST /reset` · `POST /vision/frame` · `POST /vision/simulate` · `POST /users/snapshot` · `POST /users/simulate` · `GET /users` · `POST /users/reset` · `POST /users/forget` · `POST /users/auto`
- `GET /events` (SSE) · `GET /metrics`
- Events: presence.detected/left, session.started/ended/reset, identity.confirmed, user.registered/already_registered/forgotten, users.reset/auto_remember,
  speech.transcribed/transcription_error, turn.user, llm.first_token/text_tool_call/empty_round, speech.filler, **action.mock**, action.failed,
  speech.interrupted, turn.cancelled, policy.unverified_claim/repaired/repair_failed, turn.assistant, turn.error

## Not in scope (POC)

All home actions are mocked and only logged. No Sonos/Lutron/Home Assistant traffic. No persistence of faces or
preferences across restarts (by design for the POC). The physical mirror, glass, microphone array and arrival awareness
are later phases.
