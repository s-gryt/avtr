# Aria — mirror concierge POC

A foyer-mirror concierge, fully local. One Python file (`main.py`) + one HTML page (`static/index.html`).
The portrait **is** the avatar: one deformable face+head mesh warped in WebGL, so mouth, jaw, eyes, brows,
head and hair move as part of the picture.

```
camera → OpenCV YuNet (detect) + SFace (recognize) → anonymous runtime registry (User 1, User 2, … from the snapshot button only; everyone else is a guest, never stored) → presence/visit state machine (greet once per run session)
Chrome mic (always on, echo-cancelled) → AudioWorklet VAD → 16 kHz WAV → POST /transcribe → host Ollama audio model → POST /chat → host Ollama chat model (native tool calling)
→ mock `home_action(device, action, value)`: nothing is controlled, the action is only logged (server log, event stream, browser console)
→ Kokoro TTS (ONNX, CPU) → per-20 ms lip features → NDJSON stream → WebGL mesh avatar, lips driven by the audio clock
```

Run it only in Docker: see [docs/DOCKER.md](../docs/DOCKER.md). Ollama runs natively on the host and serves the chat and audio models; open http://localhost:8000 in **Chrome**.

## What the testing page does

Layout: the avatar and its controls take the left third of the screen; face control, conversation, scenarios and the
event stream share the right two thirds.

- **Start mirror** starts a *run session* and asks for camera + mic. She never dims or goes dark: with nobody in front she simply waits. The portrait is also the stage background, so even if the GPU process resets and every canvas is wiped the stage shows her, never black; a lost WebGL context is restored, or replaced with a fresh canvas after 1.5 s, and each incident is logged as `avatar.gl_lost` / `avatar.gl_rebuilt` / `avatar.frame_error` (via `POST /client/log`); VS Code's integrated browser pauses the page whenever a workbench overlay (hover, menu, notification, chat UI) covers it and shows a screenshot instead, so the WebGL buffer is preserved (the screenshot shows her, not a black hole) and the camera is checked from a worker clock and grabbed straight from the camera track, so recognition does not slow down while the page is paused or hidden (`avatar.page_hidden` / `avatar.page_visible` in the event stream). What is on screen is a single plain 2D canvas, the same kind as the camera preview (a CPU-backed `willReadFrequently` canvas was shown by VS Code's browser as a dark block under camera + mic load); the WebGL face and the mouth layer are drawn on canvases that are not in the page at all and copied into it every frame, and the stage has no clipping layer. Every few frames the WebGL buffer is probed (face, chest, sleeve) without reading back the on-screen canvas; if the GPU handed back an empty or black frame she is drawn from the 2D portrait instead and stays on the 2D path for a few seconds (`avatar.blank_frame`). The tab running the mirror keeps animating (~15 fps) even while the page reports itself hidden; idle tabs do not. While anyone talks the status reads "hearing you" and she stays fully visible. There is no glow, outline or overlay around or over her; the state (listening, hearing you, thinking, speaking) is only the label at the bottom. Never put an overlay on top of her: in VS Code's browser a near-opaque border on a full-stage overlay made the compositor drop the avatar canvas underneath, so the stage went black whenever someone spoke. Room noise never cancels a reply she is still preparing: only her own audio is cut by sound alone (barge-in), a prepared reply is replaced only once the new utterance transcribes to real words, and pieces heard back-to-back are joined into one turn instead of being dropped; each piece belongs to the visit it was heard in and expires after 8 s, so old words never come back later as a phantom turn. Only one tab runs the mirror: camera frames carry the tab's run session, and a tab whose run was replaced by Start in another tab stops itself ("mirror stopped here — it was started in another tab"); without a running session the camera never triggers a greeting. For the smoothest demo, open http://localhost:8000 in Chrome itself; after a server restart open pages reload themselves.
  Everything the session remembers (who was greeted, conversations) lives in process memory until **Stop**.
- **Greets once, then listens.** On the first visit in a run session she greets (one sentence + one offer as a
  question) and cannot be interrupted by voice while she does; then she waits for the visitor to speak. Nothing she hears
  turns into a reply unless someone is in front of the mirror; short noises, filler-only transcripts and her own echo are dropped.
- **Talk over her (barge-in).** Any other time she speaks you can simply start talking: after ~0.1 s of your voice she
  lowers her volume (yielding the floor), after ~0.25 s she stops and listens to what you say (`avatar.barge_in`); if it was only
  a short noise her volume comes back and she carries on. **■ Interrupt** still works as a button.
- **Only a human voice counts.** The mic is streamed (16 kHz) to `POST /vad`, which runs Silero VAD v5 (ONNX, CPU, per
  32 ms frame). Ducking, barge-in, "hearing you" and starting/ending a turn all require *detected speech*, not just loudness, so
  claps, doors, dishes, keyboard, music, hiss or a cat never interrupt her or reach STT. If `/vad` is unreachable the browser
  falls back to loudness only (`avatar.vad_down` / `avatar.vad_back`). Limit: a TV or radio *voice* is still speech.
  A one-word reply counts: ~160 ms of detected speech is enough, so a quick "yes" or "no" is never dropped. A recording that
  is dropped as non-speech is logged as `avatar.utterance_dropped`.
- **Tool calls are never spoken.** Text-written tool calls in either form (`home_action(device="music", ...)` or Gemma's
  `home_action{device:<|"|>music<|"|>, ...}`) are executed as tool calls and stripped from the speech and the chat bubble.
- **Coming back.** A face gone for more than 1.2 s ends the visit (a shorter glance away is the same visit); a known face coming back is recognized from one clear match (~0.3 s) and the return line is pre-synthesized. When the same person returns in the same run session
  there is no second greeting: she only says "Anything I can do for you?" ("Welcome back, User 1. Anything I can do for
  you?" if someone else was there in between).
- **Users vs. guests.** There are no names. **📷 Snapshot → User** registers the face in front of the camera as
  **User 1**, the next as **User 2**, and so on (the guest in front, snapshotted, becomes User N and keeps the conversation).
  That button is the **only** way a face is stored: the avatar never creates a profile by itself. A face that is not
  recognized as a registered user is just *guest*: nothing about it is stored (no profile, no face samples), and its
  conversation is dropped when they step away (out of view for ~0.8 s or more; a shorter blip is ignored), so a guest is greeted on
  **every** approach with a fresh conversation, while User 1 is greeted once per run and afterwards only gets the
  follow-up line. When User 1 leaves and someone else steps in, the newcomer is not taken for User 1.
  Only SFace embeddings are kept, in memory (no database). Registered users survive Stop/Start until **Forget all users** or a restart. Guests get a general welcome, no label, no private household info, and she does not store preferences for them.
- **Conversation**: just talk. She answers general questions and acts on requests ("play Frank Sinatra", "dim the
  lights") by calling the mock `home_action` tool. Mock actions show up as `[mock action]` in the browser console, as
  `action.mock` in the event stream and as ⚙ lines in the chat.
- **Identity dropdown + Simulate approach / Simulate leave** work without a camera, through the same visit logic.
- **Interrupt**: speak over her (after the greeting), type, or press the button: audio stops immediately and she listens.
  Chrome echo cancellation plus an adaptive echo threshold and a short hold-off after she stops keep her own voice from
  triggering it (use headphones if the room is very reverberant). Only one tab runs the mirror at a time.
- **Scenarios** run scripted visits (greet then listen, return, User 1 → guest → User 1, music, guest privacy, barge-in)
  in a fresh run session and show expected vs. actual; each verdict is also logged as `scenario.pass` / `scenario.fail`.
- **Event stream** is everything the backend emits.

## Avatar rig (what moves)

Built once by `python main.py build-mesh` from `static/avatar.png` + `static/avatar_landmarks.json` (478 MediaPipe
landmarks; re-run with the `mediapipe` extra for a new portrait). Output `static/avatar_mesh.json`:
face tessellation + dense grid over head and hair + coarse grid over body, with a baked **head weight** per point
(1 = rotates rigidly with the head incl. hair, 0 = pinned: shoulders, background, image border) and depth.

In the browser, every frame (one render loop; the camera preview is drawn in the same loop): lips from the audio
*output* clock (`getOutputTimestamp`, interpolated between 20 ms lip frames, smooth attack/release so they do not
chatter), jaw drop limited to the lower lip and chin (it fades out right under the chin, so the neck never stretches),
upper-lip lift, corner width, smile/cheeks, brows, blinks, small saccades, gaze toward the detected face, breathing.
The portrait's head is rolled ~13°; the rig carries it close to upright and adds no tilt of its own. She sways
organically about 10 cm left and right (slow incommensurate waves through a critically damped spring, a slight lean toward
the visitor): head and shoulders travel fully, the body less toward the hips, the image border stays pinned behind an
overscan margin. Loose hair follows the head with a soft spring lag. Head turns are small and clamped. The mouth interior
(cavity, a soft upper-teeth hint, tongue, lip shadow) is drawn under the warped lips and follows the head; at rest the
untouched portrait shows through.

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
- `POST /chat` `greeting`: `"greet"` (LLM greeting) or `"return"` (fixed "Anything I can do for you?" line, no LLM)
- `POST /session/start` · `POST /session/stop` (run session: greetings, conversations) · `POST /reset` (the ↺ Conversation button: forget what was said; who was greeted is kept, so nobody is greeted twice)
- `POST /interrupt` · `POST /vision/frame` (→ presence, visit, `greet: {kind, identity, visit, text}`) · `POST /vision/simulate` (`{identity}` or `{leave: true}`)
- `POST /users/snapshot` (`{frames, role: "user"}`) · `POST /users/simulate` · `GET /users` (users, greeted, current) · `POST /users/reset` · `POST /users/forget`
- `GET /events` (SSE) · `GET /metrics` · `POST /scenarios/log`
- Events: session.started/stopped/reset, presence.detected/left, visit.started/ended, visitor.changed, greeting.suppressed,
  user.registered/already_registered/forgotten, guest.promoted, users.reset,
  speech.transcribed/transcription_error, speech.rejected (the speech model echoing its own instruction or describing silence, e.g. "What exact words are spoken in this audio?"), turn.user, llm.first_token/text_tool_call/empty_round, speech.filler, **action.mock**, action.failed,
  speech.interrupted, turn.cancelled, policy.unverified_claim/missing_tool_call/repaired/repair_failed/guest_tool_blocked, turn.assistant, turn.error,
  scenario.pass/fail

## Not in scope (POC)

All home actions are mocked and only logged. No Sonos/Lutron/Home Assistant traffic. No persistence of faces or
preferences across restarts (by design for the POC). The physical mirror, glass, microphone array and arrival awareness
are later phases.
