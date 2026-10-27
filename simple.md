Short answer first: the concierge needs almost no "reasoning." It needs three things from the brain: reliable function calling, carrying 5–10 turns of context ("yes" → the music question), and obeying a short policy prompt (no names when unsure or when others are present). Any model that scores well on function-calling benchmarks does this, including 4B-class ones. The fastest feedback comes from a hosted brain, with local models tested afterwards through the same scenario replay.

## Which brain

| Candidate | Runs on | Tool calling | Verdict for the POC |
|---|---|---|---|
| Gemini Live | Hosted | Native | First demo: zero setup, hearing+brain+voice+tools in one key, ~$0.02/min |
| OpenAI gpt-realtime-mini | Hosted | Native | Same role, ~$0.016/min |
| Qwen3.5 4B (Q4, ~3 GB) | M1 and M4 | Native in Ollama/llama.cpp | Best local candidate for the M1 |
| Gemma 4 E4B / 12B | M1 (E4B) / M4 | Check: Gemma 3 lacked native tool calling; verify Gemma 4's template before relying on it | Comparison row |
| NVIDIA Nemotron 3 Nano 30B-A3B | M4 only (~17–18 GB at Q4) | Native, with its own reasoning/tool markup; a custom parser interprets its reasoning and tool-calling markup so vLLM can route tool calls, and it is released under NVIDIA's open model license permitting commercial use. | This is the model you're thinking of. ~3B active parameters, so it should decode quickly on the M4, but check that your runtime (Ollama/llama.cpp/MLX) supports its hybrid architecture and parser before planning around it. Too large for the M1. |
| GPT-OSS 20B | M4 | Native | Comparison row |

The scenario replay decides. Twenty scripted turns with expected tool calls; the local model that passes them at acceptable first-token delay becomes your offline/privacy mode.

## Fastest path to a cloneable, testable application on the M1

Skip Docker for the application at first. Camera, microphone, browser and the bitHuman engine are host-native on a Mac anyway, so a `uv`-managed Python project with a lock file gives you "clone and run" reproducibility faster than a compose stack. Docker stays for Home Assistant and Music Assistant on the Dell. The hexagonal boundaries live in the Python package, independent of how it's launched.

Prerequisites on the M1: macOS 13+, Git, Python 3.11+, `uv`, Chrome, ~10 GB free disk, and three keys in a `.env`: Google AI Studio (Gemini Live), bitHuman API secret (Creator plan), and a Home Assistant token later.

| Step | What you do | Result | Time |
|---|---|---|---|
| 1 | `pipecat init` quickstart with SmallWebRTC + Gemini Live; run it; talk; interrupt it | Voice loop with barge-in in Chrome | 1–2 h |
| 2 | Follow bitHuman's Pipecat quickstart (docs.bithuman.ai/platforms/pipecat); create the avatar from your PNG in their console; point the bot at it | Your picture talking, lip-synced, 416×720 at 20 fps, rendered on the M1 | ½ day |
| 3 | Add four tools (`play_music`, `media_control`, `set_volume`, `lighting_scene`) behind a `HomeActions` port with a mock adapter that logs calls and flags `SIMULATED` | "Yes, play Sinatra" produces the right call with the right room | ½ day |
| 4 | Face script: OpenCV YuNet + SFace, enrollment from photos in a folder, multi-frame confirmation, sends `person_confirmed` to the bot over a local WebSocket; bot injects context and greets | Walk up, get greeted by name; stranger gets a generic welcome | 1 day |
| 5 | Metrics: Pipecat latency observer + browser audio start/stop + tool-call timestamps → CSV | The four numbers Mikki asked for | ½ day |
| 6 | README, `make setup` / `make up` / `make demo`, `.env.example`, `uv.lock`; push | `git clone … && make setup && make up` on the M4 | ½ day |

Roughly four to five working days to a demo someone else can clone. On the M4 the same clone runs Essence 2 (1080×1920 at 25 fps) after the macOS 26 upgrade, plus the local-model comparison. The TalkingHead puppet and a rented-GPU neural renderer are later adapters behind the same avatar port, so they add to the comparison without changing anything above.

## What "validated" means for this concept

The concept is proven when, on the M1 with hosted brain and bitHuman:
1. recognition triggers a correct, privacy-aware greeting;
2. the music question → "yes, play Sinatra" produces the correct tool call;
3. interruption stops voice and face together;
4. the four timings are recorded and look acceptable (greeting ≤2.5 s, reply ≤1.5 s, interruption ≤0.5 s, action ≤1–4 s with a real device).

Items 1–3 plus the mock executor settle the architecture and the avatar question. Real Sonos/Lutron timing and the local-brain comparison are the next two experiments, each a one-adapter addition on the Dell and the M4.