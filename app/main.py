"""Aria — mirror concierge POC. Single-file backend.

camera frames -> OpenCV YuNet (detect) + SFace (recognize, anonymous runtime registry: User 1, User 2, ...)
             -> presence/session state machine (greet once per visit)
Chrome mic   -> local VAD/audio capture -> /transcribe -> host Ollama (audio model) -> /chat -> Ollama (tool calling)
             -> mock Home API (/home/api/v1/actions): mutates mock device state, logs the Home Assistant request it would send
             -> Kokoro TTS (ONNX, CPU) + per-20 ms lip features -> NDJSON stream -> WebGL mesh avatar in the browser

CLI:  python main.py build-mesh   # regenerate static/avatar_mesh.json from static/avatar.png + static/avatar_landmarks.json
"""
import asyncio, base64, collections, json, math, os, re, sys, threading, time, uuid
from datetime import datetime
from typing import Any
import numpy as np

OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "gemma4:e4b-it-q4_K_M")
STT_MODEL = os.environ.get("STT_MODEL", "gemma4:e4b-it-q4_K_M")
VOICE = os.environ.get("VOICE", "af_heart")
FRAME_MS = 20
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ───────────────────────────── Avatar mesh builder (offline, `python main.py build-mesh`) ─────────────────────────────
def build_mesh() -> dict:
    """One deformable mesh over the portrait: MediaPipe face tessellation inside the face oval, a dense grid over
    head+hair, a coarse grid over the body/background. Per-point head weight and depth are baked so the browser only
    rotates/warps. Needs static/avatar_landmarks.json (478 MediaPipe landmarks, made once with mediapipe FaceLandmarker)."""
    import cv2
    lm_path = os.path.join(STATIC, "avatar_landmarks.json")
    if not os.path.exists(lm_path):
        import mediapipe as mp   # optional, only for a brand-new portrait
        from mediapipe.tasks.python import vision as mpv
        img = mp.Image.create_from_file(os.path.join(STATIC, "avatar.png"))
        opts = mpv.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(MODELS, "face_landmarker.task")), num_faces=1)
        res = mpv.FaceLandmarker.create_from_options(opts).detect(img)
        W, H = img.width, img.height
        pts = [[round(l.x * W, 1), round(l.y * H, 1), round(l.z * W, 1)] for l in res.face_landmarks[0]]
        json.dump({"w": W, "h": H, "pts": pts}, open(lm_path, "w"))
    lm = json.load(open(lm_path)); W, H = lm["w"], lm["h"]; P = [list(p) for p in lm["pts"]]; NF = len(P)
    faces = [[int(v.split("/")[0]) - 1 for v in line.split()[1:4]] for line in open(os.path.join(MODELS, "canonical_face_model.obj")) if line.startswith("f ")]
    OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    EYE_L = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    INNER_U = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]; INNER_L = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    A = np.array(P, dtype=np.float64)
    top, chin, l, r = A[10], A[152], A[234], A[454]
    down = (chin - top)[:2]; faceH = float(np.linalg.norm(down)); down /= faceH; right = np.array([-down[1], down[0]])
    faceW = float(np.linalg.norm((r - l)[:2])); C = ((l + r) / 2)[:2]
    loc = lambda p: (float(np.dot(p[:2] - C, right)), float(np.dot(p[:2] - C, down)))
    chinV = loc(chin)[1]
    sstep = lambda a, b, x: (lambda t: t * t * (3 - 2 * t))(min(1.0, max(0.0, (x - a) / (b - a))))
    oval_poly = np.array([A[i][:2] for i in OVAL], dtype=np.float32)
    inside_oval = lambda x, y, scale=1.0: cv2.pointPolygonTest(((oval_poly - C) * scale + C).astype(np.float32), (float(x), float(y)), False) >= 0
    # outer points: ring just outside the oval, dense grid over head+hair, coarse grid elsewhere
    pts = [p for p in P]
    for s in (1.06, 1.14):
        for i in OVAL[::2]:
            q = (A[i][:2] - C) * s + C; pts.append([round(float(q[0]), 1), round(float(q[1]), 1), 0.0])
    hx0, hx1 = C[0] - faceW * 1.0, C[0] + faceW * 1.0
    hy0, hy1 = C[1] - faceH * 1.45, C[1] + faceH * 0.95
    def add_grid(step, x0, y0, x1, y1, skip):
        xs = np.arange(x0, x1 + 0.1, step); ys = np.arange(y0, y1 + 0.1, step)
        for y in ys:
            for x in xs:
                if skip(x, y): continue
                if any(abs(x - q[0]) < step * 0.35 and abs(y - q[1]) < step * 0.35 for q in pts[NF:]): continue
                pts.append([round(float(x), 1), round(float(y), 1), 0.0])
    add_grid(30, max(0, hx0), max(0, hy0), min(W - 1, hx1), min(H - 1, hy1), lambda x, y: inside_oval(x, y, 1.17))
    add_grid(80, 0, 0, W - 1, H - 1, lambda x, y: inside_oval(x, y, 1.17) or (hx0 - 20 < x < hx1 + 20 and hy0 - 20 < y < hy1 + 20))
    for x in np.arange(0, W, 80): pts.append([float(x), float(H - 1), 0.0])
    for y in np.arange(0, H, 80): pts.append([float(W - 1), float(y), 0.0])
    pts.append([float(W - 1), float(H - 1), 0.0])
    # dedupe
    seen, uniq = {}, []
    for p in pts:
        k = (round(p[0], 1), round(p[1], 1))
        if k in seen: continue
        seen[k] = len(uniq); uniq.append(p)
    pts = uniq; N = len(pts)
    # triangulate everything outside the oval: Subdiv2D over oval landmarks + outer points, drop triangles inside the oval
    sub = cv2.Subdiv2D((0, 0, W + 2, H + 2)); idx = {}
    for i in OVAL + list(range(NF, N)):
        k = (round(pts[i][0], 1), round(pts[i][1], 1)); idx[k] = i; sub.insert((float(pts[i][0]), float(pts[i][1])))
    # the tessellation already covers eye and mouth interiors: flag them (1/2 = eye whites+iris shift with gaze, 3 = mouth hole, not drawn)
    sets = [(set(EYE_R), 1), (set(EYE_L), 2), (set(INNER_U + INNER_L), 3)]
    tris = [[a, b, c, next((fl for s, fl in sets if {a, b, c} <= s), 0)] for a, b, c in faces]
    for t in sub.getTriangleList():
        ids = [idx.get((round(float(t[j]), 1), round(float(t[j + 1]), 1))) for j in (0, 2, 4)]
        if None in ids: continue
        cx, cy = sum(pts[i][0] for i in ids) / 3, sum(pts[i][1] for i in ids) / 3
        if inside_oval(cx, cy): continue
        tris.append([*ids, 0])
    # baked head weight + depth for the outer points: the whole head incl. hair is rigid (weight 1) inside an ellipse that
    # covers the hair, fades into the background beyond it, fades down the neck, and is pinned at the image border
    head = [1.0] * NF; z_of = {i: pts[i][2] for i in range(NF)}
    for i in range(NF, N):
        x, y = pts[i][0], pts[i][1]
        cu, cv_ = loc(np.array(pts[i]))
        hv = cv_ + 0.12 * faceH
        d = math.hypot(cu / (0.92 * faceW), hv / ((1.12 if hv < 0 else 0.88) * faceH))
        w = 1 - sstep(1.0, 1.32, d)
        if cv_ > chinV + 0.02 * faceH and abs(cu) < 0.5 * faceW: w = min(w, 1 - sstep(chinV + 0.02 * faceH, chinV + 0.28 * faceH, cv_))   # neck
        w *= sstep(0, 50, min(x, y, W - 1 - x, H - 1 - y))                                                                                  # image border stays put
        head.append(round(w, 3))
        # depth: hair/head behind the face silhouette, nearest oval landmark + further back with distance
        j = min(OVAL, key=lambda k: (pts[k][0] - x) ** 2 + (pts[k][1] - y) ** 2)
        dd = math.hypot(pts[j][0] - x, pts[j][1] - y) / faceW
        pts[i][2] = round(float(z_of[j]) + 20 + 60 * min(1.0, dd / 0.6), 1)
    mesh = {"w": W, "h": H, "n_face": NF, "pts": pts, "tris": tris, "head": head, "oval": OVAL, "inner_u": INNER_U, "inner_l": INNER_L, "eye_r": EYE_R, "eye_l": EYE_L}
    json.dump(mesh, open(os.path.join(STATIC, "avatar_mesh.json"), "w"), separators=(",", ":"))
    print(f"mesh: {N} points ({NF} face), {len(tris)} triangles -> static/avatar_mesh.json")
    return mesh

if __name__ == "__main__" and sys.argv[1:] == ["build-mesh"]:
    build_mesh(); sys.exit(0)

import httpx, cv2
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ───────────────────────────── Mock home actions: Aria decides, nothing is controlled — every action is only logged ─────────────────────────────
EVENTS: list[dict] = []
SUBS: set[asyncio.Queue] = set()

def emit(type_: str, **data) -> dict:
    ev = {"id": uuid.uuid4().hex[:8], "ts": round(time.time(), 3), "type": type_, **data}
    EVENTS.append(ev); del EVENTS[:-500]
    print("EVENT", json.dumps(ev), flush=True)
    for q in list(SUBS):
        try: q.put_nowait(ev)
        except asyncio.QueueFull: pass
    return ev

def apply_action(tool: str, args: dict, identity: str) -> dict:
    if tool == "home_action":
        return {"device": str(args.get("device") or "home").strip().lower(), "action": str(args.get("action") or "").strip(), "value": str(args.get("value") or "").strip()}
    if tool == "remember_preference":
        u = USERS.get(identity)
        if not u: raise ValueError("cannot store a preference for a guest")
        key = re.sub(r"[^a-z0-9_ ]", "", (args.get("topic") or "general").lower()).strip() or "general"
        u["prefs"][key] = str(args.get("preference") or "").strip()
        return {"device": f"profile.{identity}", "action": "remember", "value": f"{key}: {u['prefs'][key]}"}
    raise ValueError(f"unknown tool {tool}")

TOOLS = [
    {"type": "function", "function": {"name": "home_action",
        "description": "Do something in the house the visitor asked for: music, lights, TV, temperature, blinds, anything. Mock: it is only logged.",
        "parameters": {"type": "object", "properties": {
            "device": {"type": "string", "description": "what to control, e.g. music, lights, tv, thermostat, blinds"},
            "action": {"type": "string", "description": "what to do, e.g. play, pause, stop, turn on, turn off, dim, set"},
            "value": {"type": "string", "description": "details such as artist or song, brightness, temperature, room; empty if none"}},
            "required": ["device", "action"]}}},
    {"type": "function", "function": {"name": "remember_preference", "description": "Remember something a registered user explicitly asked you to remember about them. Never for guests.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "short key, e.g. music, coffee, morning"},
            "preference": {"type": "string", "description": "the preference in the user's words"}}, "required": ["topic", "preference"]}}},
]

SYSTEM = """You are Aria, a friendly concierge who lives in the foyer mirror of a private home. You are having a spoken, face-to-face conversation: warm, natural, a little playful. Reply in one or two short sentences (under 35 words). No emoji, lists or markdown. Never describe your thoughts.

You are also a knowledgeable companion, not just a house controller. Always answer general questions directly from your own knowledge: science, geography, history, cooking, advice, ideas, small talk, jokes (for example "How far is the moon?" => "About 384,000 kilometers, roughly 239,000 miles."). Only for live data you cannot know (news, today's weather, sports scores) say briefly that you don't have it right now.

When the visitor asks you to do something in the house (music, lights, TV, temperature, blinds, anything), call home_action, then confirm in one short sentence what you did. A short "yes", "sure", "please" after you offered something means: do what you offered, with any detail they added. For example, you offered music and they say "yes, play Frank Sinatra": call home_action with device music, action play, value Frank Sinatra. Use the real tool-calling mechanism; never write a function call in your spoken text. Never claim you did something without calling the tool. No tools for questions or chit-chat. If a registered user asks you to remember something about them, call remember_preference.

Each user message starts with a [visitor: ...] line. A registered visitor is "User 1" (or User N); use that label only in greetings. A guest is unregistered: never use a name and never reveal stored preferences. "Thanks", "bye", "that's all": a short warm goodbye, no question, no tool."""

# ───────────────────────────── Users: anonymous, runtime-only face registry ─────────────────────────────
USERS: dict[str, dict] = {}        # "user1": {"display": "User 1", "embeddings": [...], "prefs": {...}, "registered_at": ts}
AUTO = {"remember": False}         # new faces stay guests unless auto-remember is explicitly enabled

def next_user_id() -> str:
    n = 1
    while f"user{n}" in USERS: n += 1
    return f"user{n}"

def register_user(embeddings: list[list[float]], how: str) -> dict:
    uid = next_user_id()
    USERS[uid] = {"display": f"User {uid[4:]}", "embeddings": embeddings[-12:], "prefs": {}, "registered_at": round(time.time(), 3), "how": how}
    emit("user.registered", identity=uid, display=USERS[uid]["display"], samples=len(USERS[uid]["embeddings"]), how=how, stored="runtime memory only, embeddings, no images")
    return USERS[uid]

def identity_line(identity: str, confidence: float | None, others: int) -> str:
    u = USERS.get(identity)
    others_s = f", {others} other people present (no private info)" if others else ", alone"
    if not u:
        return f"[visitor: guest, not registered — no name, no private info{others_s}]"
    prefs = "; ".join(f"{k}: {v}" for k, v in u["prefs"].items()) or "none stored yet"
    conf = f", face match {confidence:.2f}" if confidence else ", simulated recognition"
    return f"[visitor: {u['display']}, registered household user{conf}{others_s}. Preferences: {prefs}]"

# ───────────────────────────── Vision: YuNet detect + SFace recognize + presence state machine ─────────────────────────────
DET = cv2.FaceDetectorYN.create(os.path.join(MODELS, "yunet.onnx"), "", (320, 240), 0.75, 0.3, 50)
REC = cv2.FaceRecognizerSF.create(os.path.join(MODELS, "sface.onnx"), "")
SFACE_THRESHOLD = 0.363   # OpenCV's documented cosine threshold for SFace
VIS: dict[str, Any] = {"present": False, "since": 0.0, "last_seen": 0.0, "engaged_since": 0.0, "session": None,
                       "votes": collections.deque(maxlen=6), "identity": "unknown", "confidence": None, "greeted": {}, "last_faces": [],
                       "pending": []}   # embeddings of an engaged unknown face, waiting to become User N
SIM = {"identity": None}   # UI override when there is no camera: None = auto, "unknown", "user1", ...

CV_LOCK = threading.Lock()   # YuNet/SFace objects are not thread-safe; frames and snapshots run in worker threads
def detect_faces(img: np.ndarray) -> list[dict]:
    if img is None: return []
    with CV_LOCK:
        h, w = img.shape[:2]; DET.setInputSize((w, h))
        _, faces = DET.detect(img)
    out = []
    for f in (faces if faces is not None else []):
        x, y, bw, bh = f[:4]
        out.append({"box": [round(float(x) / w, 3), round(float(y) / h, 3), round(float(bw) / w, 3), round(float(bh) / h, 3)],
                    "score": round(float(f[14]), 2), "raw": f})
    out.sort(key=lambda d: -d["box"][2]); return out

def embed(img: np.ndarray, raw) -> np.ndarray:
    with CV_LOCK: return REC.feature(REC.alignCrop(img, raw))

def recognize(feat: np.ndarray) -> tuple[str, float]:
    best, best_s = "unknown", 0.0
    for uid, u in USERS.items():
        for e in u["embeddings"]:
            s = float(REC.match(feat, np.array(e, dtype=np.float32).reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
            if s > best_s: best, best_s = uid, s
    return (best if best_s >= SFACE_THRESHOLD else "unknown"), round(best_s, 3)

def presence_update(faces: list[dict], img: np.ndarray) -> dict:
    """present -> engaged (close enough, long enough) -> identity confirmed by vote (or auto-registered) -> greet once per session."""
    now = time.time(); greet = None
    if faces:
        f = faces[0]; size = f["box"][2]
        VIS["last_seen"] = now
        if not VIS["present"]:
            VIS.update(present=True, since=now, engaged_since=0.0); VIS["votes"].clear(); VIS["pending"] = []
            emit("presence.detected", faces=len(faces), size=size)
        if size >= 0.11 and f["score"] >= 0.8 and f["raw"] is not None and not SIM["identity"]:
            feat = embed(img, f["raw"]); name, s = recognize(feat); VIS["votes"].append((name, s))
            if name == "unknown" and AUTO["remember"] and size >= 0.13 and f["score"] >= 0.9:
                VIS["pending"].append([round(float(v), 5) for v in feat.flatten()])
        votes = list(VIS["votes"])[-5:]
        names = [n for n, _ in votes]
        if names:
            top = max(set(names), key=names.count)
            if names.count(top) >= 3 and top != VIS["identity"]:
                VIS["identity"] = top; VIS["confidence"] = round(max(s for n, s in votes if n == top), 3) if top != "unknown" else None
                emit("identity.confirmed" if top != "unknown" else "identity.unknown", identity=top, confidence=VIS["confidence"], votes=names)
        engaged = size >= 0.12 and (now - VIS["since"]) >= 0.7
        if engaged and not VIS["engaged_since"]: VIS["engaged_since"] = now
        # an engaged, unknown, well-seen face becomes User N (snapshot = a few embeddings, no image is stored)
        if engaged and VIS["identity"] == "unknown" and AUTO["remember"] and len(VIS["pending"]) >= 5 and not SIM["identity"]:
            u = register_user(VIS["pending"], "auto on approach"); uid = next(k for k, v in USERS.items() if v is u)
            VIS["pending"] = []; VIS["votes"].clear(); VIS["votes"].extend([(uid, 1.0)] * 3); VIS["identity"] = uid; VIS["confidence"] = 1.0
            emit("identity.confirmed", identity=uid, confidence=1.0, votes=["registered"])
        waiting_for_snapshot = AUTO["remember"] and VIS["identity"] == "unknown" and not SIM["identity"] and now - VIS["since"] < 3.5
        if engaged and not VIS["session"] and not waiting_for_snapshot and (len(names) >= 3 or now - VIS["since"] > 2.0):
            VIS["session"] = uuid.uuid4().hex[:8]
            ident = VIS["identity"]
            HISTORY.clear()
            emit("session.started", session=VIS["session"], identity=ident, confidence=VIS["confidence"], approach_s=round(now - VIS["since"], 2), others=len(faces) - 1)
            VIS["greeted"][ident] = now; greet = ident
    elif VIS["present"] and now - VIS["last_seen"] > 3.0:
        end_presence(now)
    return {"present": VIS["present"], "engaged": bool(VIS["engaged_since"]) and VIS["present"], "identity": VIS["identity"], "confidence": VIS["confidence"],
            "session": VIS["session"], "greet": greet, "others": max(0, len(faces) - 1), "display": USERS.get(VIS["identity"], {}).get("display", "guest")}

# ───────────────────────────── Kokoro TTS + lip-sync features ─────────────────────────────
KOKORO = None
FILLERS: list[dict] = []   # pre-synthesized acknowledgements, played while the LLM is still writing a tool call
FILLER_TEXTS = ["Sure.", "Of course.", "On it.", "Certainly.", "One moment."]
def get_kokoro():
    global KOKORO
    if KOKORO is None:
        from kokoro_onnx import Kokoro
        KOKORO = Kokoro(os.path.join(MODELS, "kokoro-v1.0.int8.onnx"), os.path.join(MODELS, "voices-v1.0.bin"))
    return KOKORO

def lip_features(samples: np.ndarray, sr: int) -> dict:
    """Per-20ms-frame: mouth openness (energy, smoothed) and width class (spectral brightness)."""
    n = int(sr * FRAME_MS / 1000)
    frames = len(samples) // n
    x = samples[: frames * n].reshape(frames, n).astype(np.float32)
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-9)
    ref = max(np.percentile(rms, 95), 1e-4)
    open_ = np.clip((rms / ref - 0.12) / 0.88, 0, 1) ** 0.85   # gate low energy so the lips close between syllables
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * win, axis=1))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    cent = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-6)
    wide = np.clip((cent - 900) / 2600, 0, 1)
    wide[open_ < 0.08] = 0.3
    k = np.array([0.15, 0.7, 0.15])
    open_s = np.convolve(open_, k, mode="same"); wide_s = np.convolve(wide, np.array([0.2, 0.2, 0.2, 0.2, 0.2]), mode="same")
    return {"frame_ms": FRAME_MS, "open": [round(float(v), 3) for v in open_s], "wide": [round(float(v), 3) for v in wide_s]}

def synth(text: str) -> dict:
    t0 = time.time()
    samples, sr = get_kokoro().create(text, voice=VOICE, speed=1.0, lang="en-us")
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes()
    return {"type": "speech", "text": text, "sr": sr, "pcm16": base64.b64encode(pcm).decode(),
            "dur_s": round(len(samples) / sr, 3), "synth_s": round(time.time() - t0, 3), "feats": lip_features(samples, sr)}

TEXT_CALL_RE = re.compile(r"`*\b(home_action|remember_preference)\s*\(([^)]*)\)`*\s*")   # small models sometimes write the call as text
def text_calls(content: str) -> tuple[list[dict], str]:
    calls = [{"name": n, "arguments": dict(re.findall(r"(\w+)\s*=\s*[\"']([^\"']*)[\"']", a))} for n, a in TEXT_CALL_RE.findall(content)]
    return calls, TEXT_CALL_RE.sub("", content).strip()

SENT_RE = re.compile(r"^(.*?[.!?…]+[\"')\]]*)(\s+|$)", re.S)
CLAUSE_RE = re.compile(r"^(.{12,}?[,;:—–]|.{12,}? -)\s+", re.S)
def pop_sentences(buf: str, final: bool, first: bool = False) -> tuple[list[str], str]:
    out = []
    while True:
        m = SENT_RE.match(buf) or (first and not out and CLAUSE_RE.match(buf))
        if not m or (not final and m.end() == len(buf) and not buf.endswith((" ", "\n"))): break
        s = m.group(1).strip()
        if s: out.append(s)
        buf = buf[m.end():]
    if final and buf.strip(): out.append(buf.strip()); buf = ""
    return out, buf

# ───────────────────────────── Conversation ─────────────────────────────
app = FastAPI()
HISTORY: list[dict] = []
METRICS: list[dict] = []
LLM = {"think": None}   # whether the model accepts the think=false switch (probed at startup)
TURN = {"n": 0}

class Msg(BaseModel):
    text: str
    identity: str = "unknown"
    confidence: float | None = None
    others: int = 0
    greeting: bool = False

@app.post("/transcribe")
async def transcribe(request: Request):
    audio = await request.body()
    if len(audio) > 12 * 1024 * 1024:
        return JSONResponse({"error": "audio is too large"}, status_code=413)
    if not audio:
        return {"text": ""}
    media_type = request.headers.get("content-type", "audio/webm").split(";", 1)[0]
    extension = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/wav": "wav", "audio/mp4": "m4a", "audio/mpeg": "mp3"}.get(media_type, "webm")
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(f"{OLLAMA}/v1/audio/transcriptions",
                data={"model": STT_MODEL, "language": "en", "response_format": "json",
                      "prompt": "Transcribe the spoken words verbatim. Do not paraphrase or infer."},
                files={"file": (f"utterance.{extension}", audio, media_type)})
        if response.is_error:
            return JSONResponse({"error": f"Ollama transcription failed ({response.status_code}): {response.text[:400]}"}, status_code=502)
        text = (response.json().get("text") or "").strip()
        if not re.search(r"[A-Za-z0-9]", text) or re.fullmatch(r"[\[(].*[\])]", text): text = ""   # silence / "[noise]" style outputs
        emit("speech.transcribed", text=text, bytes=len(audio))
        return {"text": text}
    except Exception as exc:
        emit("speech.transcription_error", error=repr(exc))
        return JSONResponse({"error": str(exc)}, status_code=503)

async def ollama_stream(messages: list[dict], tools: bool):
    body = {"model": MODEL, "stream": True, "keep_alive": "60m", "messages": messages,
            "options": {"num_predict": 200, "temperature": 0.3, "num_ctx": 4096}}
    if LLM["think"] is False: body["think"] = False
    if tools: body["tools"] = TOOLS
    async with httpx.AsyncClient(timeout=180) as c:
        async with c.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
            async for line in r.aiter_lines():
                if line: yield json.loads(line)

def time_of_day() -> str:
    h = datetime.now().hour
    return "morning" if h < 12 else "afternoon" if h < 17 else "evening"

@app.post("/chat")
async def chat(m: Msg):
    turn_id = uuid.uuid4().hex[:8]
    identity = identity_line(m.identity, m.confidence, m.others)
    label = USERS.get(m.identity, {}).get("display")
    if m.greeting:
        who = f"{label} just approached the mirror" if label else "an unregistered guest just approached the mirror"
        user_text = (f"({who}, it is {time_of_day()}. Greet {'them as ' + label + ' ' if label else 'them warmly without any name '}in one short sentence, "
                     f"then ask exactly one friendly question offering something{' based on their preferences' if label else ''} (for example some music to welcome them home), then stop.)")
    else:
        user_text = m.text
    TURN["n"] += 1; my_turn = TURN["n"]   # a newer turn makes this one stale: it must not write history or run actions any more
    HISTORY.append({"role": "user", "content": f"{identity} {user_text}"}); del HISTORY[:-24]
    while HISTORY and HISTORY[0]["role"] == "tool": del HISTORY[0]   # trimming must not leave a tool result without its call
    emit("turn.user", turn=turn_id, text=m.text, identity=m.identity, greeting=m.greeting)

    async def gen():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        speech_q: asyncio.Queue = asyncio.Queue()
        t0 = time.time(); st = {"first_tok": None, "first_audio": None, "scheduled": 0}

        async def speech_pump():  # keeps sentence order, pushes audio the moment it is synthesized
            while True:
                fut = await speech_q.get()
                if fut is None: break
                sp = await fut
                if st["first_audio"] is None:
                    st["first_audio"] = round(time.time() - t0, 3); sp["first_audio_s"] = st["first_audio"]
                await q.put(json.dumps(sp) + "\n")

        def schedule(sentences):
            for s in sentences:
                clean = re.sub(r"[*_#`]+", "", TEXT_CALL_RE.sub("", s))
                if clean.strip():
                    st["scheduled"] += 1; speech_q.put_nowait(loop.run_in_executor(None, synth, clean))

        async def run_round(messages, tools):
            content, buf, calls, ch = "", "", [], {}
            async for ch in ollama_stream(messages, tools):
                if ch.get("error"): raise RuntimeError(f"ollama: {ch['error']}")
                msg = ch.get("message", {})
                tok = msg.get("content", "")
                if tok:
                    if st["first_tok"] is None:
                        st["first_tok"] = round(time.time() - t0, 3); emit("llm.first_token", turn=turn_id, latency_s=st["first_tok"])
                    content += tok; buf += tok
                    await q.put(json.dumps({"type": "token", "text": tok}) + "\n")
                    if not re.search(r"\b(home_action|remember_preference)\s*\([^)]*$", buf):   # never speak half of a text tool call
                        sents, buf = pop_sentences(buf, False, first=st["scheduled"] == 0); schedule(sents)
                for tc in msg.get("tool_calls", []) or []:
                    calls.append(tc["function"])
                if ch.get("done"):
                    await q.put(json.dumps({"type": "llm_done", "eval_count": ch.get("eval_count"), "tok_s": round(ch.get("eval_count", 0) / max(ch.get("eval_duration", 1) / 1e9, 1e-3), 1),
                                            "prompt_s": round(ch.get("prompt_eval_duration", 0) / 1e9, 2)}) + "\n")
            sents, buf = pop_sentences(buf, True); schedule(sents)
            if not calls:
                calls, content = text_calls(content)
                if calls: emit("llm.text_tool_call", turn=turn_id, tools=[c["name"] for c in calls])
            else: content = TEXT_CALL_RE.sub("", content).strip()
            if not content.strip() and not calls:
                emit("llm.empty_round", turn=turn_id, eval_count=ch.get("eval_count"), note="model produced neither text nor a parsable tool call")
            return content, calls

        CLAIM_RE = re.compile(r"\b(i'?ve|i have|i'?m|i am|now) (now )?(playing|started|turned|switched|dimmed|set|put on|paused|stopped|opened|closed)\b|\b(is|are) now (playing|on|off|set)\b", re.I)
        ACTION_RE = re.compile(r"\b(play |put on|put some|turn (on|off|up|down)|switch (on|off)|dim\b|brighten|lights? (on|off)|volume|louder|quieter|pause|resume|skip|next (track|song)|stop the|set the|open the|close the|remember that)", re.I)
        AFFIRM_RE = re.compile(r"^\W*(yes|yeah|yep|sure|please|ok|okay|go ahead|do it|sounds good|why not|absolutely|of course|that would be)\b", re.I)
        prev_assistant = next((h["content"] for h in reversed(HISTORY[:-1]) if h["role"] == "assistant" and h.get("content")), "")
        guest_private = (m.identity not in USERS) and re.search(r"\b(calendar|reminder|appointment|front door|camera|schedule)\b", m.text, re.I)
        CLOSER_RE = re.compile(r"\b(that'?s all|thanks|thank you|goodbye|bye|never mind|nevermind|no thanks|nothing else)\b", re.I)
        needs_action = (not m.greeting) and not guest_private and not CLOSER_RE.search(m.text) and bool(ACTION_RE.search(m.text) or (AFFIRM_RE.match(m.text) and re.search(r"\b(would you like|shall i|should i|want me to|how about|can i)\b", prev_assistant, re.I)))

        def live() -> bool: return TURN["n"] == my_turn

        async def execute(calls):
            for call in calls:
                if not live(): return
                args = call.get("arguments") or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: args = {}
                await q.put(json.dumps({"type": "tool_call", "name": call["name"], "args": args}) + "\n")
                ta = time.time()
                res = run_action({"tool": call["name"], "args": args, "turn": turn_id, "identity": m.identity})
                res["latency_ms"] = round((time.time() - ta) * 1000)
                await q.put(json.dumps({"type": "tool_result", "name": call["name"], "result": res}) + "\n")
                HISTORY.append({"role": "tool", "tool_name": call["name"], "content": json.dumps({k: v for k, v in res.items() if k in ("status", "device", "action", "value", "error")})})

        async def producer():
            try:
                content, calls = await run_round([{"role": "system", "content": SYSTEM}] + HISTORY, True)
                if not content.strip() and not calls:  # retry once, nudged
                    nudge = {"role": "user", "content": "(system: your previous output was empty or malformed. Answer the user now: call the right tool with valid JSON, or speak one short sentence.)"}
                    content, calls = await run_round([{"role": "system", "content": SYSTEM}] + HISTORY + [nudge], True)
                    if not content.strip() and not calls:
                        content = "Sorry, I missed that. Could you say it again?"; schedule([content])
                if not live(): return
                HISTORY.append({"role": "assistant", "content": content, **({"tool_calls": [{"function": c} for c in calls]} if calls else {})})
                if calls:
                    await execute(calls)
                    if len(content.strip()) < 60:  # no real spoken confirmation yet -> second round with tool results
                        content2, _ = await run_round([{"role": "system", "content": SYSTEM}] + HISTORY, True)
                        if not live(): return
                        HISTORY.append({"role": "assistant", "content": content2}); content = (content + " " + content2).strip()
                elif not m.greeting and (CLAIM_RE.search(content) or needs_action):
                    emit("policy.unverified_claim" if CLAIM_RE.search(content) else "policy.missing_tool_call", turn=turn_id, text=content.strip(),
                         user_text=m.text, action="repair: forcing tool call round")
                    nudge = {"role": "user", "content": f"(system: the user asked for an action (\"{m.text}\") but you called no tool. Call the appropriate tool(s) now with valid JSON. Output no text.)"}
                    _, calls = await run_round([{"role": "system", "content": SYSTEM}] + HISTORY + [nudge], True)
                    if not live(): return
                    if calls:
                        HISTORY.append({"role": "assistant", "content": "", "tool_calls": [{"function": c} for c in calls]})
                        await execute(calls); emit("policy.repaired", turn=turn_id, tools=[c["name"] for c in calls])
                    else:
                        emit("policy.repair_failed", turn=turn_id)
                wd.cancel(); await speech_q.put(None); await pump
                total = round(time.time() - t0, 3)
                metr = {"turn": turn_id, "first_token_s": st["first_tok"], "first_audio_s": st["first_audio"], "total_s": total, "tools": len(calls)}
                METRICS.append(metr)
                emit("turn.assistant", turn=turn_id, text=content.strip(), tools=[c["name"] for c in calls], metrics=metr)
                await q.put(json.dumps({"type": "done", **metr, "tool_names": [c["name"] for c in calls], "text": content.strip()}) + "\n")
            except asyncio.CancelledError: raise
            except Exception as e:
                emit("turn.error", turn=turn_id, error=repr(e)); await q.put(json.dumps({"type": "error", "error": repr(e)}) + "\n")
            finally:
                wd.cancel(); speech_q.put_nowait(None)
                if not pump.done(): pump.cancel()
                await q.put(None)

        async def filler_watchdog():  # nothing spoken after 2.5 s => model is writing tool JSON; acknowledge out loud
            await asyncio.sleep(2.5)
            if st["scheduled"] == 0 and st["first_tok"] is None and FILLERS and not m.greeting:
                f = dict(FILLERS[int(time.time()) % len(FILLERS)]); f["filler"] = True
                st["scheduled"] += 1; fut = loop.create_future(); fut.set_result(f); speech_q.put_nowait(fut)
                emit("speech.filler", turn=turn_id, text=f["text"], at_s=round(time.time() - t0, 3))
        pump = asyncio.create_task(speech_pump()); prod = asyncio.create_task(producer()); wd = asyncio.create_task(filler_watchdog())
        try:
            while True:
                item = await q.get()
                if item is None: break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            emit("turn.cancelled", turn=turn_id, at_s=round(time.time() - t0, 3))
            prod.cancel(); pump.cancel(); wd.cancel(); raise
    return StreamingResponse(gen(), media_type="application/x-ndjson")

class Interrupt(BaseModel):
    reason: str
    stop_ms: float | None = None
    spoken_text: str | None = None

@app.post("/interrupt")
async def interrupt(i: Interrupt):
    return emit("speech.interrupted", reason=i.reason, stop_ms=i.stop_ms, was_saying=i.spoken_text)

def end_presence(now: float):
    VIS["present"] = False
    emit("presence.left", after_s=round(now - VIS["since"], 1))
    if VIS["session"]:
        emit("session.ended", session=VIS["session"], identity=VIS["identity"], duration_s=round(now - VIS["since"], 1)); VIS["session"] = None
    VIS["identity"] = "unknown"; VIS["confidence"] = None; VIS["votes"].clear(); VIS["engaged_since"] = 0.0; VIS["pending"] = []
    HISTORY.clear()   # the next visitor must not inherit this conversation

async def presence_watchdog():   # frames stop arriving (tab closed, camera off) => the visitor is gone
    while True:
        await asyncio.sleep(1.0)
        if VIS["present"] and time.time() - VIS["last_seen"] > 3.0: end_presence(time.time())

@app.post("/reset")
async def reset():
    """Forget the conversation and the current visit (users stay registered until /users/reset)."""
    HISTORY.clear(); VIS["greeted"].clear(); SIM["identity"] = None
    if VIS["present"]: end_presence(time.time())
    emit("session.reset"); return {"ok": True}

# ───────────────────────────── Vision endpoints ─────────────────────────────
class Frame(BaseModel):
    jpeg: str            # base64 JPEG from the browser camera
    simulate: str | None = None   # UI override: "unknown" | user id | None (auto)

def decode_jpeg(b64: str) -> np.ndarray | None:
    try: return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    except Exception: return None

@app.post("/vision/frame")
async def vision_frame(f: Frame):
    t = time.time()
    img = await asyncio.get_running_loop().run_in_executor(None, decode_jpeg, f.jpeg)
    faces = await asyncio.get_running_loop().run_in_executor(None, detect_faces, img)
    SIM["identity"] = f.simulate or None
    if SIM["identity"] and faces:   # simulated recognition: camera presence is real, the identity is forced from the UI
        VIS["votes"].clear(); VIS["votes"].extend([(SIM["identity"], 0.0)] * 5)
        if VIS["identity"] != SIM["identity"]:
            VIS["identity"] = SIM["identity"]; VIS["confidence"] = None
    pres = presence_update(faces, img)
    VIS["last_faces"] = [{"box": d["box"], "score": d["score"]} for d in faces]
    return {**pres, "faces": VIS["last_faces"], "ms": round((time.time() - t) * 1000, 1), "users": {k: len(v["embeddings"]) for k, v in USERS.items()}}

@app.post("/vision/simulate")
async def vision_simulate(e: dict):
    """No camera: fake an approach of the chosen identity through the same session logic."""
    ident = e.get("identity") or "unknown"
    SIM["identity"] = ident
    VIS.update(present=True, since=time.time() - 2.5, last_seen=time.time() + 60, engaged_since=time.time() - 1)   # stays "present" ~1 min without camera frames
    VIS["votes"].clear(); VIS["votes"].extend([(ident, 0.0)] * 5); VIS["identity"] = ident; VIS["confidence"] = None
    if VIS["session"] and (e.get("fresh", True) or e.get("force_greet", False)): VIS["session"] = None
    emit("presence.detected", faces=1, size=0.2, simulated=True)
    pres = presence_update([{"box": [0.4, 0.3, 0.2, 0.27], "score": 0.99, "raw": None}], np.zeros((2, 2, 3), np.uint8))
    return pres

class Snapshot(BaseModel):
    frames: list[str]

@app.post("/users/snapshot")
async def users_snapshot(s: Snapshot):
    """Register the face in front of the camera as the next User N (explicit button). Embeddings only, runtime memory."""
    def work():
        feats = []
        for b64 in s.frames[:12]:
            img = decode_jpeg(b64); faces = detect_faces(img)
            if faces and faces[0]["box"][2] >= 0.1:
                feats.append([round(float(v), 5) for v in embed(img, faces[0]["raw"]).flatten()])
        return feats
    feats = await asyncio.get_running_loop().run_in_executor(None, work)
    if len(feats) < 3:
        return JSONResponse({"ok": False, "error": f"need a clear face in at least 3 frames, got {len(feats)}"}, status_code=400)
    known, score = recognize(np.array(feats[-1], dtype=np.float32).reshape(1, -1))
    if known != "unknown":
        emit("user.already_registered", identity=known, confidence=score)
        return {"ok": True, "identity": known, "display": USERS[known]["display"], "already": True, "confidence": score}
    u = register_user(feats, "snapshot button"); uid = next(k for k, v in USERS.items() if v is u)
    VIS["votes"].clear(); VIS["identity"] = "unknown"; VIS["pending"] = []
    return {"ok": True, "identity": uid, "display": u["display"], "samples": len(u["embeddings"])}

@app.post("/users/simulate")
async def users_simulate():
    """No camera: register a placeholder User N with no face samples, so scenarios can run as a registered user."""
    u = register_user([], "simulated (no camera)"); uid = next(k for k, v in USERS.items() if v is u)
    return {"ok": True, "identity": uid, "display": u["display"]}

@app.get("/users")
async def users_list():
    return {"auto_remember": AUTO["remember"], "users": {k: {"display": v["display"], "samples": len(v["embeddings"]), "prefs": v["prefs"], "how": v["how"], "registered_at": v["registered_at"]} for k, v in USERS.items()}}

@app.post("/users/reset")
async def users_reset():
    n = len(USERS); USERS.clear(); VIS["greeted"].clear(); VIS["votes"].clear(); VIS["identity"] = "unknown"; VIS["pending"] = []; HISTORY.clear()
    emit("users.reset", forgotten=n); return {"ok": True, "forgotten": n}

@app.post("/users/forget")
async def users_forget(e: dict):
    uid = e.get("identity")
    if uid in USERS:
        USERS.pop(uid); VIS["greeted"].pop(uid, None); emit("user.forgotten", identity=uid)
    return {"ok": True}

@app.post("/users/auto")
async def users_auto(e: dict):
    AUTO["remember"] = bool(e.get("enabled", True)); emit("users.auto_remember", enabled=AUTO["remember"]); return {"auto_remember": AUTO["remember"]}

# ───────────────────────────── Mock action executor + event stream ─────────────────────────────
def run_action(body: dict) -> dict:
    """Nothing is controlled: the action is validated, logged (server log + browser console) and acknowledged."""
    tool, args, identity = body.get("tool"), body.get("args") or {}, body.get("identity") or "unknown"
    try:
        out = apply_action(tool, args, identity)
        ev = emit("action.mock", tool=tool, identity=identity, turn=body.get("turn"), simulated=True, **out)
        return {"status": "ok", "simulated": True, **out, "event_id": ev["id"]}
    except Exception as e:
        emit("action.failed", tool=tool, args=args, error=str(e), turn=body.get("turn"))
        return {"status": "error", "simulated": True, "error": str(e)}

@app.get("/events")
async def events(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=200); SUBS.add(q)
    async def stream():
        try:
            yield f"event: snapshot\ndata: {json.dumps({'recent': EVENTS[-30:]})}\n\n"
            while True:
                if await request.is_disconnected(): break
                try:
                    ev = await asyncio.wait_for(q.get(), 15); yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError: yield ": keepalive\n\n"
        finally: SUBS.discard(q)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

@app.get("/metrics")
async def metrics(): return METRICS

@app.on_event("startup")
async def warm():
    loop = asyncio.get_running_loop()
    for t in FILLER_TEXTS: FILLERS.append(await loop.run_in_executor(None, synth, t))
    asyncio.create_task(presence_watchdog())
    print(f"READY model={MODEL} voice={VOICE} ollama={OLLAMA}", flush=True)
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            for think in (False, None):
                body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}], "tools": TOOLS, "stream": False, "keep_alive": "60m", "options": {"num_predict": 1}}
                if think is False: body["think"] = False
                r = await c.post(f"{OLLAMA}/api/chat", json=body)
                if r.status_code == 200: LLM["think"] = think; break
                print("LLM probe", think, r.status_code, r.text[:200], flush=True)
            print(f"LLM warm think={LLM['think']}", flush=True)
    except Exception as e: print("LLM warm failed:", e, flush=True)

app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
