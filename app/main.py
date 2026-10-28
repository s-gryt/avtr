"""Aria — mirror concierge POC. Single-file backend.

camera frames -> OpenCV YuNet (detect) + SFace (recognize, anonymous runtime registry: User 1, User 2, ...)
             -> presence/visit state machine (greet once per run session, "anything I can do for you?" on return)
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

Each user message starts with a [visitor: ...] line. A registered visitor is "User 1" (or User N); use that label only in greetings. A guest is unregistered: never use a name and never reveal stored preferences; you cannot remember anything for a guest, so if a guest asks, kindly say you only remember things for registered household members (never claim you noted it). "Thanks", "bye", "that's all": a short warm goodbye, no question, no tool."""

# ───────────────────────────── Users + run session: anonymous, runtime-only face memory ─────────────────────────────
USERS: dict[str, dict] = {}        # registered household users: "user1": {"display": "User 1", "embeddings": [...], "prefs": {...}, ...}; kept until /users/reset or restart
# Nothing is ever stored for a face that is not registered: it is simply "guest" for one visit (no profile, no face samples),
# greeted on every approach, and its conversation is dropped when the visit ends. Only "Snapshot → User" registers a face.
GUEST = "guest"
# One run session = Start mirror .. Stop. Who was greeted, guests and conversations live only here, in memory.
RUN: dict[str, Any] = {"id": None, "started": 0.0, "greeted": {}, "last_visitor": None}   # greeted: registered users only
HISTORIES: dict[str, list[dict]] = {}   # per visitor key (user1, guest): a returning user continues their conversation; a guest's is dropped when they leave

def next_user_id() -> str:
    n = 1
    while f"user{n}" in USERS: n += 1
    return f"user{n}"

def register_user(embeddings: list[list[float]], how: str) -> str:
    uid = next_user_id()
    USERS[uid] = {"display": f"User {uid[4:]}", "embeddings": embeddings[-16:], "prefs": {}, "registered_at": round(time.time(), 3), "how": how}
    emit("user.registered", identity=uid, display=USERS[uid]["display"], samples=len(USERS[uid]["embeddings"]), how=how, stored="runtime memory only, embeddings, no images")
    return uid

def display_of(key: str | None) -> str:
    return (USERS.get(key) or {}).get("display") or "Guest"

def rename_visitor(old: str, new: str):
    """The guest in front was registered with the snapshot button: they were already greeted, keep their conversation."""
    RUN["greeted"][new] = time.time()
    if RUN["last_visitor"] == old: RUN["last_visitor"] = new
    if old in HISTORIES: HISTORIES[new] = HISTORIES.pop(old)
    if VIS["visit"] and VIS["visit"]["key"] == old: VIS["visit"]["key"] = new; VIS["identity"] = new
    VIS["votes"].clear()

def history_for(key: str) -> list[dict]:
    return HISTORIES.setdefault(key or "unknown", [])

def identity_line(identity: str, confidence: float | None, others: int) -> str:
    u = USERS.get(identity)
    others_s = f", {others} other people present (no private info)" if others else ", alone"
    if not u:
        return f"[visitor: guest, not registered — no name, no private info{others_s}]"
    prefs = "; ".join(f"{k}: {v}" for k, v in u["prefs"].items()) or "none stored yet"
    conf = f", face match {confidence:.2f}" if confidence else ", simulated recognition"
    return f"[visitor: {u['display']}, registered household user{conf}{others_s}. Preferences: {prefs}]"

# ───────────────────────────── Vision: YuNet detect + SFace recognize + presence/visit state machine ─────────────────────────────
DET = cv2.FaceDetectorYN.create(os.path.join(MODELS, "yunet.onnx"), "", (320, 240), 0.75, 0.3, 50)
REC = cv2.FaceRecognizerSF.create(os.path.join(MODELS, "sface.onnx"), "")
SFACE_THRESHOLD = 0.363   # OpenCV's documented cosine threshold for SFace
CLEARLY_NEW = 0.28        # below this vs. everyone known, an unknown face is safely a guest (not a registered user at a bad angle)
AWAY_S = 1.2              # face gone this long = the visit ends; coming back later in the run = "anything I can do for you?"
SURE_MATCH = 0.5          # a single SFace match this strong identifies a known face immediately
REACQUIRE_S = 0.8         # face lost at least this long and back = verify who it is again (someone else may have stepped in)
VIS: dict[str, Any] = {"present": False, "since": 0.0, "last_seen": 0.0, "visit": None, "votes": collections.deque(maxlen=8),
                       "identity": "unknown", "confidence": None, "last_faces": [], "reverify": False, "decide_since": 0.0}
SIM = {"identity": None}   # UI override: None = camera recognition, "unknown" = a new guest, or a user/guest key

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

def _best(feat: np.ndarray, registry: dict) -> tuple[str, float]:
    best, best_s = "unknown", 0.0
    for uid, u in registry.items():
        for e in u["embeddings"]:
            s = float(REC.match(feat, np.array(e, dtype=np.float32).reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
            if s > best_s: best, best_s = uid, s
    return best, best_s

def recognize(feat: np.ndarray) -> tuple[str, float, float]:
    """-> (registered user key or "unknown", score of that match, best score against any registered user)."""
    u, us = _best(feat, USERS)
    return (u if us >= SFACE_THRESHOLD else "unknown"), round(us, 3), round(us, 3)

def decide(votes: list[tuple[str, float]], waited_s: float, need_unknown: int = 3, allow_timeout: bool = True) -> str | None:
    """Who is in front of the mirror, from the last few recognitions. None = not sure yet (do not greet anybody yet).
    "unknown" needs faces that are clearly nobody known (score < CLEARLY_NEW), so a registered user at a bad angle is not taken for a guest."""
    recent = votes[-5:]
    known = collections.Counter(n for n, _ in recent if n != "unknown")
    if known:
        strong = [(n, s) for n, s in recent if n != "unknown" and s >= SURE_MATCH]
        if strong: return max(strong, key=lambda t: t[1])[0]   # one clear match is enough: no waiting when someone known comes back
        top, c = max(known.items(), key=lambda kv: (kv[1], max(s for n, s in recent if n == kv[0])))
        if c >= 2: return top
    clear = sum(1 for n, s in recent if n == "unknown" and s < CLEARLY_NEW)
    if clear >= need_unknown and len(known) == 0: return "unknown"
    if allow_timeout and waited_s > 3.5: return "unknown"   # face too small/blurred to embed: treat as a guest rather than never greeting
    return None

def start_visit(key: str, now: float, others: int, simulated: bool = False) -> dict:
    """Someone is confirmed in front of the mirror. A registered user is greeted once per run, afterwards only followed up;
    anyone not registered is a guest and greeted on every approach (nothing about them is stored)."""
    if key not in USERS: key = GUEST
    returning = key in USERS and key in RUN["greeted"]
    someone_between = returning and RUN["last_visitor"] not in (None, key)
    kind = "return" if returning else "greet"
    if returning:
        who = USERS.get(key, {}).get("display")
        text = (f"Welcome back{', ' + who if who else ''}. Anything I can do for you?") if someone_between else "Anything I can do for you?"
    else:
        text = None
        if key in USERS: RUN["greeted"][key] = now
    VIS["visit"] = {"id": uuid.uuid4().hex[:8], "key": key, "started": now, "kind": kind, "text": text}
    VIS["identity"] = key; VIS["votes"].clear(); VIS["votes"].extend([(key, VIS["confidence"] or 1.0)] * 3)
    emit("visit.started", visit=VIS["visit"]["id"], identity=key, display=display_of(key), kind=kind, confidence=VIS["confidence"],
         approach_s=round(now - VIS["since"], 2), others=others, run=RUN["id"])
    if kind == "return": emit("greeting.suppressed", identity=key, reason="already greeted in this run session", says=text)
    return {"kind": kind, "identity": key, "visit": VIS["visit"]["id"], "text": text}

def end_visit(now: float, reason: str):
    v = VIS["visit"]
    if not v: return
    RUN["last_visitor"] = v["key"]
    emit("visit.ended", visit=v["id"], identity=v["key"], display=display_of(v["key"]), reason=reason, duration_s=round(now - v["started"], 1))
    VIS["visit"] = None; VIS["identity"] = "unknown"; VIS["confidence"] = None; VIS["votes"].clear()
    VIS["decide_since"] = now; VIS["reverify"] = False
    if v["key"] == GUEST: HISTORIES.pop(GUEST, None)   # the next guest is a new person with a new conversation

def end_presence(now: float, reason: str = "left"):
    if VIS["present"]:
        VIS["present"] = False
        emit("presence.left", after_s=round(now - VIS["since"], 1))
    end_visit(now, reason)

def presence_update(faces: list[dict], img: np.ndarray) -> dict:
    """present -> identity confirmed by votes -> visit (greet once per run session, short "anything I can do for you?" on return).
    A face lost for a moment and back is verified again, so User 1 leaving and a guest stepping in starts a new visit."""
    now = time.time(); greet = None
    if faces:
        f = faces[0]; size = f["box"][2]
        if not VIS["present"]:
            VIS.update(present=True, since=now, decide_since=now, reverify=False); VIS["votes"].clear()
            emit("presence.detected", faces=len(faces), size=size)
        elif now - VIS["last_seen"] > REACQUIRE_S and VIS["visit"] and VIS["visit"]["key"] == GUEST:
            # an unregistered face cannot be verified: a guest who stepped out and back is a new guest -> greeted again
            end_visit(now, "guest stepped out"); VIS["since"] = now
        elif now - VIS["last_seen"] > REACQUIRE_S:
            VIS["votes"].clear(); VIS["decide_since"] = now; VIS["reverify"] = bool(VIS["visit"])
        VIS["last_seen"] = now
        sim = SIM["identity"]; v = VIS["visit"]
        if sim:
            VIS["votes"].append((sim, 1.0))
        elif size >= 0.09 and f["score"] >= 0.75 and f["raw"] is not None:
            feat = embed(img, f["raw"]); name, s, best_any = recognize(feat)
            VIS["votes"].append((name, s))
        votes = list(VIS["votes"]); waited = now - VIS["decide_since"]
        if sim: who = sim
        elif v and VIS["reverify"]:
            who = decide(votes, waited, need_unknown=4, allow_timeout=False)
            if who is None and waited > 5.0: who = v["key"]   # never sure: keep the current visit
        else: who = decide(votes, waited)
        if who and who != "unknown":
            VIS["confidence"] = None if sim else (round(max((s for n, s in votes if n == who), default=0.0), 3) or None)
        engaged = size >= 0.09 and now - VIS["since"] >= 0.25
        if v and VIS["reverify"] and who:
            VIS["reverify"] = False
            if not (who == v["key"] or (who == "unknown" and v["key"] == GUEST)):   # someone else stepped in while the face was out of view
                emit("visitor.changed", previous=v["key"], now=who); end_visit(now, "someone else stepped in"); v = None
        elif v and who and who not in (v["key"], "unknown") and not VIS["reverify"]:
            emit("visitor.changed", previous=v["key"], now=who); end_visit(now, "recognized someone else"); v = None
        if not v and engaged and who:
            greet = start_visit(who, now, len(faces) - 1, simulated=bool(sim))
    elif VIS["present"] and now - VIS["last_seen"] > AWAY_S:
        end_presence(now)
    v = VIS["visit"]
    return {"present": VIS["present"], "engaged": bool(v), "identity": v["key"] if v else "unknown", "confidence": VIS["confidence"],
            "session": RUN["id"], "visit": v["id"] if v else None, "greet": greet, "others": max(0, len(faces) - 1),
            "display": display_of(v["key"]) if v else "—", "registered": bool(v and v["key"] in USERS)}

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
    # ~60 ms smoothing: lips follow syllables (4-7 Hz), not 20 ms energy jitter, which reads as a flickering mouth
    open_s = np.convolve(open_, np.array([0.08, 0.2, 0.44, 0.2, 0.08]), mode="same"); wide_s = np.convolve(wide, np.full(7, 1 / 7), mode="same")
    return {"frame_ms": FRAME_MS, "open": [round(float(v), 3) for v in open_s], "wide": [round(float(v), 3) for v in wide_s]}

def synth(text: str) -> dict:
    t0 = time.time()
    samples, sr = get_kokoro().create(text, voice=VOICE, speed=1.0, lang="en-us")
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes()
    return {"type": "speech", "text": text, "sr": sr, "pcm16": base64.b64encode(pcm).decode(),
            "dur_s": round(len(samples) / sr, 3), "synth_s": round(time.time() - t0, 3), "feats": lip_features(samples, sr)}

# ───────────────────────────── Speech detection (Silero VAD, ONNX, CPU) ─────────────────────────────
# Loudness alone cannot tell a voice from a clap, a door, dishes, music or a cat. The browser streams its mic (16 kHz) here
# and gets a speech probability per 32 ms frame; only frames that are speech can interrupt her or become a turn.
VAD_SR, VAD_FRAME, VAD_CTX = 16000, 512, 64
SILERO = {"sess": None, "tabs": {}}   # per browser tab: recurrent state, last 64 samples of context, leftover samples
def silero():
    if SILERO["sess"] is None:
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        SILERO["sess"] = ort.InferenceSession(os.path.join(MODELS, "silero_vad.onnx"), o, providers=["CPUExecutionProvider"])
    return SILERO["sess"]

def vad_probs(tab: str, pcm: np.ndarray, reset: bool) -> list[float]:
    tabs = SILERO["tabs"]; now = time.time()
    for k in [k for k, v in tabs.items() if now - v["t"] > 300]: del tabs[k]
    st = tabs.get(tab)
    if st is None or reset:
        st = tabs[tab] = {"state": np.zeros((2, 1, 128), np.float32), "ctx": np.zeros(VAD_CTX, np.float32), "rest": np.zeros(0, np.float32), "t": now}
    st["t"] = now
    x = np.concatenate([st["rest"], pcm]); n = len(x) // VAD_FRAME; out = []
    sess, sr = silero(), np.array(VAD_SR, np.int64)
    for i in range(n):
        f = x[i * VAD_FRAME:(i + 1) * VAD_FRAME]
        p, st["state"] = sess.run(None, {"input": np.concatenate([st["ctx"], f])[None], "state": st["state"], "sr": sr})
        st["ctx"] = f[-VAD_CTX:]; out.append(round(float(p[0][0]), 3))
    st["rest"] = x[n * VAD_FRAME:]
    return out

# small models sometimes write the call as text: home_action(device="music", ...) or Gemma's home_action{device:<|"|>music<|"|>, ...}
TEXT_CALL_RE = re.compile(r"`*\b(home_action|remember_preference)\s*[({]([^)}]*)[)}]`*\s*")
SPECIAL_TOK_RE = re.compile(r"<\|[^|<>]*\|>")
def untok(s: str) -> str:   # Gemma's <|"|> quote token -> a real quote; any other special token is dropped
    return SPECIAL_TOK_RE.sub("", s.replace('<|"|>', '"'))
def text_calls(content: str) -> tuple[list[dict], str]:
    content = untok(content)
    calls = [{"name": n, "arguments": dict(re.findall(r"(\w+)\s*[=:]\s*[\"']([^\"']*)[\"']", a))} for n, a in TEXT_CALL_RE.findall(content)]
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
METRICS: list[dict] = []
LLM = {"think": None}   # whether the model accepts the think=false switch (probed at startup)
TURN = {"n": 0}

class Msg(BaseModel):
    text: str
    identity: str = "unknown"
    confidence: float | None = None
    others: int = 0
    greeting: bool | str = False   # True/"greet" = first greeting of this run session, "return" = came back: short fixed line, no LLM

NOISE_TEXT_RE = re.compile(r"\W*((uh|um|hm|mm|ah|er|eh|oh)+[hm]*|you|thanks for watching|subtitles.*|\.+)\W*", re.I)   # what audio models hear in noise/breaths
# On silence/noise the audio LLM tends to echo its own instruction ("What exact words are spoken in this audio?") or describe the clip.
PROMPT_ECHO_RE = re.compile(r"(exact words|words (are|were) (spoken|said)|spoken words|this audio|the audio|audio (clip|recording|file)|transcri(be|ption|pt)|verbatim|paraphrase|no (speech|words|audio)|(is|was) (silent|silence)|inaudible|unintelligible|background noise)", re.I)

@app.post("/vad")
async def vad(request: Request):
    """16 kHz mono int16 PCM from the browser mic -> speech probability per 32 ms frame (stateful per tab)."""
    body = await request.body()
    if len(body) > 64000: return JSONResponse({"error": "chunk too large"}, status_code=413)
    tab = re.sub(r"[^a-z0-9]", "", request.headers.get("x-tab", "")[:16].lower()) or "default"
    pcm = np.frombuffer(body[: len(body) // 2 * 2], "<i2").astype(np.float32) / 32768
    return {"p": vad_probs(tab, pcm, request.headers.get("x-reset") == "1")}

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
        raw = text
        if not re.search(r"[A-Za-z0-9]", text) or re.fullmatch(r"[\[(].*[\])]", text) or NOISE_TEXT_RE.fullmatch(text): text = ""   # silence / "[noise]" / filler-only outputs
        elif PROMPT_ECHO_RE.search(text): text = ""
        if raw and not text:
            emit("speech.rejected", heard=raw[:120], bytes=len(audio))
        emit("speech.transcribed", text=text, bytes=len(audio))
        return {"text": text}
    except Exception as exc:
        emit("speech.transcription_error", error=repr(exc))
        return JSONResponse({"error": str(exc)}, status_code=503)

GUEST_TOOLS = [t for t in TOOLS if t["function"]["name"] != "remember_preference"]   # a guest's face is not a household profile: nothing to remember them by

async def ollama_stream(messages: list[dict], tools: bool | list):
    body = {"model": MODEL, "stream": True, "keep_alive": "60m", "messages": messages,
            "options": {"num_predict": 200, "temperature": 0.3, "num_ctx": 4096}}
    if LLM["think"] is False: body["think"] = False
    if tools: body["tools"] = tools if isinstance(tools, list) else TOOLS
    async with httpx.AsyncClient(timeout=180) as c:
        async with c.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
            async for line in r.aiter_lines():
                if line: yield json.loads(line)

def time_of_day() -> str:
    h = datetime.now().hour
    return "morning" if h < 12 else "afternoon" if h < 17 else "evening"

RETURN_TEXT = "Anything I can do for you?"
BUILD = uuid.uuid4().hex[:8]   # changes on every server start: open pages reload instead of running stale code
LINE_CACHE: dict[str, dict] = {}

async def return_line(identity: str):
    """The visitor came back within this run session: no second greeting, one short fixed line (same speech/lip path, no LLM)."""
    v = VIS["visit"]
    text = (v or {}).get("text") if v and v["key"] == identity else None
    text = text or RETURN_TEXT
    TURN["n"] += 1; t0 = time.time()
    if text not in LINE_CACHE:
        LINE_CACHE[text] = await asyncio.get_running_loop().run_in_executor(None, synth, text)
    sp = {**LINE_CACHE[text], "first_audio_s": round(time.time() - t0, 3)}
    history_for(identity).append({"role": "assistant", "content": text})
    emit("turn.assistant", turn=uuid.uuid4().hex[:8], text=text, tools=[], kind="return")
    yield json.dumps(sp) + "\n"
    yield json.dumps({"type": "done", "turn": None, "first_token_s": None, "first_audio_s": sp["first_audio_s"], "total_s": round(time.time() - t0, 3), "tools": 0, "tool_names": [], "text": text, "kind": "return"}) + "\n"

@app.post("/chat")
async def chat(m: Msg):
    turn_id = uuid.uuid4().hex[:8]
    identity = identity_line(m.identity, m.confidence, m.others)
    label = USERS.get(m.identity, {}).get("display")
    kind = "greet" if m.greeting is True else (m.greeting or "")
    HISTORY = history_for(m.identity)
    if kind == "return":
        return StreamingResponse(return_line(m.identity), media_type="application/x-ndjson")
    if kind:
        who = f"{label} just approached the mirror" if label else "an unregistered guest just approached the mirror"
        user_text = (f"({who}, it is {time_of_day()}. Greet {'them as ' + label + ' ' if label else 'them warmly without any name '}in one short sentence, "
                     f"then ask exactly one friendly question offering something{' based on their preferences' if label else ''} (for example some music to welcome them home), then stop. Do not say 'welcome back'.)")
    else:
        user_text = m.text
    TURN["n"] += 1; my_turn = TURN["n"]   # a newer turn makes this one stale: it must not write history or run actions any more
    HISTORY.append({"role": "user", "content": f"{identity} {user_text}"}); del HISTORY[:-24]
    while HISTORY and HISTORY[0]["role"] == "tool": del HISTORY[0]   # trimming must not leave a tool result without its call
    emit("turn.user", turn=turn_id, text=m.text, identity=m.identity, greeting=kind or False)

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
                clean = re.sub(r"[*_#`]+", "", TEXT_CALL_RE.sub("", untok(s)))
                if clean.strip():
                    st["scheduled"] += 1; speech_q.put_nowait(loop.run_in_executor(None, synth, clean))

        async def run_round(messages, tools):
            content, buf, calls, ch = "", "", [], {}
            if tools and not registered: tools = GUEST_TOOLS
            async for ch in ollama_stream(messages, tools):
                if ch.get("error"): raise RuntimeError(f"ollama: {ch['error']}")
                msg = ch.get("message", {})
                tok = msg.get("content", "")
                if tok:
                    if st["first_tok"] is None:
                        st["first_tok"] = round(time.time() - t0, 3); emit("llm.first_token", turn=turn_id, latency_s=st["first_tok"])
                    content += tok; buf += tok
                    await q.put(json.dumps({"type": "token", "text": tok}) + "\n")
                    if not re.search(r"\b(home_action|remember_preference)\s*[({][^)}]*$", buf):   # never speak half of a text tool call
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
            else: content = TEXT_CALL_RE.sub("", untok(content)).strip()
            if not registered and any(c.get("name") == "remember_preference" for c in calls):
                emit("policy.guest_tool_blocked", turn=turn_id, tool="remember_preference"); calls = [c for c in calls if c.get("name") != "remember_preference"]
            if not content.strip() and not calls:
                emit("llm.empty_round", turn=turn_id, eval_count=ch.get("eval_count"), note="model produced neither text nor a parsable tool call")
            return content, calls

        CLAIM_RE = re.compile(r"\b(i'?ve|i have|i'?m|i am|now) (now )?(playing|started|turned|switched|dimmed|set|put on|paused|stopped|opened|closed)\b|\b(is|are) now (playing|on|off|set)\b", re.I)
        ACTION_RE = re.compile(r"\b(play |put on|put some|turn (on|off|up|down)|switch (on|off)|dim\b|brighten|lights? (on|off)|volume|louder|quieter|pause|resume|skip|next (track|song)|stop the|set the|open the|close the|remember that)", re.I)
        AFFIRM_RE = re.compile(r"^\W*(yes|yeah|yep|sure|please|ok|okay|go ahead|do it|sounds good|why not|absolutely|of course|that would be)\b", re.I)
        prev_assistant = next((h["content"] for h in reversed(HISTORY[:-1]) if h["role"] == "assistant" and h.get("content")), "")
        registered = m.identity in USERS
        guest_private = (not registered) and re.search(r"\b(calendar|reminder|appointment|front door|camera|schedule)\b", m.text, re.I)
        CLOSER_RE = re.compile(r"\b(that'?s all|thanks|thank you|goodbye|bye|never mind|nevermind|no thanks|nothing else)\b", re.I)
        needs_action = (not kind) and not guest_private and not CLOSER_RE.search(m.text) and not (not registered and re.search(r"\bremember\b", m.text, re.I)) and bool(ACTION_RE.search(m.text) or (AFFIRM_RE.match(m.text) and re.search(r"\b(would you like|shall i|should i|want me to|how about|can i)\b", prev_assistant, re.I)))

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
                elif not kind and (CLAIM_RE.search(content) or needs_action):
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
            if st["scheduled"] == 0 and st["first_tok"] is None and FILLERS and not kind and needs_action:
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

async def presence_watchdog():   # frames stop arriving (tab closed, camera off) => the visitor is gone
    while True:
        await asyncio.sleep(1.0)
        if VIS["present"] and time.time() - VIS["last_seen"] > AWAY_S: end_presence(time.time())

def reset_run(reason: str):
    end_presence(time.time(), reason)
    RUN.update(id=None, started=0.0, last_visitor=None); RUN["greeted"].clear()
    HISTORIES.clear(); SIM["identity"] = None

@app.post("/session/start")
async def session_start():
    """Start mirror: a fresh run session. Greetings, guests and conversations are remembered until /session/stop (registered users stay)."""
    reset_run("new run session"); RUN["id"] = uuid.uuid4().hex[:8]; RUN["started"] = round(time.time(), 3)
    emit("session.started", session=RUN["id"], registered_users=len(USERS)); return {"ok": True, "session": RUN["id"]}

@app.post("/session/stop")
async def session_stop():
    sid = RUN["id"]; reset_run("mirror stopped"); emit("session.stopped", session=sid); return {"ok": True}

@app.post("/reset")
async def reset():
    """Forget what was said. Who was greeted in this run session is kept, so the person in front is not greeted a second time."""
    HISTORIES.clear()
    emit("session.reset", session=RUN["id"]); return {"ok": True}

# ───────────────────────────── Vision endpoints ─────────────────────────────
class Frame(BaseModel):
    jpeg: str            # base64 JPEG from the browser camera
    simulate: str | None = None   # UI override: "unknown" (a new guest) | user/guest key | None (camera recognition)
    session: str | None = None    # the run session the sending tab started; frames from any other tab are ignored

def decode_jpeg(b64: str) -> np.ndarray | None:
    try: return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    except Exception: return None

def vision_state() -> dict:
    return {"users": {k: len(v["embeddings"]) for k, v in USERS.items()},
            "greeted": list(RUN["greeted"]), "session": RUN["id"], "build": BUILD}

@app.post("/vision/frame")
async def vision_frame(f: Frame):
    t = time.time(); loop = asyncio.get_running_loop()
    img = await loop.run_in_executor(None, decode_jpeg, f.jpeg)
    faces = await loop.run_in_executor(None, detect_faces, img)
    boxes = [{"box": d["box"], "score": d["score"]} for d in faces]
    if not RUN["id"] or f.session != RUN["id"]:
        # no run, or a tab whose run was replaced (another tab pressed Start): never greet from it; the tab stops itself on "stale"
        return {"present": False, "engaged": False, "identity": "unknown", "confidence": None, "visit": None, "greet": None, "others": 0,
                "display": "—", "registered": False, "stale": bool(f.session and f.session != RUN["id"]), "faces": boxes,
                "ms": round((time.time() - t) * 1000, 1), **vision_state()}
    SIM["identity"] = f.simulate or None
    pres = presence_update(faces, img)   # on the event loop: it emits events and mutates the shared visit state
    VIS["last_faces"] = [{"box": d["box"], "score": d["score"]} for d in faces]
    return {**pres, "faces": VIS["last_faces"], "ms": round((time.time() - t) * 1000, 1), **vision_state()}

@app.post("/vision/simulate")
async def vision_simulate(e: dict):
    """No camera: fake an approach ({"identity": "user1" | "unknown"}) or a departure ({"leave": true}) through the same visit logic."""
    now = time.time()
    if e.get("leave"):
        SIM["identity"] = None; end_presence(now, "left (simulated)")
        return {**presence_update([], np.zeros((2, 2, 3), np.uint8)), "faces": [], **vision_state()}
    if VIS["visit"]: end_visit(now, "simulated new approach")
    SIM["identity"] = e.get("identity") or "unknown"
    VIS.update(present=True, since=now - 1.0, decide_since=now - 1.0, last_seen=now + 120, reverify=False)   # stays "present" ~2 min without camera frames
    emit("presence.detected", faces=1, size=0.2, simulated=True)
    pres = presence_update([{"box": [0.4, 0.3, 0.2, 0.27], "score": 0.99, "raw": None}], np.zeros((2, 2, 3), np.uint8))
    VIS["last_seen"] = now + 120
    return {**pres, "faces": [], **vision_state()}

class Snapshot(BaseModel):
    frames: list[str]
    role: str = "user"   # only "user": the Snapshot → User button is the one way anyone gets registered

@app.post("/users/snapshot")
async def users_snapshot(s: Snapshot):
    """Snapshot → User: the only way a face gets registered. Embeddings only, runtime memory. A registered face is never registered twice;
    the guest currently in front snapshotted as a user becomes User N and keeps their greeting state and conversation."""
    def work():
        feats = []
        for b64 in s.frames[:12]:
            img = decode_jpeg(b64); faces = detect_faces(img)
            if faces and faces[0]["box"][2] >= 0.09:
                feats.append([round(float(v), 5) for v in embed(img, faces[0]["raw"]).flatten()])
        return feats, [recognize(np.array(f, dtype=np.float32).reshape(1, -1)) for f in feats]
    feats, matches = await asyncio.get_running_loop().run_in_executor(None, work)
    if len(feats) < 3:
        return JSONResponse({"ok": False, "error": f"need a clear face in at least 3 frames, got {len(feats)}"}, status_code=400)
    counts = collections.Counter(k for k, _, _ in matches if k != "unknown")
    known = counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1] >= max(2, len(feats) // 3) else "unknown"
    score = max((sc for k, sc, _ in matches if k == known), default=None)
    v = VIS["visit"]; single = len(VIS["last_faces"]) <= 1
    current_guest = bool(v and v["key"] == GUEST and single)
    if s.role != "user":
        return JSONResponse({"ok": False, "error": "only Snapshot → User registers a face; guests are never stored"}, status_code=400)
    if known in USERS:
        emit("user.already_registered", identity=known, confidence=score)
        return {"ok": True, "identity": known, "display": USERS[known]["display"], "already": True, "confidence": score}
    uid = register_user(feats, "snapshot button")
    if current_guest: rename_visitor(GUEST, uid); emit("guest.promoted", previous=GUEST, identity=uid)
    VIS["votes"].clear()
    return {"ok": True, "identity": uid, "display": USERS[uid]["display"], "samples": len(USERS[uid]["embeddings"]), "was": GUEST if current_guest else None}

@app.post("/users/simulate")
async def users_simulate():
    """No camera: register a placeholder User N with no face samples, so scenarios can run as a registered user."""
    uid = register_user([], "simulated (no camera)")
    return {"ok": True, "identity": uid, "display": USERS[uid]["display"]}

@app.get("/users")
async def users_list():
    return {"session": RUN["id"], "greeted": list(RUN["greeted"]), "last_visitor": RUN["last_visitor"],
            "current": VIS["visit"]["key"] if VIS["visit"] else None,
            "users": {k: {"display": v["display"], "samples": len(v["embeddings"]), "prefs": v["prefs"], "how": v["how"], "registered_at": v["registered_at"]} for k, v in USERS.items()}}

@app.post("/users/reset")
async def users_reset():
    n = len(USERS)
    if VIS["visit"] and VIS["visit"]["key"] in USERS: end_visit(time.time(), "users forgotten")
    for k in list(USERS): RUN["greeted"].pop(k, None); HISTORIES.pop(k, None)
    USERS.clear(); VIS["votes"].clear()
    emit("users.reset", forgotten=n); return {"ok": True, "forgotten": n}

@app.post("/users/forget")
async def users_forget(e: dict):
    uid = e.get("identity")
    if uid in USERS:
        if VIS["visit"] and VIS["visit"]["key"] == uid: end_visit(time.time(), "forgotten")
        USERS.pop(uid); RUN["greeted"].pop(uid, None); HISTORIES.pop(uid, None); VIS["votes"].clear()
        emit("user.forgotten", identity=uid)
    return {"ok": True}


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

@app.post("/scenarios/log")
async def scenarios_log(e: dict):
    """The browser runs scenario steps (it owns audio and the camera) and reports each verdict here so it lands in the event stream."""
    kind = re.sub(r"[^a-z_]", "", str(e.pop("kind", "step")).lower()) or "step"
    data = {k: v for k, v in e.items() if re.fullmatch(r"[a-z_]+", str(k)) and k not in ("id", "ts", "type", "type_")}
    return emit("scenario." + kind, **data)

@app.post("/client/log")
async def client_log(e: dict):
    """Rendering problems seen in the browser (GPU context lost/restored, frame errors), so they show up in the event stream and logs."""
    kind = re.sub(r"[^a-z_]", "", str(e.pop("kind", "note")).lower()) or "note"
    data = {k: str(v)[:300] for k, v in e.items() if re.fullmatch(r"[a-z_]+", str(k)) and k not in ("id", "ts", "type")}
    return emit("avatar." + kind, **data)

@app.on_event("startup")
async def warm():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, silero)
    for t in FILLER_TEXTS: FILLERS.append(await loop.run_in_executor(None, synth, t))
    LINE_CACHE[RETURN_TEXT] = await loop.run_in_executor(None, synth, RETURN_TEXT)   # the return line plays instantly
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
