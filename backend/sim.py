"""
VANGUARD SIM — Live multi-user simulation engine
Manages sessions, WebSocket connections and synthetic map positions.
"""
import uuid, math, random
from datetime import datetime, timezone
from typing import Dict, List
from fastapi import WebSocket


# ── Session / WebSocket manager ───────────────────────────────────────────────

class SimSessionManager:
    def __init__(self):
        self._sessions: Dict[str, dict]           = {}
        self._sockets:  Dict[str, List[WebSocket]] = {}

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        for _ in range(20):
            sid = str(uuid.uuid4())[:6].upper()
            if sid not in self._sessions:
                break
        self._sessions[sid] = {
            "tracks":    [],
            "created":   datetime.now(timezone.utc).isoformat(),
        }
        self._sockets[sid] = []
        return sid

    def session_exists(self, sid: str) -> bool:
        return sid in self._sessions

    def get_tracks(self, sid: str) -> list:
        return self._sessions.get(sid, {}).get("tracks", [])

    def add_track(self, sid: str, track: dict):
        if sid in self._sessions:
            self._sessions[sid]["tracks"].append(track)

    # ── WebSocket ─────────────────────────────────────────────────────────────

    async def connect(self, sid: str, ws: WebSocket):
        await ws.accept()
        self._sockets.setdefault(sid, []).append(ws)

    def disconnect(self, sid: str, ws: WebSocket):
        lst = self._sockets.get(sid, [])
        if ws in lst:
            lst.remove(ws)

    def participant_count(self, sid: str) -> int:
        return len(self._sockets.get(sid, []))

    async def broadcast(self, sid: str, msg: dict):
        dead = []
        for ws in list(self._sockets.get(sid, [])):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(sid, ws)


sim_mgr = SimSessionManager()


# ── Synthetic map position assignment ────────────────────────────────────────
#
#  Coordinate space: x ∈ [-80, 80],  y ∈ [-45, 45]
#  Ship at origin (0, 0)
#  Enemy zone  : x < -20  (left)
#  Friendly zone: x >  20  (right)
#  Neutral      : -20 ≤ x ≤ 20
#
#  SVG mapping (1000 × 600 viewBox, ship at 500,300):
#    svg_x = 500 + x * 5
#    svg_y = 300 + y * 5

def assign_position(ai_class: str) -> dict:
    if ai_class in ("HOSTILE", "SUSPECT"):
        angle  = random.uniform(130, 230)       # left arc
        radius = random.uniform(52, 72)
    elif ai_class in ("FRIEND", "ASSUMED FRIEND"):
        angle  = random.uniform(-50, 50)        # right arc
        radius = random.uniform(52, 72)
    else:                                       # UNKNOWN / NEUTRAL
        angle  = random.choice([
            random.uniform(40, 90),             # top
            random.uniform(270, 320),           # bottom
        ])
        radius = random.uniform(48, 68)

    x = round(radius * math.cos(math.radians(angle)), 1)
    y = round(radius * math.sin(math.radians(angle)), 1)
    return {"x": x, "y": y}
