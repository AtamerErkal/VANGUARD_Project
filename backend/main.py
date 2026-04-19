"""
VANGUARD AI — FastAPI Backend
Serves the PyTorch classification model and pre-computed track data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

MODELS_DIR = Path(__file__).parent.parent / "models"
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(title="Vanguard AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model definition ──────────────────────────────────────────────────────────

class ImprovedAircraftClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(torch.clamp(x, -10, 10))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    scaler        = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_cols  = joblib.load(MODELS_DIR / "feature_columns.joblib")

    model = ImprovedAircraftClassifier(len(feature_cols), len(label_encoder.classes_))
    ckpt  = torch.load(MODELS_DIR / "best_model.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model, scaler, label_encoder, feature_cols

try:
    _model, _scaler, _label_encoder, _feature_cols = load_model()
    _model_ready = True
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    _model_ready = False


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_aircraft(data: dict) -> dict:
    df = pd.DataFrame([data])
    cat = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
    df  = pd.get_dummies(df, columns=cat)
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[_feature_cols]
    X  = _scaler.transform(df.values.astype(float))
    with torch.no_grad():
        out   = _model(torch.FloatTensor(X))
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    cls   = _label_encoder.inverse_transform([pred.item()])[0]
    return {
        "classification": cls,
        "confidence": float(conf.item()),
        "probabilities": {c: float(p) for c, p in zip(_label_encoder.classes_, probs[0].numpy())},
    }


# ── Sensor votes & anomalies ──────────────────────────────────────────────────

SENSOR_ORDER        = ["radar", "esm", "irst", "iff"]
SENSOR_BASE_WEIGHTS = {"radar": 0.40, "esm": 0.35, "irst": 0.15, "iff": 0.10}


def track_sensor_votes(t: dict) -> dict:
    sig, thermal, weather = t["electronic_signature"], t["thermal_signature"], t["weather"]
    rcs, alt, spd = t["rcs_m2"], t["altitude_ft"], t["speed_kts"]

    if rcs < 3 and spd > 500:     rv, rc = "HOSTILE",  0.73
    elif rcs > 10 and spd < 500:  rv, rc = "CIVILIAN", 0.81
    elif 4 <= rcs <= 10:          rv, rc = "FRIEND",   0.66
    else:                         rv, rc = "SUSPECT",  0.51

    esm_map = {"IFF_MODE_5": ("FRIEND", 0.95), "IFF_MODE_3C": ("CIVILIAN", 0.92),
               "HOSTILE_JAMMING": ("HOSTILE", 0.90), "NO_IFF_RESPONSE": ("HOSTILE", 0.76),
               "UNKNOWN_EMISSION": ("SUSPECT", 0.45)}
    ev, ec = esm_map.get(sig, ("NEUTRAL", 0.40))

    th_map = {"High": ("HOSTILE", 0.78), "Medium": ("SUSPECT", 0.55),
              "Low": ("CIVILIAN", 0.70), "Not_Detected": ("NEUTRAL", 0.40)}
    iv, ic = th_map.get(thermal, ("NEUTRAL", 0.40))
    if weather != "Clear":
        ic = round(ic * 0.55, 2)

    iff_map = {"IFF_MODE_5": ("FRIEND", 0.98), "IFF_MODE_3C": ("CIVILIAN", 0.96),
               "HOSTILE_JAMMING": ("HOSTILE", 0.88), "NO_IFF_RESPONSE": ("HOSTILE", 0.92),
               "UNKNOWN_EMISSION": ("NEUTRAL", 0.50)}
    fv, fc = iff_map.get(sig, ("NEUTRAL", 0.40))

    return {
        "radar": {"label": "Active Radar", "icon": "📡", "vote": rv, "conf": rc,
                  "reading": f"Alt {alt:,.0f} ft · {spd:.0f} kts · RCS {rcs:.1f} m²"},
        "esm":   {"label": "ESM Suite",    "icon": "📻", "vote": ev, "conf": ec, "reading": sig},
        "irst":  {"label": "IRST Camera",  "icon": "🔥", "vote": iv, "conf": ic,
                  "reading": f"Thermal: {thermal}" + (" · degraded" if weather != "Clear" else "")},
        "iff":   {"label": "IFF System",   "icon": "🆔", "vote": fv, "conf": fc,
                  "reading": sig + (" · L2 encrypted" if sig == "IFF_MODE_5" else "")},
    }


def compute_fusion(sensor_votes: dict, weights: dict) -> dict:
    classes = ["HOSTILE", "SUSPECT", "FRIEND", "ASSUMED FRIEND", "NEUTRAL", "CIVILIAN"]
    probs   = {c: 0.0 for c in classes}
    total_w = sum(weights[s] for s in sensor_votes) or 1
    norm_w  = {s: weights[s] / total_w for s in sensor_votes}
    for s, vd in sensor_votes.items():
        voted, conf, w = vd["vote"], vd["conf"], norm_w[s]
        if voted in probs:
            probs[voted] += w * conf
        spread = w * (1 - conf) / max(len(classes) - 1, 1)
        for c in classes:
            if c != voted:
                probs[c] += spread
    total_p = sum(probs.values()) or 1
    probs   = {c: round(v / total_p, 4) for c, v in probs.items()}
    return {"best": max(probs, key=probs.get), "probs": probs, "weights": norm_w}


def detect_anomalies(t: dict) -> list:
    sig, profile = t["electronic_signature"], t["flight_profile"]
    rcs, alt, spd, thermal = t["rcs_m2"], t["altitude_ft"], t["speed_kts"], t["thermal_signature"]
    out = []
    if sig in ("IFF_MODE_3C", "IFF_MODE_5") and profile == "AGGRESSIVE_MANEUVERS":
        out.append({"title": "IFF–Maneuver Conflict",
                    "desc": f"{sig} active but profile is AGGRESSIVE_MANEUVERS — possible IFF spoofing"})
    if sig == "IFF_MODE_3C" and rcs < 5.0:
        out.append({"title": "RCS–IFF Mismatch",
                    "desc": f"Civilian IFF but RCS = {rcs:.1f} m² — too small for commercial airframe"})
    if sig == "IFF_MODE_3C" and alt < 10000 and spd > 500:
        out.append({"title": "Kinematic Anomaly",
                    "desc": f"Civil squawk + alt {alt:,.0f} ft + {spd:.0f} kts — inconsistent with civil profile"})
    if profile == "LOW_ALTITUDE_FLYING" and alt < 1000:
        out.append({"title": "Terrain Hugging",
                    "desc": f"Alt {alt:,.0f} ft — possible NOE / terrain-masking flight"})
    if sig == "NO_IFF_RESPONSE" and spd > 600:
        out.append({"title": "High-Speed Non-Cooperative",
                    "desc": f"No IFF + {spd:.0f} kts — intercept criteria met"})
    if sig == "HOSTILE_JAMMING" and thermal == "Low":
        out.append({"title": "Signature Conflict",
                    "desc": "Jamming detected but low thermal — stealth platform or sensor malfunction"})
    return out


# ── Track generation (cached at startup) ─────────────────────────────────────

_CONFIGS = [
    (45.2, 35.8, 7500,  680, 1.1, 270, "HOSTILE_JAMMING",    "AGGRESSIVE_MANEUVERS", "High",   "Clear"),
    (44.8, 34.2, 1200,  720, 0.6, 315, "NO_IFF_RESPONSE",    "LOW_ALTITUDE_FLYING",  "High",   "Clear"),
    (51.5,  0.1, 35000, 450, 18,   90, "IFF_MODE_3C",        "STABLE_CRUISE",        "Low",    "Clear"),
    (50.8,  2.3, 33000, 440, 15,   85, "IFF_MODE_3C",        "STABLE_CRUISE",        "Low",    "Cloudy"),
    (51.2, 12.4, 25000, 410,  6,   45, "IFF_MODE_5",         "STABLE_CRUISE",        "Medium", "Clear"),
    (52.1, 14.8, 18000, 390,  5.5, 60, "IFF_MODE_5",         "CLIMBING",             "Medium", "Clear"),
    (47.1, 22.3, 14000, 430,  3.2,195, "UNKNOWN_EMISSION",   "CLIMBING",             "Medium", "Cloudy"),
    (46.5, 20.1,  9000, 510,  2.8,220, "NO_IFF_RESPONSE",    "AGGRESSIVE_MANEUVERS", "Medium", "Clear"),
    (49.8,  8.5, 29000, 400, 12,  110, "IFF_MODE_3C",        "STABLE_CRUISE",        "Low",    "Clear"),
    (48.3, 26.7,  5500, 610,  1.4,260, "IFF_MODE_3C",        "AGGRESSIVE_MANEUVERS", "High",   "Clear"),
]


def _build_tracks():
    np.random.seed(77)
    tracks = []
    for i, (lat, lon, alt, spd, rcs, hdg, esig, fp, thermal, weather) in enumerate(_CONFIGS):
        lat += np.random.normal(0, 0.05)
        lon += np.random.normal(0, 0.05)
        n, ang = 9, np.radians(hdg)
        hlats = [lat - (n - j) * 0.09 * np.cos(ang) + np.random.normal(0, 0.015) for j in range(n)] + [lat]
        hlons = [lon - (n - j) * 0.09 * np.sin(ang) + np.random.normal(0, 0.015) for j in range(n)] + [lon]
        halts = [max(200, alt + (j - n) * np.random.uniform(80, 220)) for j in range(n)] + [alt]

        inp = dict(altitude_ft=alt, speed_kts=spd, rcs_m2=rcs, latitude=lat, longitude=lon,
                   heading=hdg, electronic_signature=esig, flight_profile=fp,
                   weather=weather, thermal_signature=thermal)
        try:
            res = predict_aircraft(inp)
        except Exception:
            res = {"classification": "NEUTRAL", "confidence": 0.5, "probabilities": {}}

        sv     = track_sensor_votes(inp)
        fusion = compute_fusion(sv, SENSOR_BASE_WEIGHTS)

        tracks.append({
            "track_id": f"TRK-{i+1:03d}",
            **inp,
            "ai_class":    res["classification"],
            "ai_conf":     res["confidence"],
            "ai_probs":    res["probabilities"],
            "sensor_votes": sv,
            "fusion":      fusion,
            "anomalies":   detect_anomalies(inp),
            "hist_lats":   hlats,
            "hist_lons":   hlons,
            "hist_alts":   halts,
        })
    return tracks


_tracks_cache: list | None = None

def get_tracks():
    global _tracks_cache
    if _tracks_cache is None:
        _tracks_cache = _build_tracks()
    return _tracks_cache


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "model_ready": _model_ready}


@app.get("/api/tracks")
def tracks_endpoint():
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    return get_tracks()


class PredictRequest(BaseModel):
    altitude_ft: float
    speed_kts: float
    rcs_m2: float
    latitude: float
    longitude: float
    heading: float
    electronic_signature: str
    flight_profile: str
    weather: str
    thermal_signature: str


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    return predict_aircraft(req.model_dump())
