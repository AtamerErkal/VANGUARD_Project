"""
VANGUARD AI — FastAPI Backend
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
from datetime import datetime, timezone, timedelta
from typing import Optional
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).parent.parent / "models"
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(title="Vanguard AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Model ─────────────────────────────────────────────────────────────────────

class ImprovedAircraftClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.network(torch.clamp(x, -10, 10))


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
    cat = ["electronic_signature", "flight_profile", "weather", "thermal_signature"]
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
    cls = _label_encoder.inverse_transform([pred.item()])[0]
    return {
        "classification": cls,
        "confidence":     float(conf.item()),
        "probabilities":  {c: float(p) for c, p in zip(_label_encoder.classes_, probs[0].numpy())},
    }

# ── XAI ──────────────────────────────────────────────────────────────────────

_NEUTRAL = {
    "electronic_signature": "UNKNOWN_EMISSION",
    "flight_profile":       "STABLE_CRUISE",
    "speed_kts":            400.0,
    "rcs_m2":               8.0,
    "thermal_signature":    "Not_Detected",
    "altitude_ft":          25000.0,
    "weather":              "Clear",
    "heading":              180.0,
}

_FEATURE_LABELS = {
    "electronic_signature": "Electronic Signature (IFF)",
    "flight_profile":       "Flight Profile",
    "speed_kts":            "Speed",
    "rcs_m2":               "Radar Cross-Section",
    "thermal_signature":    "Thermal Signature (IRST)",
    "altitude_ft":          "Altitude",
    "weather":              "Weather / IRST Fidelity",
    "heading":              "Heading",
}

_FEATURE_GROUPS = {
    "electronic_signature": "Electronic",
    "flight_profile":       "Kinematic",
    "speed_kts":            "Kinematic",
    "rcs_m2":               "Radar",
    "thermal_signature":    "Thermal",
    "altitude_ft":          "Kinematic",
    "weather":              "Environmental",
    "heading":              "Kinematic",
}


def compute_xai(track_input: dict, base_result: dict) -> list:
    base_cls   = base_result["classification"]
    base_p_cls = base_result["probabilities"].get(base_cls, base_result["confidence"])

    items = []
    for feature, neutral_val in _NEUTRAL.items():
        if feature not in track_input:
            continue
        if track_input[feature] == neutral_val:
            # Feature is already at neutral — still show with minimal importance
            items.append({
                "feature":   feature,
                "label":     _FEATURE_LABELS[feature],
                "group":     _FEATURE_GROUPS[feature],
                "value":     str(track_input[feature]),
                "importance": 0.01,
                "direction": "neutral",
                "delta":      0.0,
            })
            continue
        perturbed = {**track_input, feature: neutral_val}
        try:
            p = predict_aircraft(perturbed)
            p_conf = p["probabilities"].get(base_cls, 0.0)
        except Exception:
            continue
        delta     = base_p_cls - p_conf          # positive → feature supports this class
        direction = "supporting" if delta > 0.005 else ("conflicting" if delta < -0.005 else "neutral")
        items.append({
            "feature":    feature,
            "label":      _FEATURE_LABELS[feature],
            "group":      _FEATURE_GROUPS[feature],
            "value":      str(track_input[feature]),
            "importance": abs(delta),
            "direction":  direction,
            "delta":      round(delta, 4),
        })

    total = sum(i["importance"] for i in items) or 1
    for item in items:
        item["importance"] = round(item["importance"] / total, 3)

    return sorted(items, key=lambda x: -x["importance"])

# ── Sensor votes & fusion ──────────────────────────────────────────────────────

SENSOR_ORDER        = ["radar", "esm", "irst", "iff"]
SENSOR_BASE_WEIGHTS = {"radar": 0.40, "esm": 0.35, "irst": 0.15, "iff": 0.10}

WEATHER_IRST_FACTOR = {"Clear": 1.0, "Cloudy": 0.55, "Rainy": 0.30}


def track_sensor_votes(t: dict) -> dict:
    sig, thermal, weather = t["electronic_signature"], t["thermal_signature"], t["weather"]
    rcs, alt, spd = t["rcs_m2"], t["altitude_ft"], t["speed_kts"]

    # Radar: RCS + kinematics → class vote
    if rcs < 3 and spd > 500:     rv, rc = "HOSTILE",        0.73
    elif rcs > 25 and spd < 600:  rv, rc = "NEUTRAL",        0.78   # large slow → airliner/neutral
    elif 4 <= rcs <= 25:          rv, rc = "FRIEND",         0.66
    else:                         rv, rc = "UNKNOWN",        0.51

    # ESM: electronic signature → class vote
    esm_map = {
        "IFF_MODE_5":        ("FRIEND",         0.95),
        "IFF_MODE_3C":       ("ASSUMED FRIEND", 0.88),  # civil squawk — could be allied
        "HOSTILE_JAMMING":   ("HOSTILE",        0.90),
        "NO_IFF_RESPONSE":   ("SUSPECT",        0.76),  # no reply → suspect, not confirmed hostile
        "UNKNOWN_EMISSION":  ("UNKNOWN",        0.52),
    }
    ev, ec = esm_map.get(sig, ("UNKNOWN", 0.40))

    # IRST: thermal signature → class vote (weather-degraded)
    th_map = {
        "High":         ("HOSTILE",  0.78),
        "Medium":       ("SUSPECT",  0.55),
        "Low":          ("NEUTRAL",  0.65),
        "Not_Detected": ("UNKNOWN",  0.40),
    }
    iv, ic_base = th_map.get(thermal, ("UNKNOWN", 0.40))
    ic = round(ic_base * WEATHER_IRST_FACTOR.get(weather, 1.0), 2)

    # IFF system: transponder response → class vote
    iff_map = {
        "IFF_MODE_5":        ("FRIEND",         0.98),
        "IFF_MODE_3C":       ("ASSUMED FRIEND", 0.93),
        "HOSTILE_JAMMING":   ("HOSTILE",        0.88),
        "NO_IFF_RESPONSE":   ("SUSPECT",        0.84),
        "UNKNOWN_EMISSION":  ("UNKNOWN",        0.55),
    }
    fv, fc = iff_map.get(sig, ("UNKNOWN", 0.40))

    weather_note = f" · degraded ×{WEATHER_IRST_FACTOR.get(weather, 1)}" if weather != "Clear" else ""

    return {
        "radar": {"label": "Active Radar",  "icon": "📡", "vote": rv, "conf": rc,
                  "reading": f"Alt {alt:,.0f} ft · {spd:.0f} kts · RCS {rcs:.1f} m²"},
        "esm":   {"label": "ESM Suite",     "icon": "📻", "vote": ev, "conf": ec, "reading": sig},
        "irst":  {"label": "IRST Camera",   "icon": "🔥", "vote": iv, "conf": ic,
                  "reading": f"Thermal: {thermal}{weather_note}",
                  "weather_factor": WEATHER_IRST_FACTOR.get(weather, 1.0),
                  "base_conf": ic_base},
        "iff":   {"label": "IFF System",    "icon": "🆔", "vote": fv, "conf": fc,
                  "reading": sig + (" · L2 encrypted" if sig == "IFF_MODE_5" else "")},
    }


def compute_fusion(sensor_votes: dict, weights: dict) -> dict:
    classes = ["HOSTILE", "SUSPECT", "FRIEND", "ASSUMED FRIEND", "NEUTRAL", "UNKNOWN"]
    probs   = {c: 0.0 for c in classes}
    total_w = sum(weights.get(s, 0) for s in sensor_votes) or 1
    norm_w  = {s: weights.get(s, 0) / total_w for s in sensor_votes}
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
                    "desc": f"{sig} active but AGGRESSIVE_MANEUVERS — possible IFF spoofing"})
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


# ── Track generation ──────────────────────────────────────────────────────────

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

_N_HIST    = 9
_STEP_MINS = 5


def _build_tracks():
    np.random.seed(77)
    now    = datetime.now(timezone.utc)
    tracks = []

    for i, (lat, lon, alt, spd, rcs, hdg, esig, fp, thermal, weather) in enumerate(_CONFIGS):
        lat += np.random.normal(0, 0.05)
        lon += np.random.normal(0, 0.05)

        ang   = np.radians(hdg)
        hlats = [lat - (_N_HIST - j) * 0.09 * np.cos(ang) + np.random.normal(0, 0.015)
                 for j in range(_N_HIST)] + [lat]
        hlons = [lon - (_N_HIST - j) * 0.09 * np.sin(ang) + np.random.normal(0, 0.015)
                 for j in range(_N_HIST)] + [lon]
        halts = [max(200, alt + (j - _N_HIST) * np.random.uniform(80, 220))
                 for j in range(_N_HIST)] + [alt]

        # Speed history — varies more for aggressive profiles
        sp_var = {"AGGRESSIVE_MANEUVERS": 55, "LOW_ALTITUDE_FLYING": 35, "CLIMBING": 20}.get(fp, 12)
        hspds = []
        s = spd + np.random.uniform(-sp_var * 2, sp_var * 2)
        for j in range(_N_HIST):
            s += np.random.normal(0, sp_var * 0.6)
            hspds.append(int(np.clip(s, 100, 999)))
        hspds.append(int(spd))

        # Heading history — aggressive profiles make bigger turns per step
        hdg_var = {"AGGRESSIVE_MANEUVERS": 14, "LOW_ALTITUDE_FLYING": 9, "CLIMBING": 6}.get(fp, 3)
        hhdgs = []
        h = (hdg + np.random.uniform(-hdg_var * _N_HIST, hdg_var * _N_HIST)) % 360
        for j in range(_N_HIST):
            h = (h + np.random.normal(0, hdg_var)) % 360
            hhdgs.append(round(h, 1))
        hhdgs.append(round(hdg, 1))

        # Timestamps: one per history point, _STEP_MINS apart, ending at now
        tstamps = [
            (now - timedelta(minutes=(_N_HIST - j) * _STEP_MINS)).strftime("%H:%M UTC")
            for j in range(_N_HIST)
        ] + [now.strftime("%H:%M UTC")]

        inp = dict(altitude_ft=alt, speed_kts=spd, rcs_m2=rcs, latitude=lat, longitude=lon,
                   heading=hdg, electronic_signature=esig, flight_profile=fp,
                   weather=weather, thermal_signature=thermal)

        try:
            res = predict_aircraft(inp)
        except Exception:
            res = {"classification": "NEUTRAL", "confidence": 0.5, "probabilities": {}}

        sv     = track_sensor_votes(inp)
        fusion = compute_fusion(sv, SENSOR_BASE_WEIGHTS)
        xai    = compute_xai(inp, res)

        # Weather comparison for IRST
        irst_base = sv["irst"]["base_conf"]
        weather_impact = {
            w: round(irst_base * f, 2)
            for w, f in WEATHER_IRST_FACTOR.items()
        }

        tracks.append({
            "track_id": f"TRK-{i+1:03d}",
            **inp,
            "ai_class":       res["classification"],
            "ai_conf":        res["confidence"],
            "ai_probs":       res["probabilities"],
            "sensor_votes":   sv,
            "fusion":         fusion,
            "anomalies":      detect_anomalies(inp),
            "xai":            xai,
            "weather_impact": weather_impact,
            "hist_lats":       hlats,
            "hist_lons":       hlons,
            "hist_alts":       halts,
            "hist_speeds":     hspds,
            "hist_headings":   hhdgs,
            "hist_timestamps": tstamps,
        })
    return tracks


_tracks_cache: list | None = None

def get_tracks():
    global _tracks_cache
    if _tracks_cache is None:
        _tracks_cache = _build_tracks()
    return _tracks_cache


# ── Model statistics (computed once, cached) ──────────────────────────────────

_DATA_PATH        = Path(__file__).parent.parent / "data" / "vanguard_air_tracks_fused.csv"
_model_stats_cache = None

def compute_model_stats() -> dict:
    global _model_stats_cache
    if _model_stats_cache is not None:
        return _model_stats_cache

    df  = pd.read_csv(_DATA_PATH)
    cat = ["electronic_signature", "flight_profile", "weather", "thermal_signature"]
    X_raw = pd.get_dummies(df.drop(columns=["classification"]), columns=cat)
    for col in _feature_cols:
        if col not in X_raw.columns:
            X_raw[col] = 0
    X_raw = X_raw[_feature_cols]
    y = df["classification"].values

    _, X_test, _, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    X_scaled = _scaler.transform(X_test.values.astype(float))

    with torch.no_grad():
        out   = _model(torch.FloatTensor(X_scaled))
        preds = torch.argmax(torch.softmax(out, dim=1), dim=1).numpy()

    y_pred  = _label_encoder.inverse_transform(preds)
    classes = list(_label_encoder.classes_)

    f1_per  = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    pre_per = precision_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    rec_per = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred, labels=classes)

    _model_stats_cache = {
        "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro":         round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted":      round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "test_size":        int(len(y_test)),
        "train_size":       int(len(df) - len(y_test)),
        "classes":          classes,
        "per_class":        {
            cls: {
                "f1":        round(float(f1_per[i]),  4),
                "precision": round(float(pre_per[i]), 4),
                "recall":    round(float(rec_per[i]), 4),
                "support":   int(np.sum(y_test == cls)),
            }
            for i, cls in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    return _model_stats_cache


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
    altitude_ft:          float
    speed_kts:            float
    rcs_m2:               float
    latitude:             float
    longitude:            float
    heading:              float
    electronic_signature: str
    flight_profile:       str
    weather:              str
    thermal_signature:    str


@app.get("/api/model-stats")
def model_stats_endpoint():
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    return compute_model_stats()


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    inp    = req.model_dump()
    result = predict_aircraft(inp)
    xai    = compute_xai(inp, result)
    sv     = track_sensor_votes(inp)
    fusion = compute_fusion(sv, SENSOR_BASE_WEIGHTS)
    irst_base = sv["irst"]["base_conf"]
    return {
        **result,
        "xai":            xai,
        "sensor_votes":   sv,
        "fusion":         fusion,
        "weather_impact": {w: round(irst_base * f, 2) for w, f in WEATHER_IRST_FACTOR.items()},
    }
