"""
VANGUARD AI — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
from backend.kalman import ConstantVelocityKalman, compute_track_quality, dempster_shafer_fusion

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
    cat = ["esm_signature", "iff_mode", "flight_profile", "weather", "thermal_signature"]
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

# ── MC Dropout uncertainty estimation ────────────────────────────────────────

_MC_SAMPLES = 20

def predict_with_uncertainty(data: dict) -> dict:
    """
    Monte Carlo Dropout inference: run _MC_SAMPLES stochastic forward passes
    with dropout active to estimate epistemic (model) uncertainty.

    Returns the same keys as predict_aircraft() plus:
        epistemic_uncertainty — mean std across classes (0 = certain)
        uncertainty_label     — HIGH / MEDIUM / LOW
    """
    df  = pd.DataFrame([data])
    cat = ["esm_signature", "iff_mode", "flight_profile", "weather", "thermal_signature"]
    df  = pd.get_dummies(df, columns=cat)
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[_feature_cols]
    X  = _scaler.transform(df.values.astype(float))
    x_t = torch.FloatTensor(X)

    # Enable only Dropout layers — BatchNorm stays in eval mode to avoid
    # the batch-size-1 constraint while still sampling stochastic passes.
    _model.eval()
    for m in _model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    samples = []
    with torch.no_grad():
        for _ in range(_MC_SAMPLES):
            out   = _model(x_t)
            probs = torch.softmax(out, dim=1).numpy()[0]
            samples.append(probs)
    _model.eval()    # restore all layers to deterministic mode

    samples    = np.array(samples)          # (_MC_SAMPLES, n_classes)
    mean_probs = samples.mean(axis=0)
    std_probs  = samples.std(axis=0)

    pred_idx             = int(np.argmax(mean_probs))
    epistemic_uncertainty = float(std_probs.mean())

    if epistemic_uncertainty < 0.04:
        uncertainty_label = "LOW"
    elif epistemic_uncertainty < 0.10:
        uncertainty_label = "MEDIUM"
    else:
        uncertainty_label = "HIGH"

    return {
        "classification":       _label_encoder.inverse_transform([pred_idx])[0],
        "confidence":           float(mean_probs[pred_idx]),
        "probabilities":        {c: float(p) for c, p in zip(_label_encoder.classes_, mean_probs)},
        "epistemic_uncertainty": round(epistemic_uncertainty, 4),
        "uncertainty_label":    uncertainty_label,
    }


# ── XAI ──────────────────────────────────────────────────────────────────────

_NEUTRAL = {
    "esm_signature":     "UNKNOWN_EMISSION",
    "iff_mode":          "NO_RESPONSE",
    "flight_profile":    "STABLE_CRUISE",
    "speed_kts":         400.0,
    "rcs_m2":            8.0,
    "thermal_signature": "Not_Detected",
    "altitude_ft":       25000.0,
    "weather":           "Clear",
    "heading":           180.0,
}

_FEATURE_LABELS = {
    "esm_signature":     "ESM — Emission Signature",
    "iff_mode":          "IFF — Transponder Mode",
    "flight_profile":    "Flight Profile",
    "speed_kts":         "Speed",
    "rcs_m2":            "Radar Cross-Section",
    "thermal_signature": "Thermal Signature (IRST)",
    "altitude_ft":       "Altitude",
    "weather":           "Weather / IRST Fidelity",
    "heading":           "Heading",
}

_FEATURE_GROUPS = {
    "esm_signature":     "Electronic",
    "iff_mode":          "Electronic",
    "flight_profile":    "Kinematic",
    "speed_kts":         "Kinematic",
    "rcs_m2":            "Radar",
    "thermal_signature": "Thermal",
    "altitude_ft":       "Kinematic",
    "weather":           "Environmental",
    "heading":           "Kinematic",
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
    esm     = t.get("esm_signature", "UNKNOWN_EMISSION")
    iff     = t.get("iff_mode",      "NO_RESPONSE")
    thermal = t["thermal_signature"]
    weather = t["weather"]
    rcs, alt, spd = t["rcs_m2"], t["altitude_ft"], t["speed_kts"]

    # ── Radar: RCS + kinematics ───────────────────────────────────────────────
    if rcs < 3 and spd > 500:    rv, rc = "HOSTILE", 0.73
    elif rcs > 25 and spd < 600: rv, rc = "NEUTRAL", 0.78
    elif 4 <= rcs <= 25:         rv, rc = "FRIEND",  0.66
    else:                        rv, rc = "UNKNOWN", 0.51

    # ── ESM: what is the aircraft emitting / jamming? ─────────────────────────
    esm_map = {
        "CLEAN":            ("NEUTRAL",        0.72),   # quiet — civil or friendly
        "UNKNOWN_EMISSION": ("UNKNOWN",        0.52),   # some emission, unidentified
        "NOISE_JAMMING":    ("SUSPECT",        0.78),   # broadband jamming → suspicious
        "HOSTILE_JAMMING":  ("HOSTILE",        0.90),   # targeted EA → hostile
    }
    ev, ec = esm_map.get(esm, ("UNKNOWN", 0.40))

    # ── IRST: thermal signature (weather-degraded) ────────────────────────────
    th_map = {
        "High":         ("HOSTILE", 0.78),
        "Medium":       ("SUSPECT", 0.55),
        "Low":          ("NEUTRAL", 0.65),
        "Not_Detected": ("UNKNOWN", 0.40),
    }
    iv, ic_base = th_map.get(thermal, ("UNKNOWN", 0.40))
    ic = round(ic_base * WEATHER_IRST_FACTOR.get(weather, 1.0), 2)

    # ── IFF: what does the transponder say? ───────────────────────────────────
    iff_map = {
        "IFF_MODE_5":  ("FRIEND",         0.98),   # military crypto — confirmed friend
        "IFF_MODE_3C": ("ASSUMED FRIEND", 0.88),   # civil squawk — probably neutral/allied
        "DEGRADED":    ("UNKNOWN",        0.55),   # intermittent — could be equipment fault
        "NO_RESPONSE": ("SUSPECT",        0.84),   # no reply → suspect
    }
    fv, fc = iff_map.get(iff, ("UNKNOWN", 0.40))

    weather_note = f" · degraded ×{WEATHER_IRST_FACTOR.get(weather, 1)}" if weather != "Clear" else ""

    return {
        "radar": {"label": "Active Radar",  "icon": "📡", "vote": rv, "conf": rc,
                  "reading": f"Alt {alt:,.0f} ft · {spd:.0f} kts · RCS {rcs:.1f} m²"},
        "esm":   {"label": "ESM Suite",     "icon": "📻", "vote": ev, "conf": ec,
                  "reading": esm},
        "irst":  {"label": "IRST Camera",   "icon": "🔥", "vote": iv, "conf": ic,
                  "reading": f"Thermal: {thermal}{weather_note}",
                  "weather_factor": WEATHER_IRST_FACTOR.get(weather, 1.0),
                  "base_conf": ic_base},
        "iff":   {"label": "IFF System",    "icon": "🆔", "vote": fv, "conf": fc,
                  "reading": iff + (" · L2 encrypted" if iff == "IFF_MODE_5" else "")},
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
    esm     = t.get("esm_signature", "UNKNOWN_EMISSION")
    iff     = t.get("iff_mode",      "NO_RESPONSE")
    profile = t["flight_profile"]
    rcs, alt, spd, thermal = t["rcs_m2"], t["altitude_ft"], t["speed_kts"], t["thermal_signature"]
    out = []
    if iff in ("IFF_MODE_3C", "IFF_MODE_5") and profile == "AGGRESSIVE_MANEUVERS":
        out.append({"title": "IFF–Maneuver Conflict",
                    "desc": f"{iff} active but AGGRESSIVE_MANEUVERS — possible IFF spoofing"})
    if iff == "IFF_MODE_3C" and rcs < 5.0:
        out.append({"title": "RCS–IFF Mismatch",
                    "desc": f"Civilian squawk but RCS = {rcs:.1f} m² — too small for commercial airframe"})
    if iff == "IFF_MODE_3C" and alt < 10000 and spd > 500:
        out.append({"title": "Kinematic Anomaly",
                    "desc": f"Civil squawk + alt {alt:,.0f} ft + {spd:.0f} kts — inconsistent with civil profile"})
    if esm in ("HOSTILE_JAMMING", "NOISE_JAMMING") and iff in ("IFF_MODE_3C", "IFF_MODE_5"):
        out.append({"title": "ESM–IFF Conflict",
                    "desc": f"Active jamming ({esm}) while squawking {iff} — deception scenario"})
    if profile == "LOW_ALTITUDE_FLYING" and alt < 1000:
        out.append({"title": "Terrain Hugging",
                    "desc": f"Alt {alt:,.0f} ft — possible NOE / terrain-masking flight"})
    if iff == "NO_RESPONSE" and spd > 600:
        out.append({"title": "High-Speed Non-Cooperative",
                    "desc": f"No IFF response + {spd:.0f} kts — intercept criteria met"})
    if esm == "HOSTILE_JAMMING" and thermal == "Low":
        out.append({"title": "Signature Conflict",
                    "desc": "Active jamming but low thermal — possible stealth platform"})
    return out


# ── Track generation ──────────────────────────────────────────────────────────

# (lat, lon, alt, spd, rcs, hdg, esm_signature, iff_mode, flight_profile, thermal, weather)
_CONFIGS = [
    (45.2, 35.8,  7500, 680, 1.1, 270, "HOSTILE_JAMMING",   "NO_RESPONSE",  "AGGRESSIVE_MANEUVERS", "High",   "Clear"),
    (44.8, 34.2,  1200, 720, 0.6, 315, "UNKNOWN_EMISSION",  "NO_RESPONSE",  "LOW_ALTITUDE_FLYING",  "High",   "Clear"),
    (51.5,  0.1, 35000, 450,  18,  90, "CLEAN",             "IFF_MODE_3C",  "STABLE_CRUISE",        "Low",    "Clear"),
    (50.8,  2.3, 33000, 440,  15,  85, "CLEAN",             "IFF_MODE_3C",  "STABLE_CRUISE",        "Low",    "Cloudy"),
    (51.2, 12.4, 25000, 410,   6,  45, "CLEAN",             "IFF_MODE_5",   "STABLE_CRUISE",        "Medium", "Clear"),
    (52.1, 14.8, 18000, 390, 5.5,  60, "CLEAN",             "IFF_MODE_5",   "CLIMBING",             "Medium", "Clear"),
    (47.1, 22.3, 14000, 430, 3.2, 195, "UNKNOWN_EMISSION",  "NO_RESPONSE",  "CLIMBING",             "Medium", "Cloudy"),
    (46.5, 20.1,  9000, 510, 2.8, 220, "NOISE_JAMMING",     "NO_RESPONSE",  "AGGRESSIVE_MANEUVERS", "Medium", "Clear"),
    (49.8,  8.5, 29000, 400,  12, 110, "CLEAN",             "IFF_MODE_3C",  "STABLE_CRUISE",        "Low",    "Clear"),
    (48.3, 26.7,  5500, 610, 1.4, 260, "NOISE_JAMMING",     "IFF_MODE_3C",  "AGGRESSIVE_MANEUVERS", "High",   "Clear"),
]

_N_HIST    = 9
_STEP_MINS = 5
_KF        = ConstantVelocityKalman()


def _build_tracks():
    np.random.seed(77)
    now    = datetime.now(timezone.utc)
    tracks = []

    for i, (lat, lon, alt, spd, rcs, hdg, esm, iff, fp, thermal, weather) in enumerate(_CONFIGS):
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
                   heading=hdg, esm_signature=esm, iff_mode=iff, flight_profile=fp,
                   weather=weather, thermal_signature=thermal)

        try:
            res = predict_with_uncertainty(inp)
        except Exception:
            res = {
                "classification": "NEUTRAL", "confidence": 0.5, "probabilities": {},
                "epistemic_uncertainty": 0.0, "uncertainty_label": "LOW",
            }

        sv       = track_sensor_votes(inp)
        fusion   = compute_fusion(sv, SENSOR_BASE_WEIGHTS)
        ds_fusion = dempster_shafer_fusion(sv)
        xai      = compute_xai(inp, res)

        # ── Kalman track quality (Track-Level Fusion) ─────────────────────────
        meas = np.column_stack([
            hlats,
            hlons,
            [a / 1000.0 for a in halts],   # feet → kilo-feet for numeric stability
        ])
        kf_state    = _KF.filter(meas)
        track_qual  = compute_track_quality(kf_state)

        # Weather comparison for IRST
        irst_base = sv["irst"]["base_conf"]
        weather_impact = {
            w: round(irst_base * f, 2)
            for w, f in WEATHER_IRST_FACTOR.items()
        }

        tracks.append({
            "track_id": f"TRK-{i+1:03d}",
            **inp,
            "ai_class":              res["classification"],
            "ai_conf":               res["confidence"],
            "ai_probs":              res["probabilities"],
            "epistemic_uncertainty": res["epistemic_uncertainty"],
            "uncertainty_label":     res["uncertainty_label"],
            "sensor_votes":          sv,
            "fusion":                fusion,
            "ds_fusion":             ds_fusion,
            "track_quality":         track_qual,
            "anomalies":             detect_anomalies(inp),
            "xai":                   xai,
            "weather_impact":        weather_impact,
            "hist_lats":             hlats,
            "hist_lons":             hlons,
            "hist_alts":             halts,
            "hist_speeds":           hspds,
            "hist_headings":         hhdgs,
            "hist_timestamps":       tstamps,
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
    cat = ["esm_signature", "iff_mode", "flight_profile", "weather", "thermal_signature"]
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

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": _model_ready}


@app.get("/api/tracks")
def tracks_endpoint():
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    return get_tracks()


class PredictRequest(BaseModel):
    altitude_ft:      float
    speed_kts:        float
    rcs_m2:           float
    latitude:         float
    longitude:        float
    heading:          float
    esm_signature:    str
    iff_mode:         str
    flight_profile:   str
    weather:          str
    thermal_signature: str


@app.get("/api/model-stats")
def model_stats_endpoint():
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    return compute_model_stats()


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    inp       = req.model_dump()
    result    = predict_with_uncertainty(inp)
    xai       = compute_xai(inp, result)
    sv        = track_sensor_votes(inp)
    fusion    = compute_fusion(sv, SENSOR_BASE_WEIGHTS)
    ds        = dempster_shafer_fusion(sv)
    irst_base = sv["irst"]["base_conf"]
    return {
        **result,
        "xai":            xai,
        "sensor_votes":   sv,
        "fusion":         fusion,
        "ds_fusion":      ds,
        "weather_impact": {w: round(irst_base * f, 2) for w, f in WEATHER_IRST_FACTOR.items()},
    }


# ── SIMULATION ENDPOINTS ──────────────────────────────────────────────────────

from backend.sim import sim_mgr, assign_position
import uuid as _uuid
import random as _random

class SimSubmitRequest(BaseModel):
    altitude_ft:       float
    speed_kts:         float
    rcs_m2:            float
    heading:           float
    esm_signature:     str
    iff_mode:          str
    flight_profile:    str
    weather:           str
    thermal_signature: str


def generate_sim_hist(inp: dict, track_id: str):
    """Deterministic 14-point trajectory history from current kinematic state."""
    seed = sum(ord(c) for c in track_id)
    rng  = _random.Random(seed)
    N    = 14
    alt, spd, hdg = inp["altitude_ft"], inp["speed_kts"], inp["heading"]
    profile = inp["flight_profile"]

    _PROFILE_PARAMS = {
        "LOW_ALTITUDE_FLYING":   {"alt_trend": -20000, "spd_trend": 110, "hdg_jitter": 10, "alt_noise": 400},
        "AGGRESSIVE_MANEUVERS":  {"alt_trend":   4000, "spd_trend":  85, "hdg_jitter": 32, "alt_noise": 1600},
        "CLIMBING":              {"alt_trend": -12000, "spd_trend":  40, "hdg_jitter":  4, "alt_noise": 300},
        "DIVING":                {"alt_trend":  16000, "spd_trend":  65, "hdg_jitter":  6, "alt_noise": 500},
        "STABLE_CRUISE":         {"alt_trend":    600, "spd_trend":  15, "hdg_jitter":  3, "alt_noise": 200},
        "HOLDING_PATTERN":       {"alt_trend":    200, "spd_trend":  10, "hdg_jitter": 42, "alt_noise": 150},
        "EVASIVE_MANEUVERS":     {"alt_trend":   5000, "spd_trend": 100, "hdg_jitter": 48, "alt_noise": 2000},
        "LOITERING":             {"alt_trend":    800, "spd_trend":  20, "hdg_jitter": 22, "alt_noise": 300},
    }
    p = _PROFILE_PARAMS.get(profile, _PROFILE_PARAMS["STABLE_CRUISE"])

    now   = datetime.now(timezone.utc)
    alts, spds, hdgs, times = [], [], [], []
    hdg_acc = hdg

    for i in range(N):
        f        = i / (N - 1)
        base_alt = alt + p["alt_trend"] * (1 - f)
        noise    = (rng.random() - 0.5) * p["alt_noise"]
        alts.append(max(200, round(base_alt + noise)))

        base_spd = spd + p["spd_trend"] * (1 - f)
        spds.append(max(120, round(base_spd + (rng.random() - 0.5) * 20)))

        if i < N - 1:
            hdg_acc = (hdg_acc + (rng.random() - 0.5) * p["hdg_jitter"] * 2) % 360
            hdgs.append(round(hdg_acc, 1))
        else:
            hdgs.append(round(hdg, 1))

        ago = (N - 1 - i) * 5
        ts  = (now - timedelta(minutes=ago)).strftime("%H:%M UTC") if ago else now.strftime("%H:%M UTC")
        times.append(ts)

    alts[-1] = round(alt)
    spds[-1] = round(spd)
    return alts, spds, hdgs, times


@app.post("/sim/sessions")
def create_sim_session():
    sid = sim_mgr.create_session()
    return {"session_id": sid}


@app.get("/sim/{session_id}/tracks")
def get_sim_tracks(session_id: str):
    if not sim_mgr.session_exists(session_id):
        raise HTTPException(404, "Session not found")
    return {
        "session_id":        session_id,
        "tracks":            sim_mgr.get_tracks(session_id),
        "participant_count": sim_mgr.participant_count(session_id),
    }


@app.post("/sim/{session_id}/submit")
async def submit_sim_track(session_id: str, req: SimSubmitRequest):
    if not _model_ready:
        raise HTTPException(503, "Model not loaded")
    if not sim_mgr.session_exists(session_id):
        raise HTTPException(404, "Session not found")

    inp    = {**req.model_dump(), "latitude": 0.0, "longitude": 0.0}
    result = predict_with_uncertainty(inp)
    sv     = track_sensor_votes(inp)
    fusion = compute_fusion(sv, SENSOR_BASE_WEIGHTS)
    ds     = dempster_shafer_fusion(sv)
    pos    = assign_position(result["classification"])
    xai    = compute_xai(inp, result)
    anom   = detect_anomalies(inp)

    track_id = f"SIM-{str(_uuid.uuid4())[:4].upper()}"
    h_alts, h_spds, h_hdgs, h_times = generate_sim_hist(inp, track_id)

    track = {
        "track_id":        track_id,
        "submitted_at":    datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "ai_class":              result["classification"],
        "ai_conf":               result["confidence"],
        "ai_probs":              result["probabilities"],
        "epistemic_uncertainty": result["epistemic_uncertainty"],
        "uncertainty_label":     result["uncertainty_label"],
        "pos":                   pos,
        "sensor_votes":          sv,
        "fusion":                fusion,
        "ds_fusion":             ds,
        "xai":                   xai,
        "anomalies":             anom,
        "hist_alts":       h_alts,
        "hist_speeds":     h_spds,
        "hist_headings":   h_hdgs,
        "hist_timestamps": h_times,
        **req.model_dump(),
    }

    sim_mgr.add_track(session_id, track)
    await sim_mgr.broadcast(session_id, {"type": "new_track", "track": track})
    return track


@app.websocket("/sim/{session_id}/ws")
async def sim_ws(session_id: str, ws: WebSocket):
    if not sim_mgr.session_exists(session_id):
        await ws.close(code=4004)
        return
    await sim_mgr.connect(session_id, ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        sim_mgr.disconnect(session_id, ws)


# ── Static file serving (HF Spaces / single-container deployments) ────────────
# Mount AFTER all API + WebSocket routes so API paths take precedence.
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse as _FileResponse

_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _DIST.exists():
    # Serve Vite asset chunks (hashed filenames) under /assets
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="spa-assets")

    @app.get("/{full_path:path}")
    async def _spa_fallback(full_path: str):
        """SPA catch-all — return index.html for any unmatched GET so React Router works."""
        return _FileResponse(_DIST / "index.html")
