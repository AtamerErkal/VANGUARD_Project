---
title: VANGUARD AI
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# 🛡️ VANGUARD AI — Tactical Aircraft Classification System

A real-time, AI-powered air picture system that classifies airborne contacts using multi-sensor fusion and a PyTorch neural network. Built as a full-stack demo of human-machine teaming for military threat assessment.

![VANGUARD Tactical Interface](images/VANGUARD_1.png)

---

## Overview

VANGUARD AI simulates a multi-sensor fusion environment where contacts are classified into **6 NATO-standard threat categories**:

| Symbol | Class | Description |
|--------|-------|-------------|
| 🚨 H | **HOSTILE** | Confirmed hostile — aggressive maneuvers, active jamming |
| ⚠️ S | **SUSPECT** | Suspicious behavior, no IFF confirmation |
| ❓ U | **UNKNOWN** | Unidentified — default state for all new contacts |
| 🏳️ N | **NEUTRAL** | Neutral-state civil or commercial traffic |
| 🤝 A | **ASSUMED FRIEND** | Allied on civil transponder (IFF Mode-3C) |
| 🛡️ F | **FRIEND** | Confirmed friendly (IFF Mode-5 crypto) |

The AI classifies each contact by combining kinematic, radar, ESM, IRST and IFF sensor data — then a human expert can approve or override the decision.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VANGUARD TACTICAL UI                 │
│  React 18 · TypeScript · MapLibre GL · Plotly           │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                        │
│  • PyTorch MLP classifier  (23 features → 6 classes)   │
│  • Weighted sensor fusion  (Radar/ESM/IRST/IFF)        │
│  • Perturbation-based XAI  (feature importance)        │
│  • What-if scenario engine                             │
│  • Anomaly detection                                   │
└─────────────────────────────────────────────────────────┘
```

### Neural Network

| Layer | Size | Regularization |
|-------|------|----------------|
| Input | 23 features | — |
| Dense 1 | 128 | BatchNorm + ReLU + Dropout 0.3 |
| Dense 2 | 64 | BatchNorm + ReLU + Dropout 0.2 |
| Dense 3 | 32 | BatchNorm + ReLU + Dropout 0.1 |
| Output | 6 classes | Softmax |

**Test set performance:** Accuracy **94.0%** · F1 Macro **92.0%** · F1 Weighted **94.1%**

The model is trained on behavioural and sensor features only — **no geographic coordinates** — so it generalises across theaters rather than memorising locations.

Training data is **class-imbalanced** to reflect real air traffic (UNKNOWN 33%, NEUTRAL 29%, down to SUSPECT 6%), with inverse-frequency class weights to compensate. Deception scenarios are included: HOSTILE spoofing civil IFF, FRIEND in radio silence, ASSUMED FRIEND with degraded transponder.

### Sensor Fusion

Each sensor votes **independently**; votes are weighted and summed. ESM and IFF are modelled as separate sensors because they measure fundamentally different things:

| Sensor | Weight | What it measures |
|--------|--------|-----------------|
| **Radar** | 0.40 | RCS + kinematics — size and speed of the contact |
| **ESM** | 0.35 | *Passive* — what electromagnetic signals the aircraft is **emitting or jamming** |
| **IRST** | 0.15 | Thermal signature — degrades automatically in Cloudy/Rainy weather |
| **IFF** | 0.10 | *Active* — transponder challenge/response (Mode-5 crypto, Mode-3C civil, no reply) |

**ESM vs IFF distinction:** ESM intercepts the aircraft's own emissions (jamming, radar, comms). IFF is a separate interrogation system that asks "who are you?" and waits for a coded reply. A hostile aircraft can simultaneously jam (high ESM threat) while spoofing a civil squawk (IFF_MODE_3C) — this combination triggers an **ESM–IFF Conflict** anomaly.

#### ESM signature values
| Value | Meaning |
|-------|---------|
| `CLEAN` | No unusual emissions — civil or quiet military |
| `UNKNOWN_EMISSION` | Some emission detected, source unidentified |
| `NOISE_JAMMING` | Broadband jamming — suspicious |
| `HOSTILE_JAMMING` | Targeted electronic attack — hostile indicator |

#### IFF mode values
| Value | Meaning |
|-------|---------|
| `IFF_MODE_5` | Military crypto — confirmed friend |
| `IFF_MODE_3C` | Civil altitude transponder — neutral or assumed friend |
| `DEGRADED` | Intermittent signal — equipment fault or partial jamming |
| `NO_RESPONSE` | No transponder reply — suspect |

---

## Features

- **Live tactical map** — animated contacts with heading indicators, glassmorphism popups, approval state reflected on marker shape
- **AI classification badge** — class, confidence %, ESM/IFF fields, sensor fusion gap
- **Maneuver Envelope (3D)** — altitude × speed × time with turn-rate color coding (blue → straight, red → sharp turn)
- **Explainable AI panel** — perturbation-based feature importance per contact, supporting/conflicting direction
- **What-if panel** — independent ESM and IFF presets + kinematic sliders, live re-classification with diff view
- **Expert Approval** — human-in-the-loop approve or override with audit trail
- **Anomaly detection** — IFF–Maneuver Conflict, RCS–IFF Mismatch, ESM–IFF Conflict (deception), Terrain Hugging, High-Speed Non-Cooperative
- **Model Online modal** — live F1/precision/recall metrics, per-class bars, 6×6 confusion matrix heatmap
- **Theater Threat Level** — CRITICAL / HIGH / ELEVATED / GUARDED / LOW, pulsing indicator for high threat
- **Sensors Active modal** — explains AI confidence vs fused probability gap and why it matters
- **Collapsible side panel** — full-screen map mode

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1 — Clone & install

```bash
git clone https://github.com/AtamerErkal/VANGUARD_Project.git
cd VANGUARD_Project
```

**Backend:**
```bash
pip install -r backend/requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2 — Train the model

```bash
python retrain.py
```

Generates ~8 400 synthetic training samples with realistic class imbalance and deception scenarios. Saves model artifacts to `models/`.

Expected output: **~94% accuracy**, F1 Macro ~92%.

### 3 — Start the backend

```bash
python run_server.py
```

> **Note:** Use `run_server.py` instead of `python -m uvicorn` — the direct launcher ensures the model loads correctly in the same process.

API available at `http://localhost:8000`.

### 4 — Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

---

## Project Structure

```
vanguard-ai/
├── backend/
│   ├── main.py              # FastAPI — inference, sensor fusion, XAI, anomaly, what-if
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx          # Layout, track selection, threat score
│       ├── api.ts
│       ├── types.ts         # Track, SensorVote, ModelStats, ESM_SIGS, IFF_MODES
│       └── components/
│           ├── Header.tsx         # Theater threat level, Model Online, Sensors Active modals
│           ├── TacticalMap.tsx    # MapLibre GL animated contact map
│           ├── SensorFusion.tsx   # Per-sensor vote cards + fused probability bars
│           ├── Trail3D.tsx        # Maneuver envelope (altitude × speed × time)
│           ├── XAIPanel.tsx       # Perturbation-based feature importance
│           ├── WhatIfPanel.tsx    # ESM + IFF presets, kinematic sliders, re-classify
│           ├── ExpertApproval.tsx # Human-in-the-loop override
│           └── AnomalyAlerts.tsx
├── models/                  # PyTorch checkpoint + scalers (git-tracked, 23-feature)
├── data/                    # Generated CSV (git-ignored — rebuilt by retrain.py)
├── retrain.py               # Full pipeline: data gen → train → eval → save artifacts
├── run_server.py            # Backend launcher
└── images/
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — `{ model_ready: bool }` |
| GET | `/api/tracks` | All contacts with AI class, sensor votes, fusion, XAI, anomalies |
| POST | `/api/predict` | Classify a single contact, returns XAI + fusion |
| GET | `/api/model-stats` | Accuracy, per-class F1/P/R, 6×6 confusion matrix |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 · TypeScript · Vite · Tailwind CSS |
| Map | MapLibre GL JS |
| 3D Charts | Plotly.js |
| Backend | FastAPI · Uvicorn |
| ML | PyTorch · scikit-learn · NumPy · Pandas |

---

## License

MIT — see [LICENSE](LICENSE)
