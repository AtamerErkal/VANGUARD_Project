# 🛡️ VANGUARD AI — Tactical Aircraft Classification System

A real-time, AI-powered air picture system that classifies airborne contacts using sensor fusion and a PyTorch neural network. Built as a full-stack demo of human-machine teaming for military threat assessment.

![VANGUARD Tactical Interface](images/VANGUARD_1.png)

---

## Overview

VANGUARD AI simulates a multi-sensor fusion environment where contacts are classified into **6 NATO-standard threat categories**:

| Symbol | Class | Description |
|--------|-------|-------------|
| 🚨 H | **HOSTILE** | Confirmed hostile — aggressive maneuvers, jamming |
| ⚠️ S | **SUSPECT** | Suspicious behavior, no IFF confirmation |
| ❓ U | **UNKNOWN** | Unidentified — default state for new contacts |
| 🏳️ N | **NEUTRAL** | Neutral-state civil or commercial traffic |
| 🤝 A | **ASSUMED FRIEND** | Allied on civil transponder (IFF Mode-3C) |
| 🛡️ F | **FRIEND** | Confirmed friendly (IFF Mode-5) |

The AI classifies each contact by combining kinematic, electronic, radar, thermal and environmental sensor data — then a human expert can approve or override the decision.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VANGUARD TACTICAL UI                 │
│  React 18 + TypeScript + MapLibre GL + Plotly           │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                        │
│  • PyTorch MLP classifier  (20 features → 6 classes)   │
│  • Dempster-Shafer sensor fusion (Radar/ESM/IRST/IFF)   │
│  • Perturbation-based XAI  (feature importance)        │
│  • What-if scenario engine                             │
└─────────────────────────────────────────────────────────┘
```

### Neural Network

| Layer | Size | Activation |
|-------|------|------------|
| Input | 20 features | — |
| Dense 1 | 128 | BatchNorm + ReLU + Dropout 0.3 |
| Dense 2 | 64 | BatchNorm + ReLU + Dropout 0.2 |
| Dense 3 | 32 | BatchNorm + ReLU + Dropout 0.1 |
| Output | 6 classes | Softmax |

**Test set performance:** Accuracy 89.3% · F1 Macro 88.0% · F1 Weighted 89.4%

The model is trained on behavioural/sensor features only — no geographic coordinates — so it generalises across theaters.

### Sensor Fusion

Each sensor votes independently; votes are weighted and summed:

| Sensor | Weight | Signal |
|--------|--------|--------|
| Radar | 0.40 | RCS size → class hints |
| ESM | 0.35 | Electronic / jamming signature |
| IRST | 0.15 | Thermal signature (degrades in non-clear weather) |
| IFF | 0.10 | Transponder mode |

---

## Features

- **Live tactical map** — animated contacts with heading indicators, glassmorphism popups
- **AI classification badge** — class, confidence %, sensor fusion gap
- **Maneuver Envelope (3D)** — altitude × speed × time with turn-rate color coding
- **Explainable AI panel** — per-feature perturbation importance for each contact
- **What-if panel** — change any sensor parameter and see how the classification changes
- **Expert Approval** — approve or override AI decisions with audit trail
- **Model Online modal** — real F1/precision/recall metrics, 6×6 confusion matrix heatmap
- **Theater Threat Level** — CRITICAL / HIGH / ELEVATED / GUARDED / LOW aggregated from all contacts
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

Generates training data and saves model artifacts to `models/`.  
Expected output: **~89% accuracy**, F1 Macro ~88%.

### 3 — Start the backend

```bash
python run_server.py
```

API will be available at `http://localhost:8000`.

### 4 — Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Project Structure

```
vanguard-ai/
├── backend/
│   ├── main.py              # FastAPI app — inference, fusion, XAI, what-if
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── api.ts
│       ├── types.ts
│       └── components/
│           ├── Header.tsx         # Theater threat level, model/sensor modals
│           ├── TacticalMap.tsx    # MapLibre GL animated contact map
│           ├── SensorFusion.tsx   # Weighted sensor vote breakdown
│           ├── Trail3D.tsx        # Maneuver envelope (altitude×speed×time)
│           ├── XAIPanel.tsx       # Perturbation-based feature importance
│           ├── WhatIfPanel.tsx    # Interactive scenario engine
│           ├── ExpertApproval.tsx # Human-in-the-loop approval
│           └── AnomalyAlerts.tsx
├── models/                  # Saved PyTorch model + scalers (git-tracked)
├── data/                    # Generated CSV (git-ignored, rebuilt by retrain.py)
├── retrain.py               # Full training pipeline (data gen → train → eval)
├── run_server.py            # Backend launcher
└── images/
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/tracks` | All contacts with AI classification + fusion |
| POST | `/api/classify` | Classify a single contact |
| POST | `/api/whatif` | What-if scenario (modify features) |
| GET | `/api/model-stats` | Full metrics, per-class F1, confusion matrix |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 · TypeScript · Vite · Tailwind CSS |
| Map | MapLibre GL JS |
| 3D Charts | Plotly.js |
| Backend | FastAPI · Uvicorn |
| ML | PyTorch · scikit-learn |
| Data | NumPy · Pandas |

---

## License

MIT — see [LICENSE](LICENSE)
