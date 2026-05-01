"""
VANGUARD AI — Retrain  v3 (realistic imbalance + overlap)
NATO standard 6-class: F / H / S / U / N / A
- Class distribution mirrors real air traffic (HOSTILE rare, UNKNOWN dominant)
- Deception/overlap scenarios per class
- Class-weighted loss emphasises rare but critical HOSTILE/SUSPECT
Run: python retrain.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib
from pathlib import Path
import mlflow
import mlflow.pytorch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Realistic class counts ────────────────────────────────────────────────────
# Real air-picture: most contacts start UNKNOWN, few are ever truly HOSTILE
CLASS_COUNTS = {
    'UNKNOWN':        2800,   # 33% — every contact starts here
    'NEUTRAL':        2400,   # 29% — commercial / neutral-state traffic
    'ASSUMED FRIEND': 1200,   # 14% — allied military on civil transponder
    'FRIEND':          700,   #  8% — confirmed friendly (IFF Mode-5)
    'SUSPECT':         500,   #  6% — suspicious behaviour, no confirmation
    'HOSTILE':         800,   # 10% — slightly boosted so model sees enough examples
}
# Total: ~8400  HOSTILE+SUSPECT = 16%  UNKNOWN+NEUTRAL = 62%


# ── Data generation ───────────────────────────────────────────────────────────

def _sample(rng, lo, hi, noise=0.0):
    v = rng.uniform(lo, hi)
    return v + rng.normal(0, noise) if noise else v

def _thermal(base, weather, rng, fail_p=0.75):
    if weather != 'Clear' and rng.random() < fail_p:
        return 'Not_Detected'
    return base

W, WP = ['Clear', 'Cloudy', 'Rainy'], [0.65, 0.25, 0.10]


def _rows_FRIEND(rng, n):
    rows = []
    for _ in range(n):
        w             = rng.choice(W, p=WP)
        radio_silence = rng.random() < 0.10   # covert op → no IFF + unknown ESM
        rows.append({
            'classification':  'FRIEND',
            'altitude_ft':     _sample(rng, 8000,  45000, 1200),
            'speed_kts':       _sample(rng, 350,   780,   35),
            'rcs_m2':          _sample(rng, 3.0,   22.0),
            'heading':         _sample(rng, 0,     360),
            'latitude':        _sample(rng, 46,    56),
            'longitude':       _sample(rng, 5,     22),
            'esm_signature':   'UNKNOWN_EMISSION' if radio_silence else 'CLEAN',
            'iff_mode':        'NO_RESPONSE'  if radio_silence else 'IFF_MODE_5',
            'flight_profile':  rng.choice(['STABLE_CRUISE', 'CLIMBING'], p=[0.65, 0.35]),
            'weather':         w,
            'thermal_signature': _thermal(rng.choice(['Medium', 'High'], p=[0.5, 0.5]), w, rng),
        })
    return rows


def _rows_HOSTILE(rng, n):
    rows = []
    for _ in range(n):
        w   = rng.choice(W, p=WP)
        scn = rng.choice(['canonical', 'spoof_civil', 'covert'], p=[0.65, 0.22, 0.13])

        if scn == 'spoof_civil':        # mimics NEUTRAL — clean ESM + civil squawk
            rows.append({
                'classification':    'HOSTILE',
                'altitude_ft':       _sample(rng, 28000, 40000, 800),
                'speed_kts':         _sample(rng, 400,   560,   25),
                'rcs_m2':            _sample(rng, 0.5,   4.5),
                'heading':           _sample(rng, 0,     360),
                'latitude':          _sample(rng, 43,    52),
                'longitude':         _sample(rng, 20,    40),
                'esm_signature':     'CLEAN',           # spoofing: clean emissions
                'iff_mode':          'IFF_MODE_3C',     # spoofing: civil squawk
                'flight_profile':    'STABLE_CRUISE',
                'weather':           w,
                'thermal_signature': _thermal('High', w, rng),
            })
        elif scn == 'covert':           # slow approach → unknown ESM, no IFF
            rows.append({
                'classification':    'HOSTILE',
                'altitude_ft':       _sample(rng, 500,   12000, 600),
                'speed_kts':         _sample(rng, 250,   520,   30),
                'rcs_m2':            _sample(rng, 1.0,   7.0),
                'heading':           _sample(rng, 0,     360),
                'latitude':          _sample(rng, 43,    50),
                'longitude':         _sample(rng, 25,    42),
                'esm_signature':     'UNKNOWN_EMISSION',
                'iff_mode':          'NO_RESPONSE',
                'flight_profile':    'STABLE_CRUISE',
                'weather':           w,
                'thermal_signature': _thermal(rng.choice(['Medium', 'High'], p=[0.5, 0.5]), w, rng),
            })
        else:                           # canonical — active jamming, no IFF
            alt = rng.choice([_sample(rng, 200, 7000, 400),
                              _sample(rng, 42000, 60000, 1000)])
            rows.append({
                'classification':    'HOSTILE',
                'altitude_ft':       max(200, alt),
                'speed_kts':         _sample(rng, 500,   1200, 50),
                'rcs_m2':            _sample(rng, 0.01,  3.0),
                'heading':           _sample(rng, 0,     360),
                'latitude':          _sample(rng, 42,    50),
                'longitude':         _sample(rng, 25,    42),
                'esm_signature':     rng.choice(['HOSTILE_JAMMING', 'NOISE_JAMMING'], p=[0.55, 0.45]),
                'iff_mode':          'NO_RESPONSE',
                'flight_profile':    rng.choice(['AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING'], p=[0.6, 0.4]),
                'weather':           w,
                'thermal_signature': _thermal('High', w, rng),
            })
    return rows


def _rows_SUSPECT(rng, n):
    rows = []
    for _ in range(n):
        w   = rng.choice(W, p=WP)
        agg = rng.random()
        rows.append({
            'classification':    'SUSPECT',
            'altitude_ft':       _sample(rng, 1000,  35000, 1500),
            'speed_kts':         _sample(rng, 280 + agg * 300, 600 + agg * 300, 40),
            'rcs_m2':            _sample(rng, 0.5 + (1 - agg) * 4, 7 + (1 - agg) * 10),
            'heading':           _sample(rng, 0,     360),
            'latitude':          _sample(rng, 43,    53),
            'longitude':         _sample(rng, 18,    42),
            'esm_signature':     rng.choice(['UNKNOWN_EMISSION', 'NOISE_JAMMING', 'HOSTILE_JAMMING'],
                                            p=[0.55, 0.35, 0.10]),
            'iff_mode':          rng.choice(['NO_RESPONSE', 'IFF_MODE_3C', 'DEGRADED'],
                                            p=[0.55, 0.35, 0.10]),
            'flight_profile':    rng.choice(['AGGRESSIVE_MANEUVERS', 'STABLE_CRUISE', 'CLIMBING'],
                                            p=[0.40, 0.35, 0.25]),
            'weather':           w,
            'thermal_signature': _thermal(rng.choice(['Medium', 'High', 'Low'], p=[0.45, 0.35, 0.20]), w, rng),
        })
    return rows


def _rows_UNKNOWN(rng, n):
    rows = []
    for _ in range(n):
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':    'UNKNOWN',
            'altitude_ft':       _sample(rng, 500,   50000, 2500),
            'speed_kts':         _sample(rng, 80,    780,   40),
            'rcs_m2':            _sample(rng, 0.3,   55.0),
            'heading':           _sample(rng, 0,     360),
            'latitude':          _sample(rng, 42,    58),
            'longitude':         _sample(rng, -5,    42),
            'esm_signature':     rng.choice(['UNKNOWN_EMISSION', 'CLEAN', 'NOISE_JAMMING'],
                                            p=[0.60, 0.30, 0.10]),
            'iff_mode':          rng.choice(['NO_RESPONSE', 'IFF_MODE_3C', 'DEGRADED'],
                                            p=[0.55, 0.35, 0.10]),
            'flight_profile':    rng.choice(['STABLE_CRUISE', 'CLIMBING', 'AGGRESSIVE_MANEUVERS'],
                                            p=[0.65, 0.25, 0.10]),
            'weather':           w,
            'thermal_signature': _thermal(
                rng.choice(['Not_Detected', 'Low', 'Medium', 'High'], p=[0.35, 0.30, 0.25, 0.10]), w, rng),
        })
    return rows


def _rows_NEUTRAL(rng, n):
    rows = []
    for _ in range(n):
        w   = rng.choice(W, p=WP)
        amb = rng.random() < 0.12   # regional jet / ambiguous
        rows.append({
            'classification':    'NEUTRAL',
            'altitude_ft':       _sample(rng, 8000 if amb else 26000, 28000 if amb else 42000, 800),
            'speed_kts':         _sample(rng, 320,   560,   25),
            'rcs_m2':            _sample(rng, 8 if amb else 30, 35 if amb else 130),
            'heading':           _sample(rng, 0,     360),
            'latitude':          _sample(rng, 42,    56),
            'longitude':         _sample(rng, -10,   35),
            'esm_signature':     'UNKNOWN_EMISSION' if amb else 'CLEAN',
            'iff_mode':          'DEGRADED' if amb else 'IFF_MODE_3C',
            'flight_profile':    'STABLE_CRUISE',
            'weather':           w,
            'thermal_signature': _thermal('Medium', w, rng),
        })
    return rows


def _rows_ASSUMED_FRIEND(rng, n):
    rows = []
    for _ in range(n):
        w   = rng.choice(W, p=WP)
        deg = rng.random() < 0.14   # IFF equipment degraded → looks like UNKNOWN
        rows.append({
            'classification':    'ASSUMED FRIEND',
            'altitude_ft':       _sample(rng, 12000, 40000, 1200),
            'speed_kts':         _sample(rng, 320,   680,   35),
            'rcs_m2':            _sample(rng, 4.0,   28.0),
            'heading':           _sample(rng, 0,     360),
            'latitude':          _sample(rng, 44,    55),
            'longitude':         _sample(rng, 3,     26),
            'esm_signature':     'UNKNOWN_EMISSION' if deg else 'CLEAN',
            'iff_mode':          'DEGRADED' if deg else 'IFF_MODE_3C',
            'flight_profile':    rng.choice(['STABLE_CRUISE', 'CLIMBING'], p=[0.68, 0.32]),
            'weather':           w,
            'thermal_signature': _thermal(
                rng.choice(['Low', 'Medium', 'High'], p=[0.30, 0.50, 0.20]), w, rng),
        })
    return rows


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    gen = {
        'FRIEND':         _rows_FRIEND,
        'HOSTILE':        _rows_HOSTILE,
        'SUSPECT':        _rows_SUSPECT,
        'UNKNOWN':        _rows_UNKNOWN,
        'NEUTRAL':        _rows_NEUTRAL,
        'ASSUMED FRIEND': _rows_ASSUMED_FRIEND,
    }
    rows = []
    for cls, fn in gen.items():
        rows.extend(fn(rng, CLASS_COUNTS[cls]))
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ── Dataset / Model ───────────────────────────────────────────────────────────

class AircraftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class ImprovedAircraftClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(torch.clamp(x, -10, 10))


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("vanguard-air-classification")

    total = sum(CLASS_COUNTS.values())
    print("=" * 64)
    print("  VANGUARD AI — NATO Retrain  v3  (realistic imbalance)")
    print("=" * 64)
    print()
    for cls, cnt in CLASS_COUNTS.items():
        bar = '█' * int(cnt / total * 40)
        print(f"  {cls:>15}  {bar:<40}  {cnt:4d}  ({cnt/total*100:.0f}%)")

    print("\n[1/5] Generating dataset…")
    df = generate_dataset()
    print(f"      {len(df)} total samples")
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/vanguard_air_tracks_fused.csv', index=False)

    print("\n[2/5] Preprocessing…")
    cat      = ['esm_signature', 'iff_mode', 'flight_profile', 'weather', 'thermal_signature']
    # lat/lon are positional context for display only — not behavioural features
    X_raw    = pd.get_dummies(df.drop(columns=['classification', 'latitude', 'longitude']), columns=cat)
    feat_cols = list(X_raw.columns)
    le = LabelEncoder()
    y  = le.fit_transform(df['classification'])
    print(f"      Features: {len(feat_cols)}  Classes: {list(le.classes_)}")

    X = X_raw.values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.125, random_state=SEED, stratify=y_tr)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    print(f"      Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

    # Inverse-frequency class weights (penalise majority, boost minority)
    counts  = np.bincount(y_tr)
    w_arr   = 1.0 / np.sqrt(counts)   # sqrt damping — less aggressive than 1/n
    w_arr  /= w_arr.mean()
    class_w = torch.FloatTensor(w_arr)
    print("\n[3/5] Class weights:")
    for i, c in enumerate(le.classes_):
        print(f"        {c:>15}: {class_w[i]:.3f}  (n={counts[i]})")

    BS       = 64
    LR       = 2e-3
    train_dl = DataLoader(AircraftDataset(X_tr_s, y_tr), batch_size=BS, shuffle=True)

    print("\n[4/5] Training…")
    model     = ImprovedAircraftClassifier(len(feat_cols), len(le.classes_))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    best_val, patience = 0.0, 0

    with mlflow.start_run(run_name="vanguard-mlp-v3"):
        mlflow.log_params({
            "seed":          SEED,
            "batch_size":    BS,
            "learning_rate": LR,
            "weight_decay":  1e-4,
            "architecture":  "23→128→64→32→6",
            "dropout":       "0.3/0.2/0.1",
            "optimizer":     "AdamW",
            "scheduler":     "CosineAnnealing",
            "n_features":    len(feat_cols),
            "n_classes":     len(le.classes_),
            "train_samples": len(X_tr),
            "val_samples":   len(X_va),
            "test_samples":  len(X_te),
            **{f"class_{k.replace(' ', '_')}_n": v for k, v in CLASS_COUNTS.items()},
        })

        for epoch in range(1, 151):
            model.train()
            for Xb, yb in train_dl:
                optimizer.zero_grad()
                criterion(model(Xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                pv = model(torch.FloatTensor(X_va_s)).argmax(1).numpy()
            va = accuracy_score(y_va, pv)
            mlflow.log_metric("val_accuracy", va, step=epoch)

            if va > best_val:
                best_val = va
                torch.save({'model_state_dict': model.state_dict()}, 'models/best_model.pt')
                patience = 0
                print(f"  Epoch {epoch:3d}  val={va:.3f}  best={best_val:.3f}  <-- best")
            else:
                patience += 1
                if epoch % 15 == 0:
                    print(f"  Epoch {epoch:3d}  val={va:.3f}  best={best_val:.3f}")

            if patience >= 20:
                print(f"  Early stop at epoch {epoch}")
                break

        print("\n[5/5] Test evaluation…")
        ckpt = torch.load('models/best_model.pt', map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            tp = model(torch.FloatTensor(X_te_s)).argmax(1).numpy()

        tl, pl = le.inverse_transform(y_te), le.inverse_transform(tp)
        acc      = accuracy_score(y_te, tp)
        f1_macro = f1_score(y_te, tp, average='macro')
        f1_w     = f1_score(y_te, tp, average='weighted')

        print(f"\n  Accuracy:    {acc*100:.1f}%")
        print(f"  F1 Macro:    {f1_macro*100:.1f}%")
        print(f"  F1 Weighted: {f1_w*100:.1f}%\n")
        print(classification_report(tl, pl))

        mlflow.log_metrics({
            "test_accuracy":    round(acc, 4),
            "test_f1_macro":    round(f1_macro, 4),
            "test_f1_weighted": round(f1_w, 4),
        })
        per_class_f1 = f1_score(y_te, tp, average=None, labels=list(range(len(le.classes_))))
        for i, cls in enumerate(le.classes_):
            mlflow.log_metric(f"f1_{cls.replace(' ', '_')}", round(float(per_class_f1[i]), 4))

        # Gate: fail fast if macro F1 drops below threshold
        if f1_macro < 0.90:
            print(f"\n  [WARN] F1 Macro {f1_macro:.3f} < 0.90 regression threshold")

        Path('models').mkdir(exist_ok=True)
        joblib.dump(scaler,    'models/scaler.joblib')
        joblib.dump(le,        'models/label_encoder.joblib')
        joblib.dump(feat_cols, 'models/feature_columns.joblib')

        mlflow.log_artifacts('models', artifact_path="model-artifacts")
        print("  Artifacts saved & logged to MLflow. Restart FastAPI to load the new model.")


if __name__ == '__main__':
    train()
