"""
VANGUARD AI — Retrain script  v2 (realistic overlap)
NATO standard 6-class classification (F/H/S/U/N/A)
Target accuracy: ~85-90% — classes have realistic real-world overlap/deception
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Realistic overlap design ──────────────────────────────────────────────────
#
#  Confusion expected (by design):
#    HOSTILE  <-> SUSPECT    (10-15%) — hostile spoofs civilian profile
#    SUSPECT  <-> UNKNOWN    (12-18%) — ambiguous non-cooperator
#    UNKNOWN  <-> NEUTRAL    (8-12%)  — unidentified commercial-sized contact
#    FRIEND   <-> ASS.FRIEND (5-8%)   — radio silence or IFF degradation
#    ASS.FRIEND <-> NEUTRAL  (5-8%)   — IFF-3C on both
#
#  Deception scenarios embedded in HOSTILE (~25% of hostile samples):
#    - Spoof civilian transponder (IFF_MODE_3C)
#    - Blend-in altitude (commercial cruise)
#    - Reduced speed / stable cruise approach

def generate_dataset(n_per_class: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []

    def n(val, std):  return val + rng.normal(0, std)
    def weather_thermal(base, w, fail_p=0.75):
        if w != 'Clear' and rng.random() < fail_p:
            return 'Not_Detected'
        return base

    W, WP = ['Clear', 'Cloudy', 'Rainy'], [0.65, 0.25, 0.10]

    for _ in range(n_per_class):

        # ── F  FRIEND ─────────────────────────────────────────────────────
        # 94% canonical: IFF Mode-5 + military profile
        #  6% edge: radio silence (covert op) → looks like SUSPECT/UNKNOWN
        w = rng.choice(W, p=WP)
        edge = rng.random() < 0.10
        rows.append({
            'classification':       'FRIEND',
            'altitude_ft':          n(rng.uniform(8000, 45000), 1200),
            'speed_kts':            n(rng.uniform(350, 780), 35),
            'rcs_m2':               rng.uniform(3.0, 22.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(46, 56),
            'longitude':            rng.uniform(5, 22),
            'electronic_signature': 'NO_IFF_RESPONSE' if edge else 'IFF_MODE_5',
            'flight_profile':       rng.choice(['STABLE_CRUISE', 'CLIMBING'], p=[0.65, 0.35]),
            'weather':              w,
            'thermal_signature':    weather_thermal(
                rng.choice(['Medium', 'High'], p=[0.5, 0.5]), w),
        })

        # ── H  HOSTILE ────────────────────────────────────────────────────
        # 65% canonical: stealth RCS + jamming/no-IFF + aggressive
        # 22% spoofing: IFF_MODE_3C + commercial altitude (blends with NEUTRAL/ASS.FRIEND)
        # 13% covert approach: slow + stable (blends with SUSPECT)
        w = rng.choice(W, p=WP)
        scenario = rng.choice(['canonical', 'spoof_civil', 'covert'], p=[0.65, 0.22, 0.13])
        if scenario == 'spoof_civil':
            rows.append({
                'classification':       'HOSTILE',
                'altitude_ft':          n(rng.uniform(28000, 40000), 800),  # commercial altitude
                'speed_kts':            n(rng.uniform(400, 560), 25),       # commercial speed
                'rcs_m2':               rng.uniform(0.5, 4.0),              # still small-ish
                'heading':              rng.uniform(0, 360),
                'latitude':             rng.uniform(43, 52),
                'longitude':            rng.uniform(20, 40),
                'electronic_signature': 'IFF_MODE_3C',                      # spoofed transponder
                'flight_profile':       'STABLE_CRUISE',
                'weather':              w,
                'thermal_signature':    weather_thermal('High', w),
            })
        elif scenario == 'covert':
            rows.append({
                'classification':       'HOSTILE',
                'altitude_ft':          n(rng.uniform(500, 12000), 600),
                'speed_kts':            n(rng.uniform(280, 520), 30),
                'rcs_m2':               rng.uniform(1.0, 6.0),
                'heading':              rng.uniform(0, 360),
                'latitude':             rng.uniform(43, 50),
                'longitude':            rng.uniform(25, 42),
                'electronic_signature': rng.choice(['NO_IFF_RESPONSE', 'UNKNOWN_EMISSION'], p=[0.6, 0.4]),
                'flight_profile':       'STABLE_CRUISE',
                'weather':              w,
                'thermal_signature':    weather_thermal(
                    rng.choice(['Medium', 'High'], p=[0.5, 0.5]), w),
            })
        else:  # canonical
            alt = rng.choice([
                n(rng.uniform(200,   7000),  400),
                n(rng.uniform(42000, 60000), 1000),
            ])
            rows.append({
                'classification':       'HOSTILE',
                'altitude_ft':          max(200, alt),
                'speed_kts':            n(rng.uniform(500, 1200), 50),
                'rcs_m2':               rng.uniform(0.01, 3.0),
                'heading':              rng.uniform(0, 360),
                'latitude':             rng.uniform(42, 50),
                'longitude':            rng.uniform(25, 42),
                'electronic_signature': rng.choice(
                    ['HOSTILE_JAMMING', 'NO_IFF_RESPONSE'], p=[0.45, 0.55]),
                'flight_profile':       rng.choice(
                    ['AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING'], p=[0.6, 0.4]),
                'weather':              w,
                'thermal_signature':    weather_thermal('High', w),
            })

        # ── S  SUSPECT ────────────────────────────────────────────────────
        # Ambiguous: shares feature space with both HOSTILE and UNKNOWN
        # No IFF + wrong area + variable aggression
        # Some look very hostile, some look almost unknown
        w = rng.choice(W, p=WP)
        aggressiveness = rng.random()   # 0=unknown-like, 1=hostile-like
        rows.append({
            'classification':       'SUSPECT',
            'altitude_ft':          n(rng.uniform(1000, 35000), 1500),
            'speed_kts':            n(rng.uniform(280 + aggressiveness * 300, 600 + aggressiveness * 300), 40),
            'rcs_m2':               rng.uniform(0.5 + (1 - aggressiveness) * 5, 6 + (1 - aggressiveness) * 12),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(43, 53),
            'longitude':            rng.uniform(18, 42),
            'electronic_signature': rng.choice(
                ['NO_IFF_RESPONSE', 'UNKNOWN_EMISSION', 'HOSTILE_JAMMING'],
                p=[0.55, 0.35, 0.10]),
            'flight_profile':       rng.choice(
                ['AGGRESSIVE_MANEUVERS', 'STABLE_CRUISE', 'CLIMBING'],
                p=[0.40, 0.35, 0.25]),
            'weather':              w,
            'thermal_signature':    weather_thermal(
                rng.choice(['Medium', 'High', 'Low'], p=[0.45, 0.35, 0.20]), w),
        })

        # ── U  UNKNOWN ────────────────────────────────────────────────────
        # Truly unidentified: broad distribution overlapping SUSPECT + NEUTRAL
        # Represents contacts that just appeared — no history, no context
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':       'UNKNOWN',
            'altitude_ft':          n(rng.uniform(500, 50000), 2500),
            'speed_kts':            n(rng.uniform(80, 780), 40),
            'rcs_m2':               rng.uniform(0.3, 55.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(42, 58),
            'longitude':            rng.uniform(-5, 42),
            'electronic_signature': rng.choice(
                ['UNKNOWN_EMISSION', 'NO_IFF_RESPONSE', 'IFF_MODE_3C'],
                p=[0.60, 0.30, 0.10]),
            'flight_profile':       rng.choice(
                ['STABLE_CRUISE', 'CLIMBING', 'AGGRESSIVE_MANEUVERS'],
                p=[0.65, 0.25, 0.10]),
            'weather':              w,
            'thermal_signature':    weather_thermal(
                rng.choice(['Not_Detected', 'Low', 'Medium', 'High'],
                            p=[0.35, 0.30, 0.25, 0.10]), w),
        })

        # ── N  NEUTRAL ────────────────────────────────────────────────────
        # Commercial/neutral-state aircraft
        # 90% canonical: large RCS + IFF-3C + cruise altitude
        # 10% ambiguous: smaller regional jet, occasionally UNKNOWN_EMISSION
        w = rng.choice(W, p=WP)
        ambiguous = rng.random() < 0.10
        rows.append({
            'classification':       'NEUTRAL',
            'altitude_ft':          n(rng.uniform(22000 if not ambiguous else 8000,
                                                   42000 if not ambiguous else 28000), 800),
            'speed_kts':            n(rng.uniform(350, 560), 25),
            'rcs_m2':               rng.uniform(8 if ambiguous else 30,
                                                 35 if ambiguous else 130),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(42, 56),
            'longitude':            rng.uniform(-10, 35),
            'electronic_signature': rng.choice(
                ['IFF_MODE_3C', 'UNKNOWN_EMISSION'],
                p=[0.75 if not ambiguous else 0.40,
                   0.25 if not ambiguous else 0.60]),
            'flight_profile':       'STABLE_CRUISE',
            'weather':              w,
            'thermal_signature':    weather_thermal('Medium', w),
        })

        # ── A  ASSUMED FRIEND ─────────────────────────────────────────────
        # Military with civil transponder — overlaps with both FRIEND and NEUTRAL
        # 92% canonical: IFF-3C + military-ish speed/RCS
        #  8% IFF degraded: looks like UNKNOWN
        w = rng.choice(W, p=WP)
        iff_degraded = rng.random() < 0.14
        rows.append({
            'classification':       'ASSUMED FRIEND',
            'altitude_ft':          n(rng.uniform(12000, 40000), 1200),
            'speed_kts':            n(rng.uniform(320, 680), 35),
            'rcs_m2':               rng.uniform(4.0, 28.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(44, 55),
            'longitude':            rng.uniform(3, 26),
            'electronic_signature': 'UNKNOWN_EMISSION' if iff_degraded else 'IFF_MODE_3C',
            'flight_profile':       rng.choice(
                ['STABLE_CRUISE', 'CLIMBING'], p=[0.68, 0.32]),
            'weather':              w,
            'thermal_signature':    weather_thermal(
                rng.choice(['Low', 'Medium', 'High'], p=[0.30, 0.50, 0.20]), w),
        })

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
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(torch.clamp(x, -10, 10))


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("=" * 64)
    print("  VANGUARD AI — NATO Retrain  (realistic overlap v2)")
    print("  Target: ~85-90% accuracy with meaningful class confusion")
    print("=" * 64)

    print("\n[1/5] Generating dataset with realistic overlap…")
    df = generate_dataset(n_per_class=1400)
    print(f"      {len(df)} samples · {df['classification'].nunique()} classes")
    for cls, cnt in df['classification'].value_counts().items():
        print(f"        {cls}: {cnt}")

    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/vanguard_air_tracks_fused.csv', index=False)
    print("      Saved data/vanguard_air_tracks_fused.csv")

    print("\n[2/5] Preprocessing…")
    cat      = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
    X_raw    = pd.get_dummies(df.drop(columns=['classification']), columns=cat)
    feat_cols = list(X_raw.columns)
    le = LabelEncoder()
    y  = le.fit_transform(df['classification'])
    print(f"      Features: {len(feat_cols)}   Classes: {list(le.classes_)}")

    X = X_raw.values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=SEED, stratify=y_tr)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    print(f"      Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

    print("\n[3/5] Class weights (inverse frequency)…")
    counts  = np.bincount(y_tr)
    w_arr   = 1.0 / counts
    w_arr  /= w_arr.mean()
    class_w = torch.FloatTensor(w_arr)
    for i, c in enumerate(le.classes_):
        print(f"        {c}: {class_w[i]:.3f}")

    BS       = 64
    train_dl = DataLoader(AircraftDataset(X_tr_s, y_tr), batch_size=BS, shuffle=True)

    print("\n[4/5] Training…")
    model     = ImprovedAircraftClassifier(len(feat_cols), len(le.classes_))
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    best_val  = 0.0
    patience  = 0
    MAX_PAT   = 20

    for epoch in range(1, 151):
        model.train()
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pv = model(torch.FloatTensor(X_va_s)).argmax(1).numpy()
        va = accuracy_score(y_va, pv)

        if va > best_val:
            best_val = va
            torch.save({'model_state_dict': model.state_dict()}, 'models/best_model.pt')
            patience = 0
            tag = ' <-- best'
        else:
            patience += 1
            tag = ''

        if epoch % 15 == 0 or tag:
            print(f"  Epoch {epoch:3d}  val={va:.3f}  best={best_val:.3f}{tag}")

        if patience >= MAX_PAT:
            print(f"  Early stop at epoch {epoch}")
            break

    print("\n[5/5] Test evaluation…")
    ckpt = torch.load('models/best_model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        tp = model(torch.FloatTensor(X_te_s)).argmax(1).numpy()

    tl = le.inverse_transform(y_te)
    pl = le.inverse_transform(tp)
    print(f"\n  Accuracy:    {accuracy_score(y_te, tp)*100:.1f}%")
    print(f"  F1 Macro:    {f1_score(y_te, tp, average='macro')*100:.1f}%")
    print(f"  F1 Weighted: {f1_score(y_te, tp, average='weighted')*100:.1f}%")
    print()
    print(classification_report(tl, pl))

    Path('models').mkdir(exist_ok=True)
    joblib.dump(scaler,    'models/scaler.joblib')
    joblib.dump(le,        'models/label_encoder.joblib')
    joblib.dump(feat_cols, 'models/feature_columns.joblib')
    print("  Artifacts saved. Restart FastAPI to load the new model.")


if __name__ == '__main__':
    train()
