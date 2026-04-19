"""
VANGUARD AI — Retrain script
NATO standard 6-class classification (F/H/S/U/N/A)
Fixes: balanced data, heading feature, class weights, cleaner decision boundaries
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

# ── NATO class definitions ────────────────────────────────────────────────────
# F  Friend         — confirmed friendly military (IFF Mode-5)
# H  Hostile        — confirmed enemy (jamming/no-IFF + aggressive + stealth RCS)
# S  Suspect        — assumed hostile, behavior warrants intercept
# U  Unknown        — unidentified, insufficient data to classify
# N  Neutral        — non-threatening, identifiable as civilian/neutral state
# A  Assumed Friend — friendly characteristics but not fully confirmed (IFF-3C on mil)

def generate_dataset(n_per_class: int = 1400) -> pd.DataFrame:
    rng  = np.random.default_rng(SEED)
    rows = []

    def noisy(val, std): return val + rng.normal(0, std)
    def weather_thermal(base_thermal, w):
        if w != 'Clear' and rng.random() < 0.75:
            return 'Not_Detected'
        return base_thermal

    W = ['Clear', 'Cloudy', 'Rainy']
    WP = [0.65, 0.25, 0.10]

    for _ in range(n_per_class):

        # ── F  FRIEND ─────────────────────────────────────────────────────
        # Allied military: IFF Mode-5 (encrypted), medium RCS, military ops altitude
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':       'FRIEND',
            'altitude_ft':          noisy(rng.uniform(8000, 45000), 1200),
            'speed_kts':            noisy(rng.uniform(350, 780), 35),
            'rcs_m2':               rng.uniform(3.0, 22.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(46, 56),
            'longitude':            rng.uniform(5, 22),
            'electronic_signature': 'IFF_MODE_5',
            'flight_profile':       rng.choice(['STABLE_CRUISE', 'CLIMBING'], p=[0.65, 0.35]),
            'weather':              w,
            'thermal_signature':    weather_thermal(rng.choice(['Medium', 'High'], p=[0.5, 0.5]), w),
        })

        # ── H  HOSTILE ────────────────────────────────────────────────────
        # Enemy: stealth RCS, jamming or no-IFF, aggressive, low-alt penetration or hi-alt dash
        w = rng.choice(W, p=WP)
        alt = rng.choice([
            noisy(rng.uniform(200,  7000),  400),   # low-alt terrain masking
            noisy(rng.uniform(42000, 60000), 1000),  # high-alt fast dash
        ])
        rows.append({
            'classification':       'HOSTILE',
            'altitude_ft':          max(200, alt),
            'speed_kts':            noisy(rng.uniform(500, 1200), 50),
            'rcs_m2':               rng.uniform(0.01, 3.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(42, 50),
            'longitude':            rng.uniform(25, 42),
            'electronic_signature': rng.choice(['HOSTILE_JAMMING', 'NO_IFF_RESPONSE'], p=[0.45, 0.55]),
            'flight_profile':       rng.choice(['AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING'], p=[0.6, 0.4]),
            'weather':              w,
            'thermal_signature':    weather_thermal('High', w),
        })

        # ── S  SUSPECT ────────────────────────────────────────────────────
        # Assumed hostile: no IFF + aggressive + wrong area, but not yet confirmed
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':       'SUSPECT',
            'altitude_ft':          noisy(rng.uniform(1000, 28000), 1000),
            'speed_kts':            noisy(rng.uniform(300, 900), 40),
            'rcs_m2':               rng.uniform(0.8, 14.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(43, 52),
            'longitude':            rng.uniform(20, 40),
            'electronic_signature': rng.choice(['NO_IFF_RESPONSE', 'UNKNOWN_EMISSION'], p=[0.65, 0.35]),
            'flight_profile':       rng.choice(['AGGRESSIVE_MANEUVERS', 'STABLE_CRUISE', 'CLIMBING'], p=[0.5, 0.3, 0.2]),
            'weather':              w,
            'thermal_signature':    weather_thermal(rng.choice(['Medium', 'High'], p=[0.55, 0.45]), w),
        })

        # ── U  UNKNOWN ────────────────────────────────────────────────────
        # Truly unidentified: appeared on radar, parameters ambiguous, no clear intent
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':       'UNKNOWN',
            'altitude_ft':          noisy(rng.uniform(500, 50000), 2000),
            'speed_kts':            noisy(rng.uniform(80, 750), 30),
            'rcs_m2':               rng.uniform(0.5, 45.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(42, 58),
            'longitude':            rng.uniform(-5, 42),
            'electronic_signature': rng.choice(['UNKNOWN_EMISSION', 'NO_IFF_RESPONSE'], p=[0.70, 0.30]),
            'flight_profile':       rng.choice(['STABLE_CRUISE', 'CLIMBING'], p=[0.75, 0.25]),
            'weather':              w,
            'thermal_signature':    weather_thermal(rng.choice(['Not_Detected', 'Low', 'Medium'], p=[0.4, 0.35, 0.25]), w),
        })

        # ── N  NEUTRAL ────────────────────────────────────────────────────
        # Non-threatening: commercial airliner or neutral-state aircraft
        # Large RCS, civil/cruise altitude, moderate speed, civil corridor
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':       'NEUTRAL',
            'altitude_ft':          noisy(rng.uniform(28000, 42000), 600),
            'speed_kts':            noisy(rng.uniform(380, 560), 20),
            'rcs_m2':               rng.uniform(30, 130),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(42, 56),
            'longitude':            rng.uniform(-10, 35),
            'electronic_signature': rng.choice(['IFF_MODE_3C', 'UNKNOWN_EMISSION'], p=[0.80, 0.20]),
            'flight_profile':       'STABLE_CRUISE',
            'weather':              w,
            'thermal_signature':    weather_thermal('Medium', w),
        })

        # ── A  ASSUMED FRIEND ─────────────────────────────────────────────
        # Military-like profile on civilian transponder (IFF-3C), or known allied
        # Smaller RCS than neutral, faster, military altitude
        w = rng.choice(W, p=WP)
        rows.append({
            'classification':       'ASSUMED FRIEND',
            'altitude_ft':          noisy(rng.uniform(12000, 40000), 1000),
            'speed_kts':            noisy(rng.uniform(320, 650), 30),
            'rcs_m2':               rng.uniform(4.0, 26.0),
            'heading':              rng.uniform(0, 360),
            'latitude':             rng.uniform(44, 55),
            'longitude':            rng.uniform(3, 26),
            'electronic_signature': 'IFF_MODE_3C',
            'flight_profile':       rng.choice(['STABLE_CRUISE', 'CLIMBING'], p=[0.70, 0.30]),
            'weather':              w,
            'thermal_signature':    weather_thermal(rng.choice(['Low', 'Medium'], p=[0.45, 0.55]), w),
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ── Dataset / Model ───────────────────────────────────────────────────────────

class AircraftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):           return len(self.y)
    def __getitem__(self, i):    return self.X[i], self.y[i]


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
    print("=" * 62)
    print("  VANGUARD AI — NATO Classification Retrain")
    print("  F=Friend  H=Hostile  S=Suspect  U=Unknown  N=Neutral  A=AssumedFriend")
    print("=" * 62)

    # 1. Generate
    print("\n[1/5] Generating balanced NATO dataset…")
    df = generate_dataset(n_per_class=1400)   # 8400 total
    print(f"      {len(df)} samples · {df['classification'].nunique()} classes")
    for cls, cnt in df['classification'].value_counts().items():
        print(f"        {cls}: {cnt}")

    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/vanguard_air_tracks_fused.csv', index=False)
    print("      Saved → data/vanguard_air_tracks_fused.csv")

    # 2. Encode
    print("\n[2/5] Preprocessing…")
    cat      = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
    X_raw    = pd.get_dummies(df.drop(columns=['classification']), columns=cat)
    feat_cols = list(X_raw.columns)

    le = LabelEncoder()
    y  = le.fit_transform(df['classification'])
    print(f"      Features: {len(feat_cols)}   Classes: {list(le.classes_)}")

    X = X_raw.values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.125, random_state=SEED, stratify=y_tr)

    scaler   = RobustScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_va_s   = scaler.transform(X_va)
    X_te_s   = scaler.transform(X_te)
    print(f"      Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

    # 3. Class weights
    counts  = np.bincount(y_tr)
    w_arr   = (1.0 / counts) * len(le.classes_) / counts.sum() * counts.sum()
    w_arr  /= w_arr.mean()
    class_w = torch.FloatTensor(w_arr)
    print(f"\n[3/5] Class weights:")
    for i, c in enumerate(le.classes_):
        print(f"        {c}: {class_w[i]:.3f}")

    BS       = 64
    train_dl = DataLoader(AircraftDataset(X_tr_s, y_tr), batch_size=BS, shuffle=True)
    val_dl   = DataLoader(AircraftDataset(X_va_s, y_va), batch_size=BS)

    # 4. Train
    print("\n[4/5] Training…")
    model     = ImprovedAircraftClassifier(len(feat_cols), len(le.classes_))
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    best_val_acc = 0.0
    patience     = 0
    MAX_PAT      = 18
    EPOCHS       = 150

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_va_s)).argmax(1).numpy()
        val_acc = accuracy_score(y_va, preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, 'models/best_model.pt')
            patience = 0
            marker = '✓'
        else:
            patience += 1
            marker = ''

        if epoch % 15 == 0 or patience == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  val={val_acc:.3f}  best={best_val_acc:.3f}  {marker}")

        if patience >= MAX_PAT:
            print(f"  Early stop at epoch {epoch}")
            break

    # 5. Evaluate
    print("\n[5/5] Test set evaluation…")
    ckpt = torch.load('models/best_model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        te_preds = model(torch.FloatTensor(X_te_s)).argmax(1).numpy()

    y_te_lbl = le.inverse_transform(y_te)
    y_pr_lbl = le.inverse_transform(te_preds)

    print(f"\n  Accuracy:    {accuracy_score(y_te, te_preds)*100:.1f}%")
    print(f"  F1 Macro:    {f1_score(y_te, te_preds, average='macro')*100:.1f}%")
    print(f"  F1 Weighted: {f1_score(y_te, te_preds, average='weighted')*100:.1f}%")
    print()
    print(classification_report(y_te_lbl, y_pr_lbl))

    # Save artifacts
    Path('models').mkdir(exist_ok=True)
    joblib.dump(scaler,     'models/scaler.joblib')
    joblib.dump(le,         'models/label_encoder.joblib')
    joblib.dump(feat_cols,  'models/feature_columns.joblib')
    print("  Artifacts saved → models/")
    print("\n  Restart FastAPI server to load the new model.")


if __name__ == '__main__':
    train()
