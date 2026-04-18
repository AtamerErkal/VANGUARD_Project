import pandas as pd
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('data/processed/final_training_data.csv')
print(f'Original: {len(df)} samples')
print(f'Columns: {df.columns.tolist()}')

# ONLY keep essential columns for ML
essential_cols = [
    'altitude_ft', 'speed_kts', 'rcs_m2', 'latitude', 'longitude', 'heading',
    'electronic_signature', 'flight_profile', 'weather', 'thermal_signature',
    'classification'
]

# Check which columns exist
existing_cols = [col for col in essential_cols if col in df.columns]
print(f'\\nUsing columns: {existing_cols}')

df = df[existing_cols].copy()

# Clean numeric columns
numeric_cols = ['altitude_ft', 'speed_kts', 'rcs_m2', 'latitude', 'longitude', 'heading']

for col in numeric_cols:
    if col in df.columns:
        # Fill NaN with median
        df[col].fillna(df[col].median(), inplace=True)
        # Replace inf
        df[col].replace([np.inf, -np.inf], df[col].median(), inplace=True)
        # Clip extreme values
        if col == 'altitude_ft':
            df[col] = df[col].clip(0, 60000)
        elif col == 'speed_kts':
            df[col] = df[col].clip(0, 1000)
        elif col == 'rcs_m2':
            df[col] = df[col].clip(0.001, 200)
        elif col == 'latitude':
            df[col] = df[col].clip(-90, 90)
        elif col == 'longitude':
            df[col] = df[col].clip(-180, 180)
        elif col == 'heading':
            df[col] = df[col].clip(0, 360)

# Clean categorical columns
categorical_cols = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']

for col in categorical_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)

# Fill classification NaN
if df['classification'].isnull().any():
    df['classification'].fillna('NEUTRAL', inplace=True)

# Drop any remaining NaN rows
before = len(df)
df = df.dropna()
after = len(df)

print(f'\\nAfter cleaning: {after} samples (removed {before - after})')
print(f'\\nClass distribution:')
print(df['classification'].value_counts())

# Save
Path('data/processed').mkdir(parents=True, exist_ok=True)
df.to_csv('data/processed/cleaned_training_data.csv', index=False)
print('\\n✅ Saved: data/processed/cleaned_training_data.csv')