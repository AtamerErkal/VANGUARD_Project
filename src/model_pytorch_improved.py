"""
src/model_pytorch_improved.py

Improved version with better numerical stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AircraftDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ImprovedAircraftClassifier(nn.Module):
    """Improved architecture with better numerical stability"""
    
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)
        )
        
        # Better weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        # Add small epsilon to avoid numerical issues
        x = torch.clamp(x, -10, 10)
        return self.network(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def prepare_data_improved(csv_path, test_size=0.2, val_size=0.1, batch_size=32):
    """Improved data preparation with better normalization"""
    
    logger.info("📊 Loading and preparing data...")
    
    df = pd.read_csv(csv_path)
    logger.info(f"   Loaded {len(df)} samples")
    
    # Features
    numerical_features = ['altitude_ft', 'speed_kts', 'rcs_m2', 'latitude', 'longitude', 'heading']
    categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    feature_cols = [col for col in df_encoded.columns 
                   if col not in ['classification', 'callsign', 'icao24', 'timestamp', 
                                 'origin_country', 'aircraft_type', 'time_position', 
                                 'last_contact', 'baro_altitude', 'on_ground', 'velocity',
                                 'true_track', 'vertical_rate', 'sensors', 'geo_altitude',
                                 'squawk', 'spi', 'position_source']]
    
    X = df_encoded[feature_cols].values
    y = df['classification'].values
    
    # Check for invalid values
    X = X.astype(float)
    if np.isnan(X).any() or np.isinf(X).any():
        logger.warning("   Found NaN/Inf in features! Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Classes: {len(np.unique(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Use RobustScaler instead of StandardScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Final check
    if np.isnan(X_train_scaled).any():
        logger.error("   NaN values after scaling!")
        X_train_scaled = np.nan_to_num(X_train_scaled)
        X_val_scaled = np.nan_to_num(X_val_scaled)
        X_test_scaled = np.nan_to_num(X_test_scaled)
    
    # Create datasets
    train_dataset = AircraftDataset(X_train_scaled, y_train)
    val_dataset = AircraftDataset(X_val_scaled, y_val)
    test_dataset = AircraftDataset(X_test_scaled, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Save preprocessing
    Path('models').mkdir(exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    joblib.dump(feature_cols, 'models/feature_columns.joblib')
    
    logger.info("   ✅ Preprocessing complete")
    
    return train_loader, val_loader, test_loader, scaler, label_encoder, feature_cols

def train_improved(epochs=50):
    """Train with improved stability"""
    
    print("🛡️  VANGUARD AI - Improved Training")
    print("="*60)
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler, label_encoder, feature_cols = prepare_data_improved(
        'data/processed/cleaned_training_data.csv',
        batch_size=32
    )
    
    # Model
    input_dim = len(feature_cols)
    num_classes = len(label_encoder.classes_)
    
    model = ImprovedAircraftClassifier(input_dim, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"🖥️  Device: {device}")
    logger.info(f"🏗️  Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    patience = 0
    max_patience = 10
    
    logger.info("🚀 Starting training...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Check for NaN in batch
            if torch.isnan(data).any():
                logger.warning("   NaN in batch, skipping...")
                continue
            
            optimizer.zero_grad()
            output = model(data)
            
            # Check output
            if torch.isnan(output).any():
                logger.warning("   NaN in output, skipping...")
                continue
            
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                logger.warning("   NaN loss, skipping...")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_acc = 100 * correct / total if total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] "
                   f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
                   f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pt')
            logger.info(f"   ✅ Best model saved (acc: {val_acc:.2f}%)")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                logger.info("⏹️  Early stopping")
                break
    
    logger.info(f"✅ Training complete! Best accuracy: {best_val_acc:.2f}%")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    train_improved()
