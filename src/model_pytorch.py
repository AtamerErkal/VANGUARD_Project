"""
src/model_pytorch.py

PyTorch implementation - Helsing için önemli!
Deep Learning tabanlı aircraft classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== DATASET CLASS ====================

class AircraftDataset(Dataset):
    """
    PyTorch Dataset for aircraft classification
    """
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


# ==================== MODEL ARCHITECTURE ====================

class AircraftClassifier(nn.Module):
    """
    Deep Neural Network for aircraft classification
    
    Architecture:
    - Input layer
    - 3 hidden layers with BatchNorm + Dropout
    - Output layer (6 classes)
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=6, dropout=0.3):
        super(AircraftClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        return self.network(x)
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ==================== TRAINING PIPELINE ====================

class TrainingPipeline:
    """
    Complete training pipeline - MLOps best practices
    """
    
    def __init__(self, model, device='auto'):
        self.model = model
        
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-3,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """
        Complete training loop with early stopping
        """
        logger.info(f"🚀 Starting training for {epochs} epochs")
        logger.info(f"   Training samples: {len(train_loader.dataset)}")
        logger.info(f"   Validation samples: {len(val_loader.dataset)}")
        logger.info(f"   Batch size: {train_loader.batch_size}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt')
                logger.info(f"   ✅ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"⏹️  Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Best validation loss: {best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        save_path = Path('models') / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, save_path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        load_path = Path('models') / filename
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"✅ Model loaded from {load_path}")


# ==================== DATA PREPARATION ====================

def prepare_data(csv_path, test_size=0.2, val_size=0.1, batch_size=64):
    """
    Prepare data for training
    
    Returns:
        train_loader, val_loader, test_loader, scaler, label_encoder, feature_names
    """
    logger.info("📊 Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"   Loaded {len(df)} samples")
    
    # Features (numerical + categorical)
    numerical_features = ['altitude_ft', 'speed_kts', 'rcs_m2', 'latitude', 'longitude', 'heading']
    categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    # Get feature columns
    feature_cols = [col for col in df_encoded.columns if col not in ['classification', 'callsign', 'icao24', 'timestamp', 'origin_country', 'aircraft_type']]
    
    X = df_encoded[feature_cols].values
    y = df['classification'].values
    
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Classes: {len(np.unique(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    # Train/val split
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = AircraftDataset(X_train_scaled, y_train)
    val_dataset = AircraftDataset(X_val_scaled, y_val)
    test_dataset = AircraftDataset(X_test_scaled, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Save preprocessing artifacts
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    joblib.dump(feature_cols, 'models/feature_columns.joblib')
    
    logger.info("   ✅ Preprocessing artifacts saved")
    
    return train_loader, val_loader, test_loader, scaler, label_encoder, feature_cols


# ==================== EVALUATION ====================

def evaluate_model(model, test_loader, label_encoder, device='cpu'):
    """
    Comprehensive model evaluation
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            
            # Get predictions
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n✅ Confusion matrix saved: models/confusion_matrix.png")
    
    # Overall accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
    
    return accuracy, all_preds, all_targets, all_probs


# ==================== MAIN TRAINING SCRIPT ====================

def train_model(data_path='data/processed/training_data.csv', epochs=50):
    """
    Main training function
    """
    print("🛡️  VANGUARD AI - PyTorch Model Training")
    print("="*60)
    print()
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler, label_encoder, feature_cols = prepare_data(
        data_path, 
        batch_size=64
    )
    
    # Model setup
    input_dim = len(feature_cols)
    num_classes = len(label_encoder.classes_)
    
    print(f"\n🏗️  Building model...")
    print(f"   Input features: {input_dim}")
    print(f"   Output classes: {num_classes}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    model = AircraftClassifier(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_classes=num_classes,
        dropout=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Training pipeline
    pipeline = TrainingPipeline(model)
    
    # Train
    print("\n" + "="*60)
    history = pipeline.train(
        train_loader, 
        val_loader, 
        epochs=epochs,
        early_stopping_patience=10
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    # Load best model
    pipeline.load_checkpoint('best_model.pt')
    
    accuracy, preds, targets, probs = evaluate_model(
        model, 
        test_loader, 
        label_encoder,
        device=pipeline.device
    )
    
    print("\n✅ Training complete!")
    print(f"   Best model saved: models/best_model.pt")
    print(f"   Test accuracy: {accuracy:.2f}%")
    
    return model, scaler, label_encoder, feature_cols, history


# ==================== INFERENCE CLASS ====================

class VanguardInference:
    """
    Production inference class
    """
    
    def __init__(self, model_path='models/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load preprocessing
        self.scaler = joblib.load('models/scaler.joblib')
        self.label_encoder = joblib.load('models/label_encoder.joblib')
        self.feature_cols = joblib.load('models/feature_columns.joblib')
        
        # Load model
        input_dim = len(self.feature_cols)
        num_classes = len(self.label_encoder.classes_)
        
        self.model = AircraftClassifier(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            num_classes=num_classes
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Model loaded for inference")
    
    def predict(self, input_data):
        """
        Predict classification for single input
        
        Args:
            input_data: dict with features
        
        Returns:
            dict with prediction and confidence
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # One-hot encode
        categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
        df_encoded = pd.get_dummies(df, columns=categorical_features)
        
        # Align columns with training data
        for col in self.feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        df_encoded = df_encoded[self.feature_cols]
        
        # Scale
        X = self.scaler.transform(df_encoded.values)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Decode
        predicted_class = self.label_encoder.inverse_transform([predicted.item()])[0]
        confidence_value = confidence.item()
        
        # All class probabilities
        all_probs = {
            cls: float(prob) 
            for cls, prob in zip(self.label_encoder.classes_, probs[0].cpu().numpy())
        }
        
        return {
            'classification': predicted_class,
            'confidence': confidence_value,
            'all_probabilities': all_probs
        }


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    # Check if data exists
    data_path = Path('data/processed/training_data.csv')
    
    if not data_path.exists():
        print("⚠️  Training data not found!")
        print("   Please run data collection first:")
        print("   >>> from src.data_collection import OpenSkyCollector")
        print("   >>> collector = OpenSkyCollector()")
        print("   >>> dataset = collector.collect_dataset(duration_minutes=120)")
        print("   >>> collector.save_dataset(dataset, 'training_data.csv')")
    else:
        # Train model
        model, scaler, label_encoder, feature_cols, history = train_model(
            data_path=str(data_path),
            epochs=50
        )
        
        print("\n🚀 Model ready for inference!")
        print("\nExample usage:")
        print(">>> from src.model_pytorch import VanguardInference")
        print(">>> inference = VanguardInference()")
        print(">>> result = inference.predict(input_data)")