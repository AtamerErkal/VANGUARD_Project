# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import os

def run_training():
    """
    Trains a model on the new FUSED dataset, which includes data from
    simulated Radar and EO/IR sensors.
    """
    
    mlflow.set_experiment("VANGUARD - Sensor Fusion Model")

    DATA_PATH = os.path.join("data", "vanguard_air_tracks_fused.csv")
    MODEL_DIR = "models"
    
    with mlflow.start_run(run_name="RandomForest_Fusion_Data"):
        print("MLflow run started for sensor fusion model...")
        
        # --- DATA LOADING AND PREPROCESSING ---
        df = pd.read_csv(DATA_PATH)
        
        # Define which columns are categorical for one-hot encoding
        categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
        
        X = df.drop(['classification', 'latitude', 'longitude'], axis=1)
        y = df['classification']
        
        # One-hot encode ALL categorical features, including the new ones
        X_encoded = pd.get_dummies(X, columns=categorical_features)
        
        joblib.dump(X_encoded.columns, os.path.join(MODEL_DIR, 'training_columns.joblib'))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- MODEL TRAINING ---
        # We can use a simpler parameter grid for faster iteration
        param_grid = {
            'n_estimators': [150],
            'max_depth': [20],
            'min_samples_split': [5]
        }
        
        mlflow.log_params(param_grid)
        mlflow.log_param("data_source", "fused_sensors_v1")
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        
        # --- EVALUATION ---
        y_pred = best_model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        
        # --- LOG AND SAVE ASSETS ---
        mlflow.sklearn.log_model(best_model, "sensor_fusion_rf_model")
        mlflow.sklearn.log_model(scaler, "standard_scaler")

        joblib.dump(best_model, os.path.join(MODEL_DIR, 'vanguard_classifier.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'vanguard_scaler.joblib'))
        
        print("MLflow run finished. Sensor fusion model saved.")

if __name__ == '__main__':
    run_training()