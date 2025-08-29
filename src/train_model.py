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
    This function loads data, preprocesses it, trains a model using GridSearchCV,
    evaluates it, and logs everything with MLflow.
    """
    
    # --- MLFLOW SETUP ---
    # Set an experiment name to group your runs
    mlflow.set_experiment("VANGUARD Air Track Classification")

    # Define paths relative to the project root
    DATA_PATH = os.path.join("data", "vanguard_air_tracks.csv")
    MODEL_DIR = "models"
    
    # Start a new MLflow run
    with mlflow.start_run(run_name="RandomForest_Optimized"):
        print("MLflow run started...")
        
        # --- DATA LOADING AND PREPROCESSING ---
        df = pd.read_csv(DATA_PATH)
        
        X = df.drop(['classification', 'latitude', 'longitude'], axis=1)
        y = df['classification']
        X_encoded = pd.get_dummies(X, columns=['electronic_signature', 'flight_profile'])
        
        # Save training columns for the app to use
        joblib.dump(X_encoded.columns, os.path.join(MODEL_DIR, 'training_columns.joblib'))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- MODEL TRAINING with GridSearchCV ---
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Log parameters to MLflow
        mlflow.log_params(param_grid)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        
        # --- EVALUATION ---
        y_pred = best_model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        
        # --- LOG AND SAVE ASSETS ---
        # Log the scaler and the model using MLflow
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        mlflow.sklearn.log_model(scaler, "standard_scaler")

        # Save the final assets to the models directory for the app
        joblib.dump(best_model, os.path.join(MODEL_DIR, 'vanguard_classifier.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'vanguard_scaler.joblib'))
        
        print("MLflow run finished. Model and scaler saved.")

if __name__ == '__main__':
    # This block ensures the training only runs when the script is executed directly
    run_training()