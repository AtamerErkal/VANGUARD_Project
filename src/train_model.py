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
    This function loads sensor fusion data, preprocesses it,
    trains a RandomForest model using GridSearchCV,
    evaluates its performance, and logs the entire workflow using MLflow.
    It also saves the model and scaler locally for downstream use.
    """

    # Set up MLflow experiment tracking
    mlflow.set_experiment("VANGUARD - Sensor Fusion Model")

    # Define paths for input data and output models
    DATA_PATH = os.path.join("data", "vanguard_air_tracks_fused.csv")
    MODEL_DIR = "models"

    # Start an MLflow run
    with mlflow.start_run(run_name="RandomForest_With_Signature"):
        print("MLflow run started...")

        # Load the dataset
        df = pd.read_csv(DATA_PATH)

        # Define categorical features and separate target variable
        categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
        X = df.drop(['classification', 'latitude', 'longitude'], axis=1)
        y = df['classification']

        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, columns=categorical_features)

        # Save feature names for future consistency (e.g., in inference)
        joblib.dump(X_encoded.columns, os.path.join(MODEL_DIR, 'training_columns.joblib'))

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define hyperparameter grid for RandomForest
        param_grid = {
            'n_estimators': [150],
            'max_depth': [20],
            'min_samples_split': [5]
        }

        # Log parameters to MLflow
        mlflow.log_params(param_grid)
        mlflow.log_param("data_source", "fused_sensors_v1")

        # Train model using GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        # Extract the best model
        best_model = grid_search.best_estimator_

        # Evaluate model performance
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)

        # Log model and scaler to MLflow with input example
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_model",
            input_example=input_example
        )
        mlflow.sklearn.log_model(scaler, "standard_scaler")

        # Save model and scaler locally for use in Streamlit or other apps
        joblib.dump(best_model, os.path.join(MODEL_DIR, 'vanguard_classifier.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'vanguard_scaler.joblib'))

        print("MLflow run finished. Model and scaler saved with a signature.")

if __name__ == '__main__':
    run_training()