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
    evaluates it, and logs everything with MLflow, including a model signature.
    """
    
    # MLflow deneyini ayarlayın
    mlflow.set_experiment("VANGUARD - Sensor Fusion Model")

    DATA_PATH = os.path.join("data", "vanguard_air_tracks_fused.csv")
    MODEL_DIR = "models"
    
    # MLflow run'u başlatın
    with mlflow.start_run(run_name="RandomForest_With_Signature"):
        print("MLflow run started...")
        
        # Veri setini yükleyin
        df = pd.read_csv(DATA_PATH)
        
        # Özellikleri ve hedefi belirleyin
        categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
        X = df.drop(['classification', 'latitude', 'longitude'], axis=1)
        y = df['classification']
        X_encoded = pd.get_dummies(X, columns=categorical_features)
        
        # Eğitimde kullanılan sütun adlarını kaydedin (veri tutarlılığı için)
        joblib.dump(X_encoded.columns, os.path.join(MODEL_DIR, 'training_columns.joblib'))
        
        # Veri setini eğitim ve test kümelerine ayırın
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verileri ölçeklendirin
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hiperparametre ızgarasını tanımlayın
        param_grid = {
            'n_estimators': [150],
            'max_depth': [20],
            'min_samples_split': [5]
        }
        
        # MLflow'a parametreleri kaydedin
        mlflow.log_params(param_grid)
        mlflow.log_param("data_source", "fused_sensors_v1")
        
        # GridSearchCV kullanarak modeli eğitin
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Model performansını değerlendirin
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # MLflow'a metrikleri kaydedin
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        
        # --- Önemli Düzeltme: Modeli ve ölçekleyiciyi MLflow'a kaydedin ---
        # Bu kısım, MLflow'un modeli takip etmesini sağlar.
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="random_forest_model", 
            input_example=input_example
        )
        mlflow.sklearn.log_model(scaler, "standard_scaler")

        # --- En Önemli Düzeltme: Modeli ve ölçekleyiciyi `models` klasörüne kaydedin ---
        # Bu kısım, Streamlit uygulamasının modelleri doğrudan bulmasını sağlar.
        joblib.dump(best_model, os.path.join(MODEL_DIR, 'vanguard_classifier.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'vanguard_scaler.joblib'))
        
        print("MLflow run finished. Model and scaler saved with a signature.")

if __name__ == '__main__':
    run_training()
