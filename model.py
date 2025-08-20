import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('vanguard_air_tracks.csv')

# --- Exploratory Data Analysis (EDA) ---
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nStatistical Summary:")
print(df.describe())

# Visualize the class distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='classification', order=df['classification'].value_counts().index)
plt.title('Distribution of Air Track Classifications')
plt.show()




# Define features (X) and target (y)
# We drop lat/lon because the model should not learn specific coordinates, but rather behavior.
X = df.drop(['classification', 'latitude', 'longitude'], axis=1)
y = df['classification']

# Perform One-Hot Encoding for categorical features
X_encoded = pd.get_dummies(X, columns=['electronic_signature', 'flight_profile'])

# Save the column order for later use in the app
TRAINING_COLUMNS = X_encoded.columns
joblib.dump(TRAINING_COLUMNS, 'training_columns.joblib')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Initialize and run GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

print("\nBest Model Parameters:", grid_search.best_params_)

# Evaluate the final model on the test set
y_pred = best_model.predict(X_test_scaled)
print("\nFinal Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save the final model and the scaler for our application
joblib.dump(best_model, 'vanguard_classifier.joblib')
joblib.dump(scaler, 'vanguard_scaler.joblib')
print("\nModel and scaler have been saved successfully.")