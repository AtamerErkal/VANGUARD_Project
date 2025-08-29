# src/app.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="VANGUARD System", layout="wide")

# Define path to the models directory
MODEL_DIR = os.path.join("..", "models") # Go up one level from src, then into models

# --- 1. LOAD PRE-TRAINED ASSETS ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, scaler, and training columns from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, 'vanguard_classifier.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'vanguard_scaler.joblib')
        columns_path = os.path.join(MODEL_DIR, 'training_columns.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        columns = joblib.load(columns_path)
        return model, scaler, columns
    except FileNotFoundError:
        st.error("Model assets not found! Please run the training script first: `python src/train_model.py`", icon="🚨")
        st.stop()

# (The rest of the app.py code remains the same as the previous correct version)
# I will include the full code for clarity.

model, scaler, training_columns = load_assets()

# --- 2. SETUP THE USER INTERFACE (UI) ---
st.title("VANGUARD: AI Air Track Identification System")
st.markdown("Enter track data to get an AI-powered classification and see its location on the map.")

col1, col2 = st.columns((1, 1.5))

# --- 3. GET USER INPUT ---
with col1:
    st.header("Track Input Data")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=51.5, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=12.5, format="%.4f")
    altitude = st.slider("Altitude (feet)", 0, 65000, 30000)
    speed = st.slider("Speed (knots)", 0, 2000, 450)
    rcs = st.slider("Radar Cross Section (RCS, m²)", 0.0, 100.0, 15.0)
    electronic_signature = st.selectbox(
        "Electronic Signature",
        ('IFF_MODE_5', 'IFF_MODE_3C', 'NO_IFF_RESPONSE', 'HOSTILE_JAMMING', 'UNKNOWN_EMISSION')
    )
    flight_profile = st.selectbox(
        "Flight Profile",
        ('STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING', 'CLIMBING')
    )
    input_data = {
        'altitude_ft': altitude, 'speed_kts': speed, 'rcs_m2': rcs,
        'electronic_signature': electronic_signature, 'flight_profile': flight_profile
    }
    input_df = pd.DataFrame([input_data])

# --- 4. PROCESS INPUT AND MAKE PREDICTION ---
if st.button("CLASSIFY TRACK", type="primary", use_container_width=True):
    input_df_encoded = pd.get_dummies(input_df)
    input_df_processed = input_df_encoded.reindex(columns=training_columns, fill_value=0)
    input_df_scaled = scaler.transform(input_df_processed)
    prediction = model.predict(input_df_scaled)[0]
    prediction_proba = model.predict_proba(input_df_scaled)
    with col2:
        st.header("Classification Result")
        CLASSIFICATION_STYLE = {
            'HOSTILE': {"icon": '💣', "type": 'error'}, 'SUSPECT': {"icon": '❓', "type": 'warning'},
            'FRIEND': {"icon": '🛡️', "type": 'success'}, 'ASSUMED FRIEND': {"icon": '🤝', "type": 'success'},
            'NEUTRAL': {"icon": '🏳️', "type": 'info'}, 'CIVILIAN': {"icon": '✈️', "type": 'info'},
            'UNKNOWN': {"icon": '❔', "type": 'secondary'}
        }
        style = CLASSIFICATION_STYLE.get(prediction)
        getattr(st, style['type'])(f"**Classification: {prediction} {style['icon']}**")
        st.subheader("Confidence Scores")
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_).T
        proba_df.columns = ["Probability"]
        st.bar_chart(proba_df.sort_values("Probability", ascending=False))
        st.subheader("Track Location")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=5)
else:
    with col2:
        st.info("Awaiting input data for classification.")
        st.subheader("Track Location")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=5)