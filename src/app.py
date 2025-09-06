# src/app.py

import streamlit as st
import pandas as pd
import joblib
import os
import pydeck as pdk # We will use pydeck for advanced map visualization

# --- PAGE CONFIGURATION ---
# Set a page title, icon, and layout for a more professional look
st.set_page_config(
    page_title="VANGUARD AI System",
    page_icon="✈️",
    layout="wide"
)

# --- LOAD ASSETS ---
# Define the path to the models directory
MODEL_DIR = os.path.join("..", "models")

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

model, scaler, training_columns = load_assets()

# --- UI: HEADER AND DESCRIPTION ---
st.title("✈️ VANGUARD: AI Air Track Identification System")
st.info("""
**Welcome to the VANGUARD project.** This application serves as an interactive demonstration of a machine learning model designed to classify air tracks in a simulated operational environment. 

**What it does:** The system uses a Random Forest model trained on synthetic data simulating two types of sensors: a **Radar** (providing kinematic data like speed, altitude, and RCS) and an **EO/IR camera** (providing thermal signatures). It performs sensor fusion to classify a track as `FRIEND`, `HOSTILE`, `CIVILIAN`, etc.

**Purpose:** This project showcases an end-to-end MLOps workflow, including data simulation, experiment tracking with MLflow, and deployment as an interactive web app.
""")

# --- UI: LAYOUT AND INPUTS ---
col1, col2 = st.columns((1, 1.5))

with col1:
    st.header("Track Input Data")

    st.subheader("📍 Location")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=51.5, format="%.4f", help="The geographic latitude of the track.")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=12.5, format="%.4f", help="The geographic longitude of the track.")
    
    st.subheader("📡 Radar Signature")
    altitude = st.slider("Altitude (feet)", 0, 65000, 30000, help="The aircraft's flight altitude in feet above sea level.")
    speed = st.slider("Speed (knots)", 0, 2000, 450, help="The aircraft's speed over ground in knots.")
    rcs = st.slider("Radar Cross Section (RCS, m²)", 0.0, 100.0, 15.0, help="A measure of how detectable an object is by radar. Small values indicate stealth.")
    
    st.subheader("📷 Fused Sensor Data")
    weather = st.selectbox("Weather Condition", ('Clear', 'Cloudy', 'Rainy'), help="Current weather, which affects EO/IR sensor performance.")
    thermal_signature = st.selectbox("EO/IR Thermal Signature", ('Not_Detected', 'Low', 'Medium', 'High'), help="The heat signature detected by an EO/IR sensor. This sensor may fail in bad weather.")
    
    st.subheader("📡 Electronic & Flight Profile")
    electronic_signature = st.selectbox(
        "Electronic Signature",
        ('IFF_MODE_5', 'IFF_MODE_3C', 'NO_IFF_RESPONSE', 'HOSTILE_JAMMING', 'UNKNOWN_EMISSION'),
        help="IFF Mode 5: Secure military friend signal.\n\nIFF Mode 3C: Standard civilian/military transponder.\n\nHostile Jamming: Electronic warfare signal.\n\nNo Response: Not responding to IFF."
    )
    flight_profile = st.selectbox(
        "Flight Profile",
        ('STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING', 'CLIMBING'),
        help="The flight pattern of the track. Aggressive maneuvers might indicate a threat."
    )
    
    input_data = {
        'altitude_ft': altitude, 'speed_kts': speed, 'rcs_m2': rcs,
        'electronic_signature': electronic_signature, 'flight_profile': flight_profile,
        'weather': weather, 'thermal_signature': thermal_signature
    }
    input_df = pd.DataFrame([input_data])

# --- UI: ICON DATA FOR THE MAP ---
ICON_URL = "https://raw.githubusercontent.com/ajduberstein/ok-legislature-tracker/master/static/plane-icon.png"
icon_data = {
    "url": ICON_URL,
    "width": 242,
    "height": 242,
    "anchorY": 242,
}

# --- PROCESSING, PREDICTION, and RESULTS ---
if st.button("CLASSIFY TRACK", type="primary", use_container_width=True):
    # Prepare data
    input_df_encoded = pd.get_dummies(input_df)
    input_df_processed = input_df_encoded.reindex(columns=training_columns, fill_value=0)
    input_df_scaled = scaler.transform(input_df_processed)
    
    # Prediction
    prediction = model.predict(input_df_scaled)[0]
    prediction_proba = model.predict_proba(input_df_scaled)
    
    with col2:
        st.header("Classification Result")
        CLASSIFICATION_STYLE = {
            'HOSTILE': {"icon": '💣', "type": 'error'}, 'SUSPECT': {"icon": '❓', "type": 'warning'},
            'FRIEND': {"icon": '🛡️', "type": 'success'}, 'ASSUMED FRIEND': {"icon": '🤝', "type": 'success'},
            'NEUTRAL': {"icon": '🏳️', "type": 'info'}, 'CIVILIAN': {"icon": '✈️', "type": 'info'},
        }
        style = CLASSIFICATION_STYLE.get(prediction, {"icon": '❔', "type": 'secondary'})
        getattr(st, style['type'])(f"**Classification: {prediction} {style['icon']}**")
        
        st.subheader("Confidence Scores")
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_).T
        proba_df.columns = ["Probability"]
        st.bar_chart(proba_df.sort_values("Probability", ascending=False))
        
        st.subheader("Track Location")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        map_df['icon_data'] = [icon_data] # Add icon data to the dataframe
        
        # Use IconLayer in pydeck
        layer = pdk.Layer(
            "IconLayer",
            data=map_df,
            get_icon="icon_data",
            get_size=4,
            size_scale=15,
            get_position=["lon", "lat"],
        )
        view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=6, pitch=45)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{classification}"}))
else:
    # Display the initial map view before classification
    with col2:
        st.info("Awaiting input data for classification.")
        st.subheader("Track Location")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        map_df['icon_data'] = [icon_data]
        
        layer = pdk.Layer(
            "IconLayer",
            data=map_df,
            get_icon="icon_data",
            get_size=4,
            size_scale=15,
            get_position=["lon", "lat"],
        )
        view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=6, pitch=45)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))