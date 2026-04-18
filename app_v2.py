# app.py

"""
VANGUARD AI - Complete Streamlit Application
Production-ready interface for aircraft classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys

# Yolu manuel olarak ekleme (Gerekli)
if 'src' not in sys.path:
    # Bu, app.py'nin bulunduğu kök dizini sys.path'e ekler
    sys.path.append(str(Path(__file__).parent)) 
    
# Gerekli importlar
from src.explainable_ai import VanguardExplainer
# Modül adınızı kontrol edin: 'from src.model_pytorch_improved import ImprovedAircraftClassifier'
from src.model_pytorch_improved import ImprovedAircraftClassifier 

# ==================== CACHED FUNCTIONS ====================

# Modeli ve gerekli nesneleri sadece bir kez yükler
@st.cache_resource
def load_vanguard_assets():
    with st.spinner("Loading core assets (model, scaler)..."):
        try:
            scaler = joblib.load('models/scaler.joblib')
            label_encoder = joblib.load('models/label_encoder.joblib')
            feature_cols = joblib.load('models/feature_columns.joblib')
            
            # Model yükleme
            input_dim = len(feature_cols)
            num_classes = len(label_encoder.classes_)
            model = ImprovedAircraftClassifier(input_dim, num_classes)
            
            checkpoint = torch.load('models/best_model.pt', map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            return model, scaler, feature_cols, label_encoder
        except Exception as e:
             st.error(f"Error loading model assets. Is 'models/best_model.pt' present? Error: {e}")
             return None, None, None, None

# Explainer nesnesini (SHAP) sadece bir kez başlatır
@st.cache_resource
def create_vanguard_explainer(model, scaler, feature_cols, label_encoder):
    if model is None:
        return None
        
    with st.spinner("Initializing SHAP Explainer (this may take a moment)..."):
        try:
            explainer = VanguardExplainer(model, scaler, feature_cols, label_encoder)
            
            # Arka plan verisini yükle ve explainer'ı oluştur
            background_df = pd.read_csv('data/processed/cleaned_training_data.csv')
            explainer.create_explainer(background_df, n_samples=50) 
            
            return explainer
        except Exception as e:
            st.error(f"Error creating SHAP explainer: {e}")
            return None

# ==================== PAGE CONFIG ====================\

st.set_page_config(
    page_title="VANGUARD AI Defense System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ASSET LOADING ====================

# Yüklenen varlıkları al
model, scaler, feature_cols, label_encoder = load_vanguard_assets()
explainer_instance = create_vanguard_explainer(model, scaler, feature_cols, label_encoder)

if model is None or explainer_instance is None:
    st.error("Application failed to load core components. Check console output for detailed errors.")
    st.stop() # Eğer yükleme başarısızsa uygulamayı durdur

# ==================== CUSTOM CSS ====================\

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #2a5298;
    }
    .stSpinner {
        color: #2a5298 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== PREDICTION FUNCTIONS ====================\

def predict(data, model, scaler, feature_cols, label_encoder):
    """Predicts the class of the aircraft based on input data."""
    
    df = pd.DataFrame([data])
    
    # One-hot encode categorical features and align columns (must match training data prep)
    categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[feature_cols]
    
    X_scaled = scaler.transform(df_encoded.values.astype(float))
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
    
    predicted_class_idx = torch.argmax(probs).item()
    confidence = probs[0, predicted_class_idx].item()
    predicted_class = label_encoder.classes_[predicted_class_idx]
    
    all_probs = {
        cls: prob.item() for cls, prob in zip(label_encoder.classes_, probs[0])
    }
    
    return predicted_class, confidence, all_probs

# ==================== UTILITY FUNCTIONS ====================\

def get_status_color(prediction):
    if prediction == 'HOSTILE':
        return '#CC0000' # Red
    elif prediction == 'CIVILIAN':
        return '#00CC66' # Green
    elif prediction == 'FRIEND':
        return '#0099CC' # Blue
    elif prediction == 'UNKNOWN':
        return '#FFCC00' # Yellow
    else:
        return '#AAAAAA' # Gray

# ==================== MAIN APP LAYOUT ====================\

def main_app():
    
    st.markdown("<h1 class='main-header'>🛡️ VANGUARD AI: AIR DEFENSE SYSTEM</h1>", unsafe_allow_html=True)
    
    st.subheader("Simulated Sensor Input")
    
    # --- Sidebar Input ---
    with st.sidebar:
        st.header("Aircraft Data Input")
        
        # Numeric Inputs
        altitude = st.slider("Altitude (ft)", 0, 60000, 25000)
        speed = st.slider("Speed (knots)", 0, 1000, 450)
        rcs = st.slider("Radar Cross Section (m²)", 0.1, 10.0, 1.5, 0.1)
        heading = st.slider("Heading (degrees)", 0, 360, 270)
        
        # Categorical Inputs
        electronic_signature = st.selectbox(
            "Electronic Signature (IFF)",
            ['MODE_5', 'MODE_3C', 'NO_IFF_RESPONSE', 'ANOMALOUS']
        )
        flight_profile = st.selectbox(
            "Flight Profile",
            ['STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'UNPREDICTABLE']
        )
        weather = st.selectbox(
            "Weather Conditions",
            ['Clear', 'Heavy_Rain', 'Snow']
        )
        thermal_signature = st.selectbox(
            "Thermal Signature",
            ['Low', 'Medium', 'High', 'Extremely_High']
        )
        
        # Log location (static for simplicity, can be expanded)
        latitude = 34.0522
        longitude = -118.2437
        
        st.markdown("---")
        st.info("Inputs are used for real-time classification and explainability.")


    # --- Prediction ---
    input_data = {
        'altitude_ft': altitude,
        'speed_kts': speed,
        'rcs_m2': rcs,
        'latitude': latitude,
        'longitude': longitude,
        'heading': heading,
        'electronic_signature': electronic_signature,
        'flight_profile': flight_profile,
        'weather': weather,
        'thermal_signature': thermal_signature
    }
    
    predicted_class, confidence, all_probs = predict(input_data, model, scaler, feature_cols, label_encoder)
    
    # --- Metrics ---
    st.markdown("### Real-Time Classification")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {get_status_color(predicted_class)};">
            <small>CLASSIFICATION</small>
            <h2>{predicted_class}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <small>CONFIDENCE</small>
            <h2>{confidence:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Probability Plot
        prob_df = pd.DataFrame(all_probs.items(), columns=['Class', 'Probability'])
        prob_fig = px.bar(
            prob_df.sort_values('Probability', ascending=False),
            x='Probability',
            y='Class',
            orientation='h',
            title='Class Probabilities',
            color='Class',
            color_discrete_map={'HOSTILE': '#CC0000', 'CIVILIAN': '#00CC66', 'FRIEND': '#0099CC', 'UNKNOWN': '#FFCC00', 'NEUTRAL': '#FF9900', 'CRITICAL': '#800080'},
            height=200
        )
        prob_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    
    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["🗺️ Radar View (Mock)", "🔍 Explainable AI", "📝 Scenario Guide"])
    
    with tab1:
        st.header("Situational Awareness")
        st.info("A full-featured radar simulation requires complex geospatial libraries. This is a mock display.")
        
        # Simple map visualization
        map_data = pd.DataFrame({
            'lat': [latitude], 
            'lon': [longitude],
            'size': [confidence * 100],
            'color': [get_status_color(predicted_class)]
        })
        
        st.map(map_data, latitude=latitude, longitude=longitude, zoom=10, size='size', color='color')
        
        st.subheader("Detailed Sensor Readings")
        st.json(input_data)
        
    with tab2:
        st.header("Decision Rationale (XAI)")
        
        with st.expander("🔍 Explainable AI - Why this classification?", expanded=True):
            st.markdown("### Understanding the Decision")
            
            try:
                # Önceden önbelleğe alınan explainer'ı kullanın
                with st.spinner("Generating explanation..."):
                    # explainer_instance'ı kullanın
                    explanation = explainer_instance.explain_prediction(input_data)
                
                # 1. Human-readable explanation
                text_exp = explainer_instance.get_human_readable_explanation(explanation)
                st.markdown(text_exp)
                
                # 2. Plotting
                st.subheader("Feature Contribution Waterfall")
                st.pyplot(explainer_instance.plot_waterfall(explanation, save_path=None))

                st.subheader("Force Plot Visualization")
                st.pyplot(explainer_instance.plot_force(explanation, save_path=None))

                # 3. Counterfactual
                st.subheader("Counterfactual Analysis")
                target_class = st.selectbox("Check what would be needed to classify as:", 
                                            label_encoder.classes_, 
                                            index=list(label_encoder.classes_).index('CIVILIAN'))
                
                counterfactual_text = explainer_instance.analyze_counterfactual(input_data, target_class)
                st.info(counterfactual_text)


            except Exception as e:
                # Eğer buraya düşerse, SHAP problemi kalıcı olarak çözülmemiş demektir.
                st.warning(f"Explainability feature unavailable: {e}")
                st.info("The explainer failed during prediction/plotting. Please check Python console for errors.")


    with tab3:
        st.header("Standard Operating Procedures")
        
        with st.expander("Example Scenarios"):
            st.markdown("""
            **Scenario 1: Commercial Aircraft**
            - Altitude: 35,000 ft
            - Speed: 450 knots
            - IFF: Mode 3C (Civilian)
            - Profile: Stable Cruise
            - Expected: **CIVILIAN**
            
            **Scenario 2: Hostile Threat**
            - Altitude: 8,000 ft (low!)
            - Speed: 600 knots (fast!)
            - IFF: No Response
            - Profile: Aggressive Maneuvers
            - Expected: **HOSTILE**
            
            **Scenario 3: Friendly Military**
            - Altitude: 25,000 ft
            - Speed: 400 knots
            - IFF: Mode 5 (Military)
            - Profile: Stable Cruise
            - Expected: **FRIEND**
            """)
    
    # Footer
    st.markdown("---")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        **🔧 System Info**
        - Version: 2.1 (XAI Fixed)
        - Model: PyTorch
        - Framework: Streamlit
        """)
    
    with col_f2:
        st.markdown("""
        **📈 Performance**
        - Accuracy: 87%+
        - Latency: <200ms
        - Classes: 6
        """)

if __name__ == "__main__":
    main_app()