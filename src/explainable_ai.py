# src/explainable_ai.py

"""
Explainable AI module using SHAP (SHapley Additive exPlanations)
Critical for defense applications - understand WHY the model makes decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VanguardExplainer:
    """
    Explainable AI wrapper for VANGUARD model
    
    Provides:
    - Feature importance analysis
    - Individual prediction explanations
    - Decision boundary visualization
    - Counterfactual analysis
    """
    
    def __init__(self, model, scaler, feature_cols, label_encoder):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.label_encoder = label_encoder
        
        # Set model to eval mode
        self.model.eval()
        
        logger.info("✅ Explainer initialized")
    
    def _predict_proba(self, X):
        """
        Prediction function for SHAP
        Returns probabilities for all classes
        """
        # X is numpy array from SHAP
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.numpy()
    
    def create_explainer(self, background_data, n_samples=100):
        """
        Create SHAP explainer
        
        Args:
            background_data: DataFrame with sample data for background distribution
            n_samples: Number of background samples to use
        """
        logger.info("Creating SHAP explainer...")
        
        # Prepare background data
        if len(background_data) > n_samples:
            background_sample = background_data.sample(n_samples, random_state=42)
        else:
            background_sample = background_data
        
        # One-hot encode
        categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
        background_encoded = pd.get_dummies(background_sample, columns=categorical_features)
        
        # Align columns
        for col in self.feature_cols:
            if col not in background_encoded.columns:
                background_encoded[col] = 0
        
        background_encoded = background_encoded[self.feature_cols]
        
        # Scale
        background_scaled = self.scaler.transform(background_encoded.values.astype(float))
        
        # Create explainer
        self.explainer = shap.KernelExplainer(
            self._predict_proba,
            background_scaled
        )
        
        logger.info("✅ SHAP explainer created")
        
        return self.explainer
    
    def explain_prediction(self, input_data, class_names=None):
        """
        Explain a single prediction
        
        Args:
            input_data: dict with aircraft features
            class_names: list of class names (optional)
        
        Returns:
            dict with explanation
        """
        if not hasattr(self, 'explainer'):
            raise ValueError("Explainer not created! Call create_explainer() first")
        
        # Prepare input
        df = pd.DataFrame([input_data])
        
        categorical_features = ['electronic_signature', 'flight_profile', 'weather', 'thermal_signature']
        df_encoded = pd.get_dummies(df, columns=categorical_features)
        
        for col in self.feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        df_encoded = df_encoded[self.feature_cols]
        X_scaled = self.scaler.transform(df_encoded.values.astype(float))
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Get prediction
        probs = self._predict_proba(X_scaled)[0]
        predicted_class_idx = np.argmax(probs)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        
        # Handle different SHAP output formats (list of arrays vs single array)
        if isinstance(shap_values, list):
            # Classic format: list of [num_samples, num_features]
            class_shap_values = shap_values[predicted_class_idx][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # New format: [num_samples, num_features, num_outputs]
            class_shap_values = shap_values[0, :, predicted_class_idx]
        else:
            # Fallback or binary case
            class_shap_values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Create feature importance dict
        feature_importance = {}
        for feat, shap_val in zip(self.feature_cols, class_shap_values):
            if abs(shap_val) > 0.0001:  # Lower threshold for completeness
                feature_importance[feat] = float(shap_val)
        
        # Sort by absolute value
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(probs[predicted_class_idx]),
            'all_probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.label_encoder.classes_, probs)
            },
            'feature_importance': dict(sorted_features[:10]),  # Top 10
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value
        }
    
    def plot_waterfall(self, explanation, save_path='models/explanation_waterfall.png'):
        """
        Create waterfall plot showing feature contributions
        """
        predicted_class = explanation['predicted_class']
        predicted_idx = list(self.label_encoder.classes_).index(predicted_class)
        shap_values = explanation['shap_values']
        
        # Handle different SHAP formats
        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_idx][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values_class = shap_values[0, :, predicted_idx]
        else:
            shap_values_class = shap_values[0]
            
        # Create explanation object for waterfall
        exp = shap.Explanation(
            values=shap_values_class,
            base_values=explanation['base_value'][predicted_idx] if isinstance(explanation['base_value'], (list, np.ndarray)) else explanation['base_value'],
            data=np.zeros(len(self.feature_cols)),
            feature_names=self.feature_cols
        )
        
        # Plot
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(exp, show=False)
        plt.title(f"Why {predicted_class}? - Feature Contributions")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Waterfall plot saved: {save_path}")
        
        return plt.gcf()
    
    def plot_force(self, explanation, save_path='models/explanation_force.png'):
        """
        Create force plot showing push/pull of features
        """
        predicted_class = explanation['predicted_class']
        predicted_idx = list(self.label_encoder.classes_).index(predicted_class)
        shap_values = explanation['shap_values']
        
        # Handle different SHAP formats
        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_idx][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values_class = shap_values[0, :, predicted_idx]
        else:
            shap_values_class = shap_values[0]
            
        # Create force plot
        shap.force_plot(
            explanation['base_value'][predicted_idx] if isinstance(explanation['base_value'], (list, np.ndarray)) else explanation['base_value'],
            shap_values_class,
            feature_names=self.feature_cols,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Force plot saved: {save_path}")
        
        return plt.gcf()
    
    def get_human_readable_explanation(self, explanation):
        """
        Convert SHAP values to human-readable explanation
        
        Returns natural language explanation
        """
        predicted_class = explanation['predicted_class']
        confidence = explanation['confidence']
        features = explanation['feature_importance']
        
        # Start explanation
        text = f"## Classification: {predicted_class} ({confidence:.1%} confidence)\n\n"
        text += "### Key Decision Factors:\n\n"
        
        # Positive contributors (increase probability)
        positive = [(k, v) for k, v in features.items() if v > 0]
        if positive:
            text += "**Factors supporting this classification:**\n"
            for feat, val in positive[:5]:
                # Clean feature name
                clean_feat = feat.replace('_', ' ').title()
                text += f"- {clean_feat}: +{abs(val):.3f}\n"
            text += "\n"
        
        # Negative contributors (decrease probability)
        negative = [(k, v) for k, v in features.items() if v < 0]
        if negative:
            text += "**Factors against this classification:**\n"
            for feat, val in negative[:5]:
                clean_feat = feat.replace('_', ' ').title()
                text += f"- {clean_feat}: -{abs(val):.3f}\n"
            text += "\n"
        
        # Alternative classifications
        all_probs = explanation['all_probabilities']
        sorted_classes = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_classes) > 1:
            text += "### Alternative Classifications:\n"
            for cls, prob in sorted_classes[1:4]:  # Top 3 alternatives
                text += f"- {cls}: {prob:.1%}\n"
        
        return text
    
    def analyze_counterfactual(self, input_data, target_class):
        """
        Find what needs to change to get different classification
        
        Args:
            input_data: Current input dict
            target_class: Desired classification
        
        Returns:
            Suggested changes
        """
        # Get current prediction
        current_exp = self.explain_prediction(input_data)
        current_class = current_exp['predicted_class']
        
        if current_class == target_class:
            return "Already classified as target class!"
        
        # Simple counterfactual analysis
        suggestions = []
        
        # Check feature importance for target vs current
        target_idx = list(self.label_encoder.classes_).index(target_class)
        
        suggestions.append(f"To change classification from {current_class} to {target_class}:")
        
        # Analyze key differentiating features
        # This is simplified - real counterfactual would be more sophisticated
        if target_class == 'CIVILIAN':
            suggestions.append("- Increase altitude above 28,000 ft")
            suggestions.append("- Set IFF to Mode 3C")
            suggestions.append("- Maintain stable cruise profile")
        
        elif target_class == 'HOSTILE':
            suggestions.append("- Decrease altitude below 10,000 ft")
            suggestions.append("- Increase speed above 500 knots")
            suggestions.append("- Remove IFF response")
        
        elif target_class == 'FRIEND':
            suggestions.append("- Set IFF to Mode 5")
            suggestions.append("- Maintain moderate altitude (15,000-30,000 ft)")
        
        return "\n".join(suggestions)


# ==================== UTILITY FUNCTIONS ====================

def load_explainer(model_path='models/best_model.pt'):
    """
    Load model and create explainer
    """
    # model_pytorch_improved modülünüzün nerede olduğuna göre bu içe aktarma yolu değişebilir.
    # Eğer bu satır hata verirse (ImportError), lütfen "from src.model_pytorch_improved import ImprovedAircraftClassifier" satırını doğru yolla güncelleyin.
    from src.model_pytorch_improved import ImprovedAircraftClassifier 
    
    # Load preprocessing
    scaler = joblib.load('models/scaler.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    feature_cols = joblib.load('models/feature_columns.joblib')
    
    # Load model
    input_dim = len(feature_cols)
    num_classes = len(label_encoder.classes_)
    
    model = ImprovedAircraftClassifier(input_dim, num_classes)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create explainer
    explainer = VanguardExplainer(model, scaler, feature_cols, label_encoder)
    
    return explainer


def generate_explanation_report(input_data, save_dir='models/explanations'):
    """
    Generate complete explanation report
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load explainer
    explainer = load_explainer()
    
    # Load background data
    background_df = pd.read_csv('data/processed/cleaned_training_data.csv')
    explainer.create_explainer(background_df, n_samples=50)
    
    # Get explanation
    explanation = explainer.explain_prediction(input_data)
    
    # Generate plots
    explainer.plot_waterfall(explanation, f'{save_dir}/waterfall.png')
    explainer.plot_force(explanation, f'{save_dir}/force.png')
    
    # Get text explanation
    text_exp = explainer.get_human_readable_explanation(explanation)
    
    # Save report
    with open(f'{save_dir}/explanation_report.md', 'w') as f:
        f.write(text_exp)
    
    logger.info(f"✅ Explanation report generated in {save_dir}/")
    
    return explanation, text_exp


# ==================== DEMO ====================

if __name__ == "__main__":
    print("🔍 VANGUARD AI - Explainable AI Demo")
    print("="*70)
    
    # Example input
    test_input = {
        'altitude_ft': 8000,  # Low altitude
        'speed_kts': 580,     # High speed
        'rcs_m2': 2.5,        # Small RCS
        'latitude': 51.5,
        'longitude': -0.1,
        'heading': 270,
        'electronic_signature': 'NO_IFF_RESPONSE',  # No IFF!
        'flight_profile': 'AGGRESSIVE_MANEUVERS',    # Aggressive
        'weather': 'Clear',
        'thermal_signature': 'High'
    }
    
    print("\nTest Input:")
    print(f"  Altitude: {test_input['altitude_ft']:,} ft (LOW)")
    print(f"  Speed: {test_input['speed_kts']} kts (FAST)")
    print(f"  IFF: {test_input['electronic_signature']} (NONE)")
    print(f"  Profile: {test_input['flight_profile']} (AGGRESSIVE)")
    
    print("\n" + "="*70)
    print("Generating explanation...")
    print("="*70)
    
    # Generate report
    explanation, text = generate_explanation_report(test_input)
    
    print("\n" + text)
    
    print("\n" + "="*70)
    print("✅ Explanation complete!")
    print("   Check models/explanations/ for detailed reports")
    print("="*70)