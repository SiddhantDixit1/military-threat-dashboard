import pandas as pd
import numpy as np
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TacticalModelEngine:
    def __init__(self, model_type='RandomForest'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = [
            'latitude', 'longitude', 'altitude', 'distance_to_base_km',
            'signal_strength_rssi', 'frequency_mhz', 'failed_logins', 
            'malware_score', 'drone_count', 'radar_cross_section', 
            'sensor_confidence', 'rolling_login_avg'
        ]
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        self.explainer = None

    def train_system(self, df):
        """Trains the core defense classifier and optimizes the SHAP Explainer."""
        X = df[self.features]
        y_type = self.label_encoder.fit_transform(df['threat_type'])
        
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y_type)
        
        # Initialize a tree explainer optimized for tree-based ensembles
        # We pass check_additivity=False to speed up calculations during live updates
        self.explainer = shap.TreeExplainer(self.clf)
        return "System Training & XAI Calibration Complete."

    @st.cache_data(show_spinner=False, ttl=600)
    def compute_cached_shap(_self, X_scaled_array):
        """Caches global SHAP values for performance over larger arrays."""
        return _self.explainer.shap_values(X_scaled_array)

    def predict_and_explain(self, single_row_df, historical_df_sample):
        """
        Executes inference and computes highly localized SHAP diagnostics 
        for positive and negative threat vectors.
        """
        X_live = single_row_df[self.features]
        X_live_scaled = self.scaler.transform(X_live)
        
        # Core Class Inference
        pred_encoded = self.clf.predict(X_live_scaled)[0]
        pred_prob = self.clf.predict_proba(X_live_scaled)[0]
        
        threat_class = self.label_encoder.inverse_transform([pred_encoded])[0]
        confidence = pred_prob[pred_encoded] * 100
        
        # Compute raw SHAP matrix safely
        raw_shap = self.explainer.shap_values(X_live_scaled)
        
        # Robust handling for list vs. 3D array output variants
        if isinstance(raw_shap, list):
            # If it's a list of arrays (one per class), select the correct class array
            class_shap_values = raw_shap[pred_encoded][0]
        elif len(raw_shap.shape) == 3:
            # If it's a 3D array with shape (samples, features, classes)
            class_shap_values = raw_shap[0, :, pred_encoded]
        else:
            # Fallback for standard 2D arrays
            class_shap_values = raw_shap[0] 
        
        # Build Diagnostic DataFrame
        diag_df = pd.DataFrame({
            'Feature': self.features,
            'Actual_Value': X_live.iloc[0].values,
            'SHAP_Impact': class_shap_values
        }).sort_values(by='SHAP_Impact', ascending=False)
        
        # Segregate Positive and Negative Drivers
        positive_drivers = diag_df[diag_df['SHAP_Impact'] > 0].to_dict(orient='records')
        negative_drivers = diag_df[diag_df['SHAP_Impact'] < 0].to_dict(orient='records')
        
        # Wrap everything into a single operational payload
        explanation_obj = shap.Explanation(
            values=class_shap_values,
            base_values=self.explainer.expected_value[pred_encoded],
            data=X_live.iloc[0].values,
            feature_names=self.features
        )
        
        return {
            'predicted_threat': threat_class,
            'confidence': confidence,
            'risk_score': int(confidence), # Maps to 0-100 gauge metric
            'top_10_features': diag_df.head(10),
            'positive_contributions': positive_drivers,
            'negative_contributions': negative_drivers,
            'explanation_object': explanation_obj,
            'raw_shap_all_classes': raw_shap,
            'predicted_class_index': pred_encoded
        }