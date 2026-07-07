import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_pipeline import generate_intelligence_data

class ThreatModelEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.features = ['signal_frequency_mhz', 'signal_strength_dbm', 'network_packets', 'login_attempts']
        self.train_engine()
        
    def train_engine(self):
        """Trains the internal model on synthetic data."""
        df = generate_intelligence_data(500)
        X = df[self.features]
        y = df['threat_type']
        self.model.fit(X, y)
        
    def predict_threat(self, input_data: pd.DataFrame):
        """Predicts the threat type and calculates a dynamic risk score."""
        X_input = input_data[self.features]
        
        # Predict type and confidence probabilities
        prediction = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]
        max_prob = float(np.max(probabilities))
        
        # Calculate XAI: Extract top 2 feature contributions for this specific prediction
        # We simulate feature importance impact per instance for simple, quick metrics
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        top_features = [self.features[idx] for idx in sorted_indices[:2]]
        
        # Dynamic risk score scaled out of 100
        risk_score = int(max_prob * 100)
        
        return {
            "predicted_threat": prediction,
            "risk_score": risk_score,
            "top_drivers": top_features
        }

# Quick test check
if __name__ == "__main__":
    engine = ThreatModelEngine()
    sample_threat = pd.DataFrame([{
        'signal_frequency_mhz': 850.0,
        'signal_strength_dbm': -40,
        'network_packets': 6000,
        'login_attempts': 12
    }])
    print("Test Prediction Output:", engine.predict_threat(sample_threat))