import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ThreatDataSimulator:
    def __init__(self, n_samples=25000, seed=42):
        self.n_samples = n_samples
        np.random.seed(seed)
        self.base_lat = 20.5937  # Operational field coordinates
        self.base_lon = 78.9629
        self.start_time = datetime.now() - timedelta(days=5)

    def _generate_base_features(self):
        n = self.n_samples
        lat = np.random.normal(self.base_lat, 0.5, n)
        lon = np.random.normal(self.base_lon, 0.5, n)
        alt = np.abs(np.random.normal(500, 2000, n))
        dist_base = np.sqrt((lat - self.base_lat)**2 + (lon - self.base_lon)**2) * 111
        
        rssi = np.random.uniform(10, 100, n)
        freq = np.random.uniform(30, 3000, n)
        failed_logins = np.random.poisson(lam=1, size=n)
        malware_score = np.random.uniform(0, 100, n)
        drone_count = np.random.poisson(lam=0.5, size=n)
        radar_cross_section = np.random.uniform(0.1, 50, n)
        
        timestamps = [self.start_time + timedelta(seconds=i*10) for i in range(n)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lat, 'longitude': lon, 'altitude': alt,
            'distance_to_base_km': dist_base,
            'signal_strength_rssi': rssi, 'frequency_mhz': freq,
            'failed_logins': failed_logins, 'malware_score': malware_score,
            'drone_count': drone_count, 'radar_cross_section': radar_cross_section
        })

    def _apply_tactical_rules(self, df):
        df['threat_type'] = 'Safe'
        df['threat_severity'] = 'Low'
        df['risk_score'] = np.random.uniform(0, 20, self.n_samples)

        # Drone rule mapping
        mask_drone = (df['drone_count'] > 3) & (df['signal_strength_rssi'] > 75) & (df['altitude'] < 3000)
        df.loc[mask_drone, 'threat_type'] = 'Hostile Drone'
        df.loc[mask_drone, 'threat_severity'] = 'High'
        df.loc[mask_drone, 'risk_score'] = np.random.uniform(75, 95, mask_drone.sum())

        # Cyber attack mapping
        mask_cyber = (df['failed_logins'] > 5) & (df['malware_score'] > 80)
        df.loc[mask_cyber, 'threat_type'] = 'Cyber Intrusion'
        df.loc[mask_cyber, 'threat_severity'] = 'Critical'
        df.loc[mask_cyber, 'risk_score'] = np.random.uniform(90, 100, mask_cyber.sum())

        # Radar trace mapping
        mask_radar = (df['radar_cross_section'] > 30) & (df['distance_to_base_km'] < 50) & (df['altitude'] > 10000)
        df.loc[mask_radar, 'threat_type'] = 'Hostile Radar'
        df.loc[mask_radar, 'threat_severity'] = 'High'
        df.loc[mask_radar, 'risk_score'] = np.random.uniform(70, 89, mask_radar.sum())

        return df

    def _feature_engineering(self, df):
        df['sensor_confidence'] = (100 - (df['distance_to_base_km'] / df['distance_to_base_km'].max()) * 100)
        df = df.sort_values('timestamp')
        df['rolling_login_avg'] = df['failed_logins'].rolling(window=10, min_periods=1).mean()
        return df

    def generate_pipeline(self):
        df = self._generate_base_features()
        df = self._apply_tactical_rules(df)
        df = self._feature_engineering(df)
        return df.fillna(0)