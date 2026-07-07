import pandas as pd
import numpy as np

def generate_intelligence_data(num_samples=200):
    """
    Generates synthetic multi-modal military intelligence data.
    Simulates Sensor Fusion: GEOINT, SIGINT, and Cyber logs.
    """
    np.random.seed(42)
    
    # 1. GEOINT: Coordinates centered around a simulated tactical area
    latitudes = np.random.uniform(34.0, 36.5, num_samples)
    longitudes = np.random.uniform(65.0, 68.5, num_samples)
    
    # 2. SIGINT: Radio frequency (MHz) and signal strength (dBm)
    signal_frequency = np.random.uniform(130.0, 950.0, num_samples)
    signal_strength = np.random.randint(-110, -30, num_samples) # Closer to 0 means stronger signal
    
    # 3. Cyber: Network packet sizes and intrusion attempts
    network_packets = np.random.randint(50, 8000, num_samples)
    login_attempts = np.random.randint(1, 15, num_samples)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'signal_frequency_mhz': signal_frequency,
        'signal_strength_dbm': signal_strength,
        'network_packets': network_packets,
        'login_attempts': login_attempts
    })
    
    # Define rules to create realistic threat labels based on sensor signatures
    threat_types = []
    for _, row in df.iterrows():
        if row['network_packets'] > 5000 and row['login_attempts'] > 8:
            threat_types.append('Cyber Infiltration')
        elif row['signal_strength_dbm'] > -50 and row['signal_frequency_mhz'] > 700:
            threat_types.append('Hostile Radar')
        elif row['signal_strength_dbm'] > -65 and row['network_packets'] > 3000:
            threat_types.append('Unidentified Drone')
        else:
            threat_types.append('Troop Movement')
            
    df['threat_type'] = threat_types
    return df

if __name__ == "__main__":
    # Test file execution
    test_df = generate_intelligence_data(5)
    print("Sample generated data:\n", test_df.head())