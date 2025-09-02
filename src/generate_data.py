# src/data_generator.py

import pandas as pd
import numpy as np
import os

def generate_data(n_samples=2500):
    """
    Generates a more realistic synthetic dataset simulating two sensor types:
    1. Radar: Provides kinematic data (altitude, speed, rcs).
    2. EO/IR Camera: Provides a visual/thermal signature, affected by weather.
    """
    
    data = []
    
    ZONE_BOUNDARIES = {
        "FRIENDLY_AIRSPACE": {"lat": (48, 54), "lon": (8, 15)},
        "HOSTILE_AIRSPACE": {"lat": (44, 47), "lon": (33, 40)},
        "NEUTRAL_TERRITORY": {"lat": (46, 47.5), "lon": (6, 10.5)},
        "CIVIL_AIR_CORRIDOR": {"lat": (42, 52), "lon": (0, 30)},
        "CONTESTED_ZONE": {"lat": (54, 58), "lon": (18, 25)}
    }

    def get_random_coords(zone):
        bounds = ZONE_BOUNDARIES[zone]
        lat = np.random.uniform(bounds["lat"][0], bounds["lat"][1])
        lon = np.random.uniform(bounds["lon"][0], bounds["lon"][1])
        return lat, lon

    for _ in range(n_samples):
        class_choice = np.random.choice(
            ['FRIEND', 'HOSTILE', 'CIVILIAN', 'SUSPECT', 'NEUTRAL'],
            p=[0.20, 0.15, 0.40, 0.20, 0.05]
        )

        # --- NEW: Simulate weather conditions ---
        # Weather will affect the EO/IR sensor's ability to see the target.
        weather = np.random.choice(['Clear', 'Cloudy', 'Rainy'], p=[0.7, 0.2, 0.1])
        
        # Base attributes
        base_attributes = {
            'latitude': 0, 'longitude': 0,
            'altitude_ft': 0, 'speed_kts': 0, 'rcs_m2': 0,
            'electronic_signature': 'UNKNOWN_EMISSION', 'flight_profile': 'STABLE_CRUISE',
            'weather': weather,
            'thermal_signature': 'Not_Detected', # NEW: Default value for our camera sensor
            'classification': class_choice
        }

        # --- Generate data based on class ---
        if class_choice == 'HOSTILE':
            lat, lon = get_random_coords("HOSTILE_AIRSPACE")
            base_attributes.update({
                'latitude': lat, 'longitude': lon,
                'altitude_ft': np.random.uniform(30000, 60000) + np.random.normal(0, 1500), # Add noise
                'speed_kts': np.random.uniform(500, 1300) + np.random.normal(0, 50),
                'rcs_m2': np.random.uniform(0.01, 4),
                'electronic_signature': np.random.choice(['HOSTILE_JAMMING', 'NO_IFF_RESPONSE']),
                'flight_profile': 'AGGRESSIVE_MANEUVERS',
                'thermal_signature': 'High' # Fighter jets have a high thermal signature
            })
        elif class_choice == 'CIVILIAN':
            lat, lon = get_random_coords("CIVIL_AIR_CORRIDOR")
            base_attributes.update({
                'latitude': lat, 'longitude': lon,
                'altitude_ft': np.random.uniform(30000, 42000) + np.random.normal(0, 500),
                'speed_kts': np.random.uniform(400, 550) + np.random.normal(0, 20),
                'rcs_m2': np.random.uniform(30, 100),
                'electronic_signature': 'IFF_MODE_3C',
                'thermal_signature': 'Medium' # Airliners have a noticeable but not extreme signature
            })
        # (You can add more detailed rules for other classes as well)
        else: # Generic catch-all for FRIEND, SUSPECT, NEUTRAL
            zone = np.random.choice(["FRIENDLY_AIRSPACE", "CONTESTED_ZONE", "NEUTRAL_TERRITORY"])
            lat, lon = get_random_coords(zone)
            base_attributes.update({
                'latitude': lat, 'longitude': lon,
                'altitude_ft': np.random.uniform(1000, 55000),
                'speed_kts': np.random.uniform(200, 1500),
                'rcs_m2': np.random.uniform(0.1, 40),
                'electronic_signature': 'IFF_MODE_5' if class_choice == 'FRIEND' else 'NO_IFF_RESPONSE',
                'thermal_signature': np.random.choice(['Low', 'Medium'])
            })
        
        # --- SENSOR FUSION SIMULATION ---
        # The EO/IR camera can fail in bad weather. This is sensor fusion!
        # If the weather is not 'Clear', there's a 75% chance the camera fails.
        if base_attributes['weather'] != 'Clear' and np.random.rand() < 0.75:
            base_attributes['thermal_signature'] = 'Not_Detected' # Simulate sensor failure

        data.append(base_attributes)

    df = pd.DataFrame(data)
    
    # Save to the data directory
    output_path = os.path.join("data", "vanguard_air_tracks_fused.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated fused dataset at '{output_path}'")

if __name__ == '__main__':
    generate_data()