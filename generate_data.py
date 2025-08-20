# generate_data.py
import pandas as pd
import numpy as np

# Define geographic bounding boxes for each zone (Lat, Lon)
# These are rough estimates for demonstration purposes
ZONE_BOUNDARIES = {
    "FRIENDLY_AIRSPACE": {"lat": (48, 54), "lon": (8, 15)},      # e.g., Germany
    "HOSTILE_AIRSPACE": {"lat": (44, 47), "lon": (33, 40)},       # e.g., Black Sea region
    "NEUTRAL_TERRITORY": {"lat": (46, 47.5), "lon": (6, 10.5)},  # e.g., Switzerland
    "CIVIL_AIR_CORRIDOR": {"lat": (42, 52), "lon": (0, 30)},     # e.g., A path across Europe
    "CONTESTED_ZONE": {"lat": (54, 58), "lon": (18, 25)}         # e.g., Baltic Sea region
}

def get_random_coords(zone):
    """Generates random latitude and longitude within a zone's boundaries."""
    bounds = ZONE_BOUNDARIES[zone]
    lat = np.random.uniform(bounds["lat"][0], bounds["lat"][1])
    lon = np.random.uniform(bounds["lon"][0], bounds["lon"][1])
    return lat, lon

# --- Data Generation Logic ---
n_samples = 2500
data = []

for _ in range(n_samples):
    class_choice = np.random.choice(
        ['FRIEND', 'ASSUMED FRIEND', 'HOSTILE', 'SUSPECT', 'NEUTRAL', 'UNKNOWN', 'CIVILIAN'],
        p=[0.15, 0.15, 0.10, 0.15, 0.05, 0.10, 0.30]
    )

    if class_choice == 'FRIEND':
        lat, lon = get_random_coords("FRIENDLY_AIRSPACE")
        data.append({
            'latitude': lat, 'longitude': lon,
            'altitude_ft': np.random.uniform(25000, 50000),
            'speed_kts': np.random.uniform(450, 1200),
            'rcs_m2': np.random.uniform(1, 15),
            'electronic_signature': 'IFF_MODE_5',
            'flight_profile': 'STABLE_CRUISE',
            'classification': 'FRIEND'
        })
    elif class_choice == 'HOSTILE':
        lat, lon = get_random_coords("HOSTILE_AIRSPACE")
        data.append({
            'latitude': lat, 'longitude': lon,
            'altitude_ft': np.random.uniform(30000, 60000),
            'speed_kts': np.random.uniform(500, 1300),
            'rcs_m2': np.random.uniform(0.01, 4),
            'electronic_signature': np.random.choice(['HOSTILE_JAMMING', 'NO_IFF_RESPONSE']),
            'flight_profile': 'AGGRESSIVE_MANEUVERS',
            'classification': 'HOSTILE'
        })
    elif class_choice == 'CIVILIAN':
        lat, lon = get_random_coords("CIVIL_AIR_CORRIDOR")
        data.append({
            'latitude': lat, 'longitude': lon,
            'altitude_ft': np.random.uniform(30000, 42000),
            'speed_kts': np.random.uniform(400, 550),
            'rcs_m2': np.random.uniform(30, 100),
            'electronic_signature': 'IFF_MODE_3C',
            'flight_profile': 'STABLE_CRUISE',
            'classification': 'CIVILIAN'
        })
    # ... (Add similar logic for SUSPECT, NEUTRAL, etc., assigning them to zones)
    else: # Simplified catch-all for other classes
        zone = np.random.choice(["NEUTRAL_TERRITORY", "CONTESTED_ZONE", "FRIENDLY_AIRSPACE"])
        lat, lon = get_random_coords(zone)
        data.append({
            'latitude': lat, 'longitude': lon,
            'altitude_ft': np.random.uniform(1000, 55000),
            'speed_kts': np.random.uniform(200, 1500),
            'rcs_m2': np.random.uniform(0.1, 40),
            'electronic_signature': np.random.choice(['NO_IFF_RESPONSE', 'UNKNOWN_EMISSION', 'IFF_MODE_3C']),
            'flight_profile': np.random.choice(['STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS']),
            'classification': class_choice
        })

df = pd.DataFrame(data)
df.to_csv('vanguard_air_tracks.csv', index=False)

print(f"Successfully generated 'vanguard_air_tracks.csv' with {len(df)} samples.")