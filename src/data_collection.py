"""
src/data_collection.py

OpenSky Network veri toplama ve işleme modülü
Gerçek uçak verilerini toplar ve sınıflandırma için hazırlar
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenSkyCollector:
    """OpenSky Network'ten gerçek uçak verisi toplama"""
    
    BASE_URL = "https://opensky-network.org/api"
    
    # RCS Database (gerçek değerler - açık kaynaklardan)
    RCS_DATABASE = {
        # Ticari uçaklar
        'B737': {'mean': 40, 'std': 5, 'type': 'CIVILIAN'},
        'B738': {'mean': 40, 'std': 5, 'type': 'CIVILIAN'},
        'A320': {'mean': 35, 'std': 4, 'type': 'CIVILIAN'},
        'A321': {'mean': 38, 'std': 4, 'type': 'CIVILIAN'},
        'B77W': {'mean': 80, 'std': 10, 'type': 'CIVILIAN'},
        'A359': {'mean': 75, 'std': 8, 'type': 'CIVILIAN'},
        
        # Askeri uçaklar (tahmini değerler)
        'C130': {'mean': 100, 'std': 15, 'type': 'FRIEND'},
        'C17': {'mean': 120, 'std': 20, 'type': 'FRIEND'},
        'A400': {'mean': 110, 'std': 18, 'type': 'FRIEND'},
        
        # Küçük uçaklar
        'C172': {'mean': 1.5, 'std': 0.3, 'type': 'CIVILIAN'},
        'C208': {'mean': 3.0, 'std': 0.5, 'type': 'CIVILIAN'},
        
        # Savaş uçakları (literature values)
        'F16': {'mean': 1.2, 'std': 0.3, 'type': 'FRIEND'},
        'F15': {'mean': 25, 'std': 5, 'type': 'FRIEND'},
        'F18': {'mean': 1.0, 'std': 0.2, 'type': 'FRIEND'},
        'F35': {'mean': 0.005, 'std': 0.001, 'type': 'FRIEND'},
        
        # Default
        'UNKNOWN': {'mean': 10, 'std': 3, 'type': 'UNKNOWN'}
    }
    
    def __init__(self, username=None, password=None):
        self.auth = (username, password) if username and password else None
        self.last_request_time = 0
        self.min_request_interval = 10  # Rate limiting
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_states(self, bbox=None):
        """
        Gerçek zamanlı uçak durumlarını çek
        
        Args:
            bbox: tuple (lat_min, lon_min, lat_max, lon_max)
                  Default: Avrupa (45, 5, 55, 15)
        
        Returns:
            DataFrame with flight data
        """
        if bbox is None:
            # Default: Central Europe
            bbox = (45, 5, 55, 15)
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/states/all"
            params = {
                'lamin': bbox[0], 'lomin': bbox[1],
                'lamax': bbox[2], 'lomax': bbox[3]
            }
            
            logger.info(f"Fetching states from OpenSky (bbox: {bbox})")
            response = requests.get(url, params=params, auth=self.auth, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['states'] is None or len(data['states']) == 0:
                    logger.warning("No aircraft data returned")
                    return pd.DataFrame()
                
                # Parse response
                df = self._parse_states(data['states'])
                
                logger.info(f"✅ Retrieved {len(df)} aircraft")
                return df
            
            else:
                logger.error(f"API Error: Status {response.status_code}")
                return pd.DataFrame()
        
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _parse_states(self, states):
        """Parse OpenSky states response"""
        
        columns = [
            'icao24', 'callsign', 'origin_country', 'time_position',
            'last_contact', 'longitude', 'latitude', 'baro_altitude',
            'on_ground', 'velocity', 'true_track', 'vertical_rate',
            'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'
        ]
        
        df = pd.DataFrame(states, columns=columns)
        
        # Clean data
        df['callsign'] = df['callsign'].str.strip()
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['baro_altitude'] = pd.to_numeric(df['baro_altitude'], errors='coerce')
        df['velocity'] = pd.to_numeric(df['velocity'], errors='coerce')
        df['true_track'] = pd.to_numeric(df['true_track'], errors='coerce')
        
        # Filter out invalid data
        df = df.dropna(subset=['latitude', 'longitude', 'baro_altitude', 'velocity'])
        df = df[df['on_ground'] == False]  # Only airborne
        
        # Convert units
        df['altitude_ft'] = df['baro_altitude'] * 3.28084  # m to ft
        df['speed_kts'] = df['velocity'] * 1.94384  # m/s to knots
        df['heading'] = df['true_track']
        
        # Timestamp
        df['timestamp'] = datetime.now().isoformat()
        
        return df
    
    def enrich_with_features(self, df):
        """
        OpenSky verisine ek özellikler ekle (Helsing için önemli!)
        """
        logger.info("Enriching data with additional features...")
        
        # 1. RCS Estimation
        df['aircraft_type'] = df['callsign'].apply(self._infer_aircraft_type)
        df['rcs_m2'] = df.apply(self._estimate_rcs, axis=1)
        
        # 2. Electronic Signature (callsign'dan çıkarım)
        df['electronic_signature'] = df.apply(self._infer_electronic_sig, axis=1)
        
        # 3. Flight Profile
        df['flight_profile'] = df.apply(self._infer_flight_profile, axis=1)
        
        # 4. Weather (simplified - her yer "Clear" olarak başla)
        df['weather'] = 'Clear'
        
        # 5. Thermal Signature (altitude ve speed'den tahmin)
        df['thermal_signature'] = df.apply(self._infer_thermal, axis=1)
        
        # 6. Classification (rule-based labeling)
        df['classification'] = df.apply(self._infer_classification, axis=1)
        
        return df
    
    def _infer_aircraft_type(self, callsign):
        """Callsign'dan uçak tipini tahmin et"""
        if pd.isna(callsign) or len(callsign) < 3:
            return 'UNKNOWN'
        
        # Ticari havayolları genellikle 3 harf + sayı
        prefix = callsign[:3].upper()
        
        # Major airlines use specific aircraft types
        airline_fleet = {
            'DLH': 'A320',  # Lufthansa
            'AFR': 'A320',  # Air France
            'BAW': 'A320',  # British Airways
            'UAL': 'B737',  # United
            'AAL': 'B737',  # American
            'DAL': 'B737',  # Delta
            'KLM': 'B737',  # KLM
        }
        
        return airline_fleet.get(prefix, 'UNKNOWN')
    
    def _estimate_rcs(self, row):
        """RCS tahmini - physics-based approach"""
        
        aircraft_type = row['aircraft_type']
        altitude = row['altitude_ft']
        
        # Base RCS from database
        rcs_data = self.RCS_DATABASE.get(aircraft_type, self.RCS_DATABASE['UNKNOWN'])
        base_rcs = np.random.normal(rcs_data['mean'], rcs_data['std'])
        
        # Aspect angle effect (simplified)
        # Varsayım: random aspect between 0-180 degrees
        aspect_angle = np.random.uniform(0, 180)
        aspect_factor = 1 + 0.5 * np.sin(np.radians(aspect_angle))
        
        # Altitude effect (iyonosferik propagation)
        altitude_factor = 1 - (altitude / 100000) * 0.1
        altitude_factor = max(0.8, altitude_factor)
        
        # Final RCS
        rcs = base_rcs * aspect_factor * altitude_factor
        
        return max(0.01, rcs)  # Minimum 0.01 m²
    
    def _infer_electronic_sig(self, row):
        """Electronic signature inference"""
        
        callsign = str(row['callsign']).strip().upper()
        
        # Military callsigns
        military_keywords = ['ARMY', 'NAVY', 'AIR', 'NATO', 'FORCE']
        if any(keyword in callsign for keyword in military_keywords):
            return 'IFF_MODE_5'  # Secure military
        
        # Commercial airlines
        if len(callsign) >= 3 and callsign[:3].isalpha():
            return 'IFF_MODE_3C'  # Standard civilian
        
        # No callsign or unusual
        if callsign == 'NAN' or len(callsign) < 3:
            return 'NO_IFF_RESPONSE'
        
        return 'UNKNOWN_EMISSION'
    
    def _infer_flight_profile(self, row):
        """Flight profile inference"""
        
        altitude = row['altitude_ft']
        speed = row['speed_kts']
        vertical_rate = row.get('vertical_rate', 0)
        
        # Climbing/Descending
        if pd.notna(vertical_rate):
            if vertical_rate > 10:  # m/s
                return 'CLIMBING'
            elif vertical_rate < -10:
                return 'DESCENDING'
        
        # Low altitude + high speed = aggressive
        if altitude < 10000 and speed > 400:
            return 'AGGRESSIVE_MANEUVERS'
        
        # Very low altitude
        if altitude < 5000:
            return 'LOW_ALTITUDE_FLYING'
        
        # Default
        return 'STABLE_CRUISE'
    
    def _infer_thermal(self, row):
        """Thermal signature inference"""
        
        altitude = row['altitude_ft']
        speed = row['speed_kts']
        
        # High speed = more heat
        if speed > 500:
            return 'High'
        elif speed > 400:
            return 'Medium'
        else:
            return 'Low'
    
    def _infer_classification(self, row):
        """
        Rule-based classification (ground truth için)
        Gerçek dünyada bu label'lar manuel olarak verilir
        """
        
        callsign = str(row['callsign']).strip().upper()
        altitude = row['altitude_ft']
        speed = row['speed_kts']
        electronic_sig = row['electronic_signature']
        
        # Military identification
        if electronic_sig == 'IFF_MODE_5':
            return 'FRIEND'
        
        # Commercial airlines (high confidence)
        commercial_prefixes = ['DLH', 'AFR', 'BAW', 'UAL', 'AAL', 'DAL', 'KLM', 'RYR', 'EZY']
        if any(callsign.startswith(prefix) for prefix in commercial_prefixes):
            return 'CIVILIAN'
        
        # Suspicious behavior
        if altitude < 8000 and speed > 450 and electronic_sig == 'NO_IFF_RESPONSE':
            return 'SUSPECT'
        
        # High altitude civilian traffic
        if altitude > 28000 and 400 < speed < 550:
            return 'CIVILIAN'
        
        # Unknown
        return 'NEUTRAL'
    
    def collect_dataset(self, bbox=None, duration_minutes=60, samples_per_hour=6):
        """
        Belirli süre boyunca veri topla
        
        Args:
            bbox: Geographic bounding box
            duration_minutes: Toplama süresi
            samples_per_hour: Saatte kaç sample
        
        Returns:
            DataFrame with collected data
        """
        interval = 60 / samples_per_hour  # minutes
        num_samples = int(duration_minutes / interval)
        
        logger.info(f"🚀 Starting data collection:")
        logger.info(f"   Duration: {duration_minutes} minutes")
        logger.info(f"   Samples: {num_samples}")
        logger.info(f"   Interval: {interval} minutes")
        
        all_data = []
        
        for i in range(num_samples):
            logger.info(f"\n[{i+1}/{num_samples}] Fetching data...")
            
            # Fetch states
            df = self.fetch_states(bbox)
            
            if len(df) > 0:
                # Enrich with features
                df = self.enrich_with_features(df)
                all_data.append(df)
                
                logger.info(f"   ✅ Collected {len(df)} aircraft")
            else:
                logger.warning("   ⚠️  No data collected")
            
            # Wait for next sample (except last iteration)
            if i < num_samples - 1:
                wait_time = interval * 60  # Convert to seconds
                logger.info(f"   ⏳ Waiting {interval} minutes until next sample...")
                time.sleep(wait_time)
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates (same aircraft in multiple samples)
            final_df = final_df.drop_duplicates(
                subset=['icao24', 'latitude', 'longitude'], 
                keep='first'
            )
            
            logger.info(f"\n✅ Collection complete!")
            logger.info(f"   Total unique aircraft: {len(final_df)}")
            logger.info(f"   Classification breakdown:")
            
            for cls, count in final_df['classification'].value_counts().items():
                logger.info(f"      - {cls}: {count}")
            
            return final_df
        
        else:
            logger.error("❌ No data collected!")
            return pd.DataFrame()
    
    def save_dataset(self, df, filename='flight_data.csv'):
        """Save dataset to CSV"""
        
        output_path = Path('data/processed') / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Dataset saved: {output_path}")
        
        # Also save as JSON for backup
        json_path = output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"💾 JSON backup saved: {json_path}")
        
        return output_path


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    print("🛡️  VANGUARD AI - Data Collection Module")
    print("=" * 60)
    print()
    
    # Initialize collector
    collector = OpenSkyCollector()
    
    # Quick test - single fetch
    print("📡 Test: Fetching current flights...")
    test_df = collector.fetch_states(bbox=(48, 5, 53, 15))  # Germany region
    
    if len(test_df) > 0:
        print(f"✅ Success! Found {len(test_df)} aircraft")
        print("\nSample data:")
        print(test_df[['callsign', 'altitude_ft', 'speed_kts', 'latitude', 'longitude']].head())
        
        # Enrich
        enriched = collector.enrich_with_features(test_df)
        print("\n✅ Enriched with ML features")
        print("\nFeatures added:")
        print(enriched[['callsign', 'rcs_m2', 'classification', 'electronic_signature']].head())
        
        print("\n" + "=" * 60)
        print("\n🚀 Ready for full data collection!")
        print("\nTo collect a full dataset, run:")
        print(">>> dataset = collector.collect_dataset(duration_minutes=120, samples_per_hour=6)")
        print(">>> collector.save_dataset(dataset, 'training_data.csv')")
        
    else:
        print("⚠️  No aircraft found. Try again or check your internet connection.")