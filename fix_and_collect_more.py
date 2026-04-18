"""
fix_and_collect_more.py

HIZLI ÇÖZÜM:
1. Models klasörünü oluştur
2. Daha fazla veri topla (daha geniş alan)
3. Sentetik veri ekle (çeşitlilik için)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def create_directories():
    """Gerekli klasörleri oluştur"""
    print("📁 Creating directories...")
    
    dirs = ['models', 'data/raw', 'data/processed', 'data/cache']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {d}/")

def enhance_dataset(csv_path):
    """
    Mevcut veriyi zenginleştir - daha fazla çeşitlilik ekle
    """
    print(f"\n🔧 Enhancing dataset: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Original: {len(df)} samples")
    print(f"   Classes: {df['classification'].unique()}")
    
    # Mevcut veriyi analiz et
    class_counts = df['classification'].value_counts()
    print("\n   Current distribution:")
    for cls, count in class_counts.items():
        print(f"      {cls}: {count}")
    
    # Problem: Sadece CIVILIAN ve belki NEUTRAL var
    # Çözüm: Synthetic data augmentation
    
    synthetic_data = []
    
    # 1. HOSTILE uçaklar oluştur (düşük irtifa + yüksek hız)
    print("\n   Creating HOSTILE samples...")
    for i in range(300):
        hostile = {
            'altitude_ft': np.random.uniform(3000, 12000),  # Düşük
            'speed_kts': np.random.uniform(500, 650),       # Hızlı
            'rcs_m2': np.random.uniform(0.5, 5.0),         # Küçük (stealth)
            'latitude': np.random.uniform(48, 53),
            'longitude': np.random.uniform(5, 15),
            'heading': np.random.uniform(0, 360),
            'electronic_signature': np.random.choice(['NO_IFF_RESPONSE', 'HOSTILE_JAMMING', 'UNKNOWN_EMISSION']),
            'flight_profile': np.random.choice(['AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING']),
            'weather': np.random.choice(['Clear', 'Cloudy', 'Rainy']),
            'thermal_signature': np.random.choice(['High', 'Medium']),
            'classification': 'HOSTILE'
        }
        synthetic_data.append(hostile)
    
    # 2. FRIEND (askeri dostlar)
    print("   Creating FRIEND samples...")
    for i in range(200):
        friend = {
            'altitude_ft': np.random.uniform(15000, 30000),
            'speed_kts': np.random.uniform(350, 550),
            'rcs_m2': np.random.uniform(10, 100),  # Büyük askeri uçak
            'latitude': np.random.uniform(48, 53),
            'longitude': np.random.uniform(5, 15),
            'heading': np.random.uniform(0, 360),
            'electronic_signature': 'IFF_MODE_5',  # Askeri IFF
            'flight_profile': np.random.choice(['STABLE_CRUISE', 'CLIMBING']),
            'weather': np.random.choice(['Clear', 'Cloudy']),
            'thermal_signature': np.random.choice(['Medium', 'High']),
            'classification': 'FRIEND'
        }
        synthetic_data.append(friend)
    
    # 3. SUSPECT (şüpheli)
    print("   Creating SUSPECT samples...")
    for i in range(250):
        suspect = {
            'altitude_ft': np.random.uniform(8000, 20000),
            'speed_kts': np.random.uniform(400, 600),
            'rcs_m2': np.random.uniform(5, 30),
            'latitude': np.random.uniform(48, 53),
            'longitude': np.random.uniform(5, 15),
            'heading': np.random.uniform(0, 360),
            'electronic_signature': np.random.choice(['NO_IFF_RESPONSE', 'UNKNOWN_EMISSION']),
            'flight_profile': np.random.choice(['AGGRESSIVE_MANEUVERS', 'STABLE_CRUISE']),
            'weather': np.random.choice(['Clear', 'Cloudy', 'Rainy']),
            'thermal_signature': np.random.choice(['Low', 'Medium', 'High']),
            'classification': 'SUSPECT'
        }
        synthetic_data.append(suspect)
    
    # 4. ASSUMED FRIEND
    print("   Creating ASSUMED FRIEND samples...")
    for i in range(150):
        assumed = {
            'altitude_ft': np.random.uniform(20000, 35000),
            'speed_kts': np.random.uniform(400, 500),
            'rcs_m2': np.random.uniform(15, 50),
            'latitude': np.random.uniform(48, 53),
            'longitude': np.random.uniform(5, 15),
            'heading': np.random.uniform(0, 360),
            'electronic_signature': 'IFF_MODE_3C',  # Standard civilian IFF
            'flight_profile': 'STABLE_CRUISE',
            'weather': np.random.choice(['Clear', 'Cloudy']),
            'thermal_signature': 'Medium',
            'classification': 'ASSUMED FRIEND'
        }
        synthetic_data.append(assumed)
    
    # Synthetic DataFrame oluştur
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Orijinal veriyle birleştir
    enhanced_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Shuffle
    enhanced_df = enhanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n   ✅ Enhanced: {len(enhanced_df)} samples")
    print("\n   New distribution:")
    for cls, count in enhanced_df['classification'].value_counts().items():
        print(f"      {cls}: {count}")
    
    # Save enhanced dataset
    output_path = Path('data/processed/enhanced_training_data.csv')
    enhanced_df.to_csv(output_path, index=False)
    print(f"\n   💾 Saved: {output_path}")
    
    return enhanced_df

def quick_collect_more_data():
    """
    Hızlı veri toplama - farklı bölgeler
    """
    print("\n📡 Collecting more real data from different regions...")
    
    try:
        from src.data_collection import OpenSkyCollector
        
        collector = OpenSkyCollector()
        
        # Farklı bölgeler
        regions = [
            ('Germany', (48, 5, 53, 15)),
            ('France', (45, 0, 50, 7)),
            ('UK', (50, -5, 55, 2)),
            ('Benelux', (50, 3, 53, 7))
        ]
        
        all_data = []
        
        for region_name, bbox in regions:
            print(f"\n   Fetching from {region_name}...")
            df = collector.fetch_states(bbox=bbox)
            
            if len(df) > 0:
                df = collector.enrich_with_features(df)
                all_data.append(df)
                print(f"      ✅ Got {len(df)} aircraft")
            else:
                print(f"      ⚠️  No data from {region_name}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=['icao24'], keep='first')
            
            print(f"\n   ✅ Total collected: {len(combined)} unique aircraft")
            
            # Save
            output_path = Path('data/processed/additional_real_data.csv')
            combined.to_csv(output_path, index=False)
            print(f"   💾 Saved: {output_path}")
            
            return combined
        
        return pd.DataFrame()
    
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   Skipping additional real data collection")
        return pd.DataFrame()

def merge_all_datasets():
    """
    Tüm veri kaynaklarını birleştir
    """
    print("\n🔗 Merging all datasets...")
    
    datasets = []
    
    # Original data
    original_path = Path('data/processed/training_data.csv')
    if original_path.exists():
        df = pd.read_csv(original_path)
        datasets.append(df)
        print(f"   ✅ Original: {len(df)} samples")
    
    # Enhanced data
    enhanced_path = Path('data/processed/enhanced_training_data.csv')
    if enhanced_path.exists():
        df = pd.read_csv(enhanced_path)
        datasets.append(df)
        print(f"   ✅ Enhanced: {len(df)} samples")
    
    # Additional real data
    additional_path = Path('data/processed/additional_real_data.csv')
    if additional_path.exists():
        df = pd.read_csv(additional_path)
        datasets.append(df)
        print(f"   ✅ Additional: {len(df)} samples")
    
    if not datasets:
        print("   ❌ No datasets found!")
        return None
    
    # Merge
    final_df = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates
    before = len(final_df)
    final_df = final_df.drop_duplicates(subset=['latitude', 'longitude', 'altitude_ft'], keep='first')
    after = len(final_df)
    
    print(f"\n   Removed {before - after} duplicates")
    print(f"   ✅ Final dataset: {after} samples")
    
    # Class distribution
    print("\n   📊 Final class distribution:")
    for cls, count in final_df['classification'].value_counts().items():
        percentage = (count / len(final_df)) * 100
        print(f"      {cls}: {count} ({percentage:.1f}%)")
    
    # Save final dataset
    final_path = Path('data/processed/final_training_data.csv')
    final_df.to_csv(final_path, index=False)
    print(f"\n   💾 Saved: {final_path}")
    
    return final_df

def main():
    print("="*70)
    print("🛡️  VANGUARD AI - Quick Fix & Data Enhancement")
    print("="*70)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Check existing data
    original_data = Path('data/processed/training_data.csv')
    
    if not original_data.exists():
        print("\n❌ No training data found!")
        print("   Please run data collection first:")
        print("   python -c \"from src.data_collection import OpenSkyCollector; ...\"")
        return
    
    # Step 3: Enhance existing data
    enhanced_df = enhance_dataset(original_data)
    
    # Step 4: Collect more real data (optional, quick)
    print("\n" + "="*70)
    collect_more = input("\nCollect more real data? (takes 2-3 minutes) [y/n]: ")
    
    if collect_more.lower() == 'y':
        additional_df = quick_collect_more_data()
    
    # Step 5: Merge everything
    final_df = merge_all_datasets()
    
    if final_df is not None:
        print("\n" + "="*70)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*70)
        print("\n🚀 Now you can train the model:")
        print("   python src/model_pytorch.py")
        print("\n   OR use the enhanced dataset directly:")
        print("   python -c \"from src.model_pytorch import train_model;")
        print("   train_model('data/processed/final_training_data.csv')\"")
        print("\n" + "="*70)

if __name__ == "__main__":
    main()