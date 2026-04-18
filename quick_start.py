"""
quick_start.py

Tek komutla VANGUARD AI projesini başlatın!
Bu script tüm adımları otomatik yapar:
1. Proje yapısını oluşturur
2. Gerçek veri toplar (OpenSky Network)
3. PyTorch modelini eğitir
4. Streamlit uygulamasını başlatır

Kullanım: python quick_start.py
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step, description):
    print(f"\n🔹 STEP {step}: {description}")
    print("-" * 70)

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"   Running: {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        print(f"   Output: {e.output}")
        return False

def create_project_structure():
    """Create complete project structure"""
    print_step(1, "Creating Project Structure")
    
    structure = {
        'src': ['__init__.py', 'data_collection.py', 'model_pytorch.py', 'inference.py'],
        'models': [],
        'data': ['raw', 'processed', 'cache'],
        'tests': ['__init__.py'],
        'notebooks': [],
    }
    
    for folder, files in structure.items():
        folder_path = Path(folder)
        folder_path.mkdir(exist_ok=True)
        print(f"   📁 Created: {folder}/")
        
        for file in files:
            if '/' in file or not file:
                # Subdirectory
                if file:
                    sub_dir = folder_path / file
                    sub_dir.mkdir(exist_ok=True)
            else:
                # File
                file_path = folder_path / file
                if not file_path.exists():
                    file_path.touch()
    
    # Root files
    root_files = ['README.md', 'requirements.txt', '.gitignore', 'app.py']
    for file in root_files:
        Path(file).touch()
    
    print("   ✅ Project structure created!")

def install_dependencies():
    """Install required packages"""
    print_step(2, "Installing Dependencies")
    
    requirements = """torch==2.1.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3
joblib==1.3.2
streamlit==1.28.0
requests==2.31.0
plotly==5.18.0
matplotlib==3.8.2
seaborn==0.13.0
tqdm==4.66.1
"""
    
    # Write requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("   📦 Installing packages (this may take a few minutes)...")
    
    # Try to install
    success = run_command(
        f"{sys.executable} -m pip install -q -r requirements.txt",
        "pip install"
    )
    
    if not success:
        print("   ⚠️  Some packages failed to install. Please run manually:")
        print("   pip install -r requirements.txt")
    
    return success

def collect_data():
    """Collect real-world data from OpenSky Network"""
    print_step(3, "Collecting Real-World Flight Data")
    
    print("""
   📡 Connecting to OpenSky Network...
   
   This will collect REAL aircraft data from European airspace.
   Duration: ~30 minutes (collecting 10 samples)
   Coverage: Germany, France, Netherlands, Belgium
   
   ⏳ Please wait... (grab a coffee! ☕)
    """)
    
    # Check if data collection module exists
    data_collection_path = Path('src/data_collection.py')
    
    if not data_collection_path.exists():
        print("   ⚠️  Data collection module not found!")
        print("   Please ensure all artifacts are properly saved.")
        return False
    
    try:
        # Import and run data collection
        print("   🚀 Starting data collection...")
        
        # Simple inline data collection
        collection_code = """
from src.data_collection import OpenSkyCollector
import logging

logging.basicConfig(level=logging.INFO)

collector = OpenSkyCollector()
dataset = collector.collect_dataset(
    bbox=(48, 5, 53, 15),  # Germany region
    duration_minutes=30,
    samples_per_hour=20
)

if len(dataset) > 0:
    collector.save_dataset(dataset, 'training_data.csv')
    print(f"✅ Collected {len(dataset)} flight records!")
else:
    print("❌ No data collected")
"""
        
        # Execute collection
        exec(collection_code)
        
        # Check if file was created
        data_file = Path('data/processed/training_data.csv')
        if data_file.exists():
            print(f"   ✅ Data saved: {data_file}")
            return True
        else:
            print("   ⚠️  Data file not created")
            return False
    
    except Exception as e:
        print(f"   ❌ Error during data collection: {e}")
        print("""
   💡 TIP: You can collect data manually:
   
   from src.data_collection import OpenSkyCollector
   collector = OpenSkyCollector()
   dataset = collector.collect_dataset(duration_minutes=60)
   collector.save_dataset(dataset, 'training_data.csv')
        """)
        return False

def train_model():
    """Train the PyTorch model"""
    print_step(4, "Training PyTorch Model")
    
    # Check if data exists
    data_file = Path('data/processed/training_data.csv')
    
    if not data_file.exists():
        print("   ⚠️  Training data not found!")
        print("   Skipping model training...")
        return False
    
    print("""
   🤖 Training neural network...
   
   Architecture: 3-layer feedforward network
   Optimizer: AdamW with learning rate scheduling
   Expected time: 5-10 minutes (CPU) or 1-2 minutes (GPU)
   
   ⏳ Training in progress...
    """)
    
    try:
        # Train model
        training_code = """
from src.model_pytorch import train_model

print("Starting training...")
model, scaler, label_encoder, feature_cols, history = train_model(
    data_path='data/processed/training_data.csv',
    epochs=30  # Reduced for quick start
)
print("Training complete!")
"""
        
        exec(training_code)
        
        # Check if model was saved
        model_file = Path('models/best_model.pt')
        if model_file.exists():
            print(f"   ✅ Model trained and saved: {model_file}")
            return True
        else:
            print("   ⚠️  Model file not created")
            return False
    
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        return False

def create_streamlit_app():
    """Create a simple Streamlit app"""
    print_step(5, "Creating Streamlit Application")
    
    app_code = '''"""
VANGUARD AI - Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="VANGUARD AI", page_icon="🛡️", layout="wide")

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; border-radius: 10px;">
    <h1>🛡️ VANGUARD AI</h1>
    <p>Advanced Air Defense Classification System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if model exists
model_path = Path('models/best_model.pt')
data_path = Path('data/processed/training_data.csv')

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 System Status")
    
    if model_path.exists():
        st.success("✅ Model: Loaded")
    else:
        st.warning("⚠️ Model: Not trained yet")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.info(f"📡 Dataset: {len(df):,} flights")
        
        st.subheader("📈 Dataset Statistics")
        class_counts = df['classification'].value_counts()
        st.bar_chart(class_counts)
    else:
        st.error("❌ Dataset: Not collected yet")

with col2:
    st.subheader("🎯 Quick Classification Demo")
    
    if model_path.exists():
        try:
            from src.inference import VanguardInference
            
            inference = VanguardInference()
            
            # Input form
            altitude = st.slider("Altitude (ft)", 0, 60000, 35000)
            speed = st.slider("Speed (knots)", 100, 800, 450)
            
            col_a, col_b = st.columns(2)
            with col_a:
                electronic = st.selectbox("Electronic Signature", 
                    ['IFF_MODE_5', 'IFF_MODE_3C', 'NO_IFF_RESPONSE'])
            with col_b:
                flight_profile = st.selectbox("Flight Profile",
                    ['STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'CLIMBING'])
            
            if st.button("🔍 Classify Aircraft", type="primary"):
                input_data = {
                    'altitude_ft': altitude,
                    'speed_kts': speed,
                    'rcs_m2': 15.0,
                    'latitude': 51.5,
                    'longitude': -0.1,
                    'heading': 270,
                    'electronic_signature': electronic,
                    'flight_profile': flight_profile,
                    'weather': 'Clear',
                    'thermal_signature': 'Medium'
                }
                
                result = inference.predict(input_data)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px; margin-top: 1rem;">
                    <h2>{result['classification']}</h2>
                    <p style="font-size: 1.5rem;">Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("All Probabilities")
                probs_df = pd.DataFrame(
                    result['all_probabilities'].items(),
                    columns=['Class', 'Probability']
                ).sort_values('Probability', ascending=False)
                
                st.bar_chart(probs_df.set_index('Class'))
        
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.info("Please train the model first")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛡️ VANGUARD AI v1.0 | Built with PyTorch & Streamlit</p>
    <p>Real-world ADS-B data from OpenSky Network</p>
</div>
""", unsafe_allow_html=True)
'''
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_code)
    
    print("   ✅ Streamlit app created: app.py")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Data
data/raw/*
data/cache/*
*.csv
*.json

# Models
models/*.pt
models/*.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
logs/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)

def main():
    """Main execution"""
    
    print_header("🛡️ VANGUARD AI - Quick Start Setup")
    
    print("""
    This script will:
    1. Create project structure
    2. Install dependencies
    3. Collect REAL flight data from OpenSky Network
    4. Train PyTorch model
    5. Launch Streamlit application
    
    Estimated time: 45-60 minutes
    
    Press Ctrl+C to cancel at any time.
    """)
    
    input("Press Enter to continue...")
    
    start_time = time.time()
    
    # Execute steps
    steps = [
        (create_project_structure, "Project structure"),
        (create_gitignore, "Git configuration"),
        (install_dependencies, "Dependencies"),
        (collect_data, "Data collection"),
        (train_model, "Model training"),
        (create_streamlit_app, "Streamlit app"),
    ]
    
    completed = []
    failed = []
    
    for step_func, step_name in steps:
        try:
            success = step_func()
            if success or success is None:
                completed.append(step_name)
            else:
                failed.append(step_name)
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            failed.append(step_name)
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print_header("Setup Complete!")
    
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes\n")
    
    print("✅ Completed steps:")
    for step in completed:
        print(f"   • {step}")
    
    if failed:
        print("\n⚠️  Failed/Skipped steps:")
        for step in failed:
            print(f"   • {step}")
    
    print("\n" + "="*70)
    print("\n🚀 Next Steps:\n")
    
    if Path('models/best_model.pt').exists():
        print("   1. Launch the application:")
        print("      streamlit run app.py")
        print("\n   2. Open browser at: http://localhost:8501")
        print("\n   3. Start classifying aircraft!")
    else:
        print("   ⚠️  Model training incomplete. Please run:")
        print("      python src/model_pytorch.py")
    
    print("\n📚 Documentation: README.md")
    print("🐛 Issues: Check logs for errors")
    print("\n" + "="*70 + "\n")
    
    # Offer to start app
    if Path('models/best_model.pt').exists():
        start_app = input("\nStart Streamlit app now? (y/n): ")
        if start_app.lower() == 'y':
            print("\n🚀 Starting Streamlit...\n")
            os.system("streamlit run app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)