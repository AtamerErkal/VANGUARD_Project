\# 🛡️ VANGUARD AI Defense System



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)



An advanced AI-powered air track classification and threat assessment platform utilizing machine learning for real-time defense monitoring and analysis.



!\[VANGUARD System](https://img.shields.io/badge/Status-Operational-success)



\## 📋 Table of Contents



\- \[Overview](#overview)

\- \[Key Features](#key-features)

\- \[System Architecture](#system-architecture)

\- \[Installation](#installation)

\- \[Usage](#usage)

\- \[Operating Modes](#operating-modes)

\- \[Technical Components](#technical-components)

\- \[Model Information](#model-information)

\- \[Screenshots](#screenshots)

\- \[Contributing](#contributing)

\- \[License](#license)

\- \[Contact](#contact)



\## 🎯 Overview



VANGUARD (Vigilant AI Network for Ground-based UAV and Aircraft Recognition in Defense) is a comprehensive defense system designed to classify, track, and assess aerial threats in real-time. The system leverages machine learning algorithms to analyze radar signatures, flight patterns, and sensor data to provide accurate threat assessments.



\### Mission



To provide military and civilian air traffic controllers with an intelligent, automated system for identifying and tracking aircraft, detecting anomalies, and assessing potential threats with high accuracy and minimal latency.



\## ✨ Key Features



\### 🎯 Core Capabilities

\- \*\*AI-Powered Classification\*\*: Machine learning model for aircraft identification (HOSTILE, FRIEND, CIVILIAN, SUSPECT, NEUTRAL, ASSUMED FRIEND)

\- \*\*Real-Time Threat Assessment\*\*: Dynamic threat scoring system with multi-factor analysis

\- \*\*Multi-Track Monitoring\*\*: Simultaneous tracking of multiple aircraft with conflict detection

\- \*\*Anomaly Detection\*\*: Automated detection of unusual flight patterns and behaviors

\- \*\*3D Flight Path Visualization\*\*: Interactive track history with Plotly 3D rendering

\- \*\*Conflict Detection\*\*: Real-time separation monitoring and collision warning



\### 📊 Advanced Analytics

\- \*\*Threat Matrix Dashboard\*\*: Comprehensive threat level visualization

\- \*\*Historical Track Playback\*\*: Realistic flight path generation and analysis

\- \*\*Sensor Data Integration\*\*: Multi-source data fusion (radar, thermal, electronic signatures)

\- \*\*System Metrics Monitoring\*\*: Performance tracking and uptime statistics



\### 🔬 Intelligence Features

\- Speed anomaly detection with classification-specific thresholds

\- Altitude pattern analysis for terrain-following detection

\- Electronic signature profiling (IFF modes, jamming detection)

\- Weather-adaptive threat assessment

\- Proximity-based risk evaluation



\## 🏗️ System Architecture



```

VANGUARD System

│

├── Frontend (Streamlit)

│   ├── Single Track Analysis

│   ├── Multi-Track Monitoring

│   ├── Track History Visualization

│   └── Advanced Analytics Dashboard

│

├── AI/ML Pipeline

│   ├── Pre-trained Classification Model

│   ├── Feature Scaler

│   └── Training Column Schema

│

├── Core Modules

│   ├── TrackHistoryGenerator

│   ├── MultiAircraftTracker

│   ├── ConflictDetector

│   ├── AnomalyDetector

│   └── ThreatAssessment

│

└── Data Processing

&nbsp;   ├── Sensor Data Integration

&nbsp;   ├── Real-time Track Management

&nbsp;   └── Historical Analysis

```



\## 🚀 Installation



\### Prerequisites



\- Python 3.8 or higher

\- pip package manager

\- 4GB+ RAM recommended



\### Step 1: Clone the Repository



```bash

git clone https://github.com/AtamerErkal/VANGUARD\_Project.git

cd VANGUARD\_Project

```



\### Step 2: Create Virtual Environment (Recommended)



```bash

\# On Windows

python -m venv venv

venv\\Scripts\\activate



\# On macOS/Linux

python3 -m venv venv

source venv/bin/activate

```



\### Step 3: Install Dependencies



```bash

pip install -r requirements.txt

```



\### Required Dependencies



```txt

streamlit>=1.28.0

pandas>=2.0.0

numpy>=1.24.0

joblib>=1.3.0

plotly>=5.17.0

scikit-learn>=1.3.0

```



\### Step 4: Model Files



Ensure the following model files are present in the `models/` directory:

\- `vanguard\_classifier.joblib` - Main classification model

\- `vanguard\_scaler.joblib` - Feature scaler

\- `training\_columns.joblib` - Column schema



\## 💻 Usage



\### Running the Application



```bash

streamlit run app.py

```



The application will launch in your default web browser at `http://localhost:8501`



\### Quick Start Guide



1\. \*\*Single Track Analysis Mode\*\*

&nbsp;  - Enter aircraft position (latitude/longitude)

&nbsp;  - Set radar parameters (altitude, speed, RCS)

&nbsp;  - Configure sensor data (weather, thermal signature, electronic profile)

&nbsp;  - Click "ANALYZE TRACK" to classify



2\. \*\*Multi-Track Monitoring\*\*

&nbsp;  - View all active tracks in real-time

&nbsp;  - Monitor conflict detection alerts

&nbsp;  - Assess threat levels across all aircraft



3\. \*\*Track History\*\*

&nbsp;  - Generate realistic flight paths

&nbsp;  - Visualize 3D trajectories

&nbsp;  - Analyze altitude and speed profiles



4\. \*\*Advanced Analytics\*\*

&nbsp;  - Review anomaly detection results

&nbsp;  - Examine threat matrix

&nbsp;  - Monitor system metrics



\## 🎮 Operating Modes



\### 🎯 Single Track Analysis

Analyze individual aircraft tracks with detailed sensor data input and AI classification.



\*\*Input Parameters:\*\*

\- Geographic Position (Lat/Lon)

\- Altitude (0-65,000 ft)

\- Speed (0-2,000 knots)

\- Radar Cross Section (0.1-100 m²)

\- Weather Conditions

\- Thermal Signature

\- Electronic Signature (IFF modes)

\- Flight Profile



\*\*Output:\*\*

\- Classification result

\- Map visualization

\- Anomaly warnings

\- Track ID assignment



\### 🌐 Multi-Track Monitoring

Real-time dashboard for tracking multiple aircraft simultaneously.



\*\*Features:\*\*

\- Active track count

\- Hostile aircraft counter

\- Average altitude/speed metrics

\- Conflict detection system

\- Threat assessment matrix

\- Interactive map with all tracks



\*\*Conflict Detection:\*\*

\- Horizontal separation minimum: 5 NM

\- Vertical separation minimum: 1,000 ft

\- Severity levels: CRITICAL (<2 NM) / WARNING (2-5 NM)



\### 📊 Track History

Generate and visualize realistic flight paths based on aircraft classification.



\*\*Capabilities:\*\*

\- 30-minute track generation

\- 30-second interval updates

\- Classification-specific behavior patterns

\- 3D flight path rendering

\- Altitude/speed profile charts



\### 🔍 Advanced Analytics

Comprehensive analysis dashboard for system-wide intelligence.



\*\*Analytics Modules:\*\*

1\. \*\*Anomaly Detection\*\*

&nbsp;  - Speed deviation analysis

&nbsp;  - Altitude anomaly detection

&nbsp;  - Behavior pattern recognition



2\. \*\*Threat Matrix\*\*

&nbsp;  - Multi-factor threat scoring

&nbsp;  - Priority-sorted track list

&nbsp;  - Visual threat level indicators



3\. \*\*System Metrics\*\*

&nbsp;  - Total track count

&nbsp;  - System uptime monitoring

&nbsp;  - Model accuracy statistics



\## 🔧 Technical Components



\### TrackHistoryGenerator

Generates realistic flight paths based on aircraft classification.



```python

generator = TrackHistoryGenerator(start\_lat, start\_lon, classification)

track\_data = generator.generate\_realistic\_track(duration\_minutes=30, interval\_seconds=30)

```



\### MultiAircraftTracker

Manages multiple aircraft tracks with persistent state.



\*\*Methods:\*\*

\- `add\_track()` - Register new aircraft

\- `get\_all\_tracks()` - Retrieve active tracks

\- Track history maintenance



\### ConflictDetector

Monitors separation standards and identifies potential conflicts.



\*\*Parameters:\*\*

\- Horizontal separation minimum: 5.0 NM

\- Vertical separation minimum: 1,000 ft



\*\*Output:\*\*

\- Conflict pairs

\- Separation distances

\- Severity classification



\### AnomalyDetector

Identifies unusual behavior patterns.



\*\*Detection Types:\*\*

\- Speed anomalies (unusually slow/fast)

\- Altitude anomalies (low altitude, extreme altitude)

\- Terrain-following behavior

\- Classification-specific threshold violations



\### ThreatAssessment

Multi-factor threat scoring system.



\*\*Threat Score Weights:\*\*

\- Classification: 35%

\- Speed: 15%

\- Altitude: 15%

\- Proximity: 20%

\- Anomalies: 15%



\*\*Threat Levels:\*\*

\- 🔴 CRITICAL (75-100)

\- 🟠 HIGH (60-74)

\- 🟡 MEDIUM (40-59)

\- 🟢 LOW (0-39)



\## 🤖 Model Information



\### Classification Categories



| Classification | Description | Typical Use Case |

|---------------|-------------|------------------|

| \*\*HOSTILE\*\* | Confirmed enemy aircraft | Combat situations |

| \*\*SUSPECT\*\* | Unidentified with hostile indicators | Investigation required |

| \*\*NEUTRAL\*\* | Non-aligned aircraft | International airspace |

| \*\*FRIEND\*\* | Allied military aircraft | Friendly operations |

| \*\*ASSUMED FRIEND\*\* | Likely friendly, unconfirmed | Peacetime operations |

| \*\*CIVILIAN\*\* | Commercial/private aircraft | Air traffic control |



\### Input Features



The model processes the following features:

\- Altitude (feet)

\- Speed (knots)

\- Radar Cross Section (m²)

\- Electronic Signature (IFF response)

\- Flight Profile (maneuver patterns)

\- Weather Conditions

\- Thermal Signature



\### Model Performance



\- \*\*Accuracy\*\*: 94.7% (on validation set)

\- \*\*Inference Time\*\*: <50ms per classification

\- \*\*Model Type\*\*: Ensemble classifier (Random Forest/Gradient Boosting)



\## 📸 Screenshots



\### Main Dashboard

\*Single Track Analysis interface with real-time classification\*



\### Multi-Track Monitoring

\*Live tracking of multiple aircraft with conflict detection\*



\### 3D Track History

\*Interactive flight path visualization\*



\### Threat Matrix

\*Comprehensive threat assessment dashboard\*



\## 🛠️ Configuration



\### System Settings



The application supports the following configuration options:



```python

\# Conflict Detection Thresholds

HORIZONTAL\_SEPARATION\_MIN = 5.0  # Nautical Miles

VERTICAL\_SEPARATION\_MIN = 1000   # Feet



\# Threat Assessment Weights

THREAT\_WEIGHTS = {

&nbsp;   'classification': 0.35,

&nbsp;   'speed': 0.15,

&nbsp;   'altitude': 0.15,

&nbsp;   'proximity': 0.20,

&nbsp;   'anomalies': 0.15

}

```



\## 📊 Data Format



\### Track Data Structure



```python

{

&nbsp;   'track\_id': 'TRACK\_0001',

&nbsp;   'latitude': 51.5074,

&nbsp;   'longitude': -0.1278,

&nbsp;   'altitude': 35000,

&nbsp;   'speed': 450,

&nbsp;   'classification': 'CIVILIAN',

&nbsp;   'first\_seen': datetime,

&nbsp;   'last\_updated': datetime,

&nbsp;   'history': \[...]

}

```



\## 🤝 Contributing



Contributions are welcome! Please follow these steps:



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\### Development Guidelines



\- Follow PEP 8 style guide

\- Add unit tests for new features

\- Update documentation

\- Ensure all tests pass before submitting



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 🔒 Security \& Disclaimer



\*\*IMPORTANT\*\*: This is a demonstration/educational project. For production defense systems:

\- Implement proper authentication and authorization

\- Use encrypted communications

\- Follow military-grade security protocols

\- Comply with relevant defense regulations



This system is for educational and research purposes only. Not intended for actual military deployment without proper certification and security hardening.



\## 📞 Contact



\*\*Project Maintainer:\*\* Atamer Erkal



\- GitHub: \[@AtamerErkal](https://github.com/AtamerErkal)

\- Project Link: \[https://github.com/AtamerErkal/VANGUARD\_Project](https://github.com/AtamerErkal/VANGUARD\_Project)



\## 🙏 Acknowledgments



\- Streamlit for the web framework

\- Plotly for interactive visualizations

\- scikit-learn for machine learning capabilities

\- The open-source community



\## 🗓️ Version History



\- \*\*v3.0\*\* (Current) - Production build with advanced analytics

\- \*\*v2.0\*\* - Multi-track monitoring and conflict detection

\- \*\*v1.0\*\* - Initial release with single track classification



---



\*\*🛡️ VANGUARD AI System\*\* | Advanced Air Defense Intelligence Platform | © 2025



\*Built with ❤️ for enhanced aviation security\*

