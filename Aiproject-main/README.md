﻿# TORCS AI Racing Controller

This project implements an AI controller for the TORCS racing simulator using supervised learning from telemetry data.

####################################################################################### Recent Changes
### Data Analysis
- ✅ Completed analysis of 450,000 telemetry records
- ✅ Generated comprehensive data visualizations
- ✅ Identified key metrics and patterns

### Preprocessing
- ✅ Implemented action threshold verification
- ✅ Adjusted steering thresholds (0.05/-0.05) for better balance
- ✅ Optimized gear change thresholds (RPM > 6500 for upshift)
- ✅ Implemented feature scaling and data splitting

### Action Definitions
- ✅ Refined steering logic based on data analysis
- ✅ Improved gear change conditions
- ✅ Enhanced acceleration/braking rules
- ✅ Balanced action distributions

### Visualization
- ✅ Added speed and track position distributions
- ✅ Implemented gear and RPM analysis
- ✅ Created feature correlation heatmaps
- ✅ Generated threshold verification plots


## Data Collection and Analysis

### Telemetry Data
- Collected 450,000 records of driving data
- 16 features including speed, angle, track position, RPM, etc.
- No missing values detected
- Data includes both successful and challenging driving scenarios

### Key Metrics
- Average speed: 70.81 units
- Maximum speed: 226.48 units
- Gear distribution shows good coverage of all gears
- Track position range: [-2.87, 1.67]

## Preprocessing Pipeline

### Action Definitions
1. **Steering**:
   - Left turn: angle > 0.05 or trackPos < -0.5
   - Right turn: angle < -0.05 or trackPos > 0.5

2. **Gear Changes**:
   - Gear up: RPM > 6500 and current gear < 6
   - Gear down: RPM < 3000 and current gear > 1

3. **Acceleration/Braking**:
   - Accelerate: speedX > 0 and gear > 0
   - Brake: speedX < 0 or gear < 0

### Feature Engineering
- 15 input features selected for training
- Features normalized using StandardScaler
- 80/20 train/test split
- Binary classification for each action

## Action Distribution
- Steering actions balanced using adjusted thresholds
- Gear changes: 12,711 upshifts vs 4,525 downshifts
- Acceleration/braking based on speed and gear state

## Visualization
- Speed distribution
- Track position distribution
- Gear distribution
- RPM distribution
- Feature correlations
- Action threshold analysis

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

```

