⚡ StormSentinel AI
Multi-hazard weather risk intelligence for the United States — wildfire, tornado, hail, thunderstorm wind, flash flood, extreme heat, and drought risk, predicted from live weather data by a 7-head multi-task neural network.
Search any US location and get a real-time composite threat score plus a per-hazard breakdown with the specific conditions driving each risk.
Live Demo
🔗 [Add your Streamlit Cloud URL here after deployment]
Features
Free-text location search — type any US city, powered by Open-Meteo's geocoding API
7 independent hazard predictions from a single shared neural network
Composite threat index (0–100) with a 6-level severity scale (Minimal → Extreme)
Live weather-driven inference — pulls a 21-day rolling weather window at request time, no stale data
Transparent extrapolation warnings — flags when a searched location falls outside the model's core training region instead of silently guessing
How It Works
User searches a location (or picks one of the 32 cities used in training)
App fetches a 21-day weather history from Open-Meteo's archive API and the latest state-level PDSI drought index from NOAA
The same feature engineering pipeline used in training (dewpoint, rolling humidity/precipitation windows, wind-humidity interaction, cyclical seasonality encoding, etc.) is applied to the live data
StormSentinelNet — a shared ResNet-style trunk (BatchNorm + GELU + residual blocks) branching into 7 hazard-specific heads — outputs a probability per hazard
Probabilities are converted to 0–100 risk scores and rendered with hazard-specific contextual factors
Model Architecture
Code
Trained with AdamW + OneCycleLR + mixed precision (AMP), early stopping on mean validation AUC across all 7 heads, per-head pos_weight to handle class imbalance (tornado imbalance ratio is by far the most extreme — its head also gets extra hidden capacity to compensate).
Test Set Performance (2024, held out)
Hazard
AUC
Best F1
Drought
0.998
0.971
Heat
0.960
0.656
Flash Flood
0.913
0.658
Hail
0.906
0.558
Thunderstorm Wind
0.905
0.623
Tornado
0.884
0.393
Wildfire
0.809
0.518
(Update this table with your actual final numbers before publishing.)
Honest Limitations (Model Card)
Being upfront about scope matters more than the headline metrics — here's what these numbers do and don't mean:
"Wildfire" means vegetation fire activity, not narrowly uncontrolled ignition. Labels come from NASA FIRMS thermal detections filtered to vegetation fires (type==0, confidence≥30). This includes wildfires, escaped burns, and prescribed/agricultural fire — MODIS satellite data can't cleanly separate these categories, and neither can most open fire-risk products. The features that predict this (low humidity, high wind, prolonged dryness) genuinely track fire risk regardless of ignition source, since even prescribed burns are scheduled around similar weather windows.
Drought's near-perfect AUC (0.998) reflects persistence, not day-ahead prediction. PDSI is published monthly, so every day within a city-month shares one label. The model is largely learning "this region, this season = drought," which is a legitimate and useful signal, but it isn't forecasting drought onset the way the other 6 hazards forecast event days.
Tornado has the weakest F1 despite a respectable AUC. Tornado events are extremely sparse at city-day resolution, and the model works from surface weather features only — no true CAPE, wind shear profiles, or radar data. Treat tornado output as a coarse favorability signal, not a substitute for NOAA Storm Prediction Center guidance.
Training footprint is 32 cities across 15 states (CA, OR, WA, AZ, NV, OK, KS, TX, NE, IA, FL, GA, IL, CO, MN). The app allows searching any US location, but predictions outside this footprint are extrapolating beyond what the model has seen — the app flags this explicitly rather than hiding it.
No real atmospheric pressure data. Open-Meteo's archive API only exposes pressure hourly, not as a daily aggregate, so storm-relevant features rely on a derived dewpoint/instability proxy instead of true pressure tendency.
Data Sources
Source
Used For
Open-Meteo Archive API
Daily historical + rolling weather features
NASA FIRMS
Wildfire/vegetation fire detections (MODIS)
NOAA Storm Events Database
Tornado, hail, thunderstorm wind, flash flood, excessive heat labels
NOAA Climate Divisional Database
Palmer Drought Severity Index (PDSI)
All sources are free and require no paid API keys (FIRMS requires a free instant-signup key).
Tech Stack
Model: PyTorch (multi-task deep learning, GPU-trained on Kaggle)
App: Streamlit
Data: Open-Meteo, NASA FIRMS, NOAA Storm Events, NOAA PDSI
Feature engineering: pandas, NumPy, scikit-learn (StandardScaler)
Running Locally
Bash
Requires stormsentinel_model.pt, feature_scaler.pkl, and feature_columns.json in the repo root (included).
Project Background
StormSentinel AI is part of a portfolio of applied AI/data science projects, alongside FloodGuard (flood risk intelligence) and SkyMind (AI weather forecasting). Built end-to-end: data collection across 4 independent public APIs, EDA-driven feature engineering validated with effect-size diagnostics, multi-task deep learning, and full-stack deployment.
Author
Your Name: Daniel William
[LinkedIn] · [GitHub] · [Portfolio site]
