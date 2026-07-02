# ============================================================================
# STORMSENTINEL AI — STREAMLIT APP
# Multi-hazard risk intelligence: wildfire, tornado, hail, thunderstorm wind,
# flash flood, extreme heat, drought.
#
# HOW TO RUN:
#   pip install streamlit torch joblib plotly folium streamlit-folium
#   streamlit run app.py
#
# Required files (from training pipeline):
#   stormsentinel_model.pt   — trained model weights
#   feature_scaler.pkl       — fitted StandardScaler
#   feature_columns.json     — feature column names + order
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import joblib
import json
import re
import io
import time
from datetime import datetime, timedelta


def st_html(html_string):
    """
    Renders an HTML string via st.markdown, safely.

    Streamlit's st.markdown() runs content through a Markdown parser before
    display. Markdown treats any line starting with 4+ spaces of indentation
    as a preformatted code block — which means multi-line HTML f-strings
    written with normal Python indentation (for readability) get partially
    rendered as literal escaped text instead of parsed HTML. Stripping each
    line's leading whitespace before rendering avoids that entirely, since
    HTML/CSS don't care about leading whitespace anyway.

    Defined here, immediately after imports, because the global CSS block
    below calls it before any other function definitions would normally
    appear — Python executes top-to-bottom, so this must exist first.
    """
    cleaned = "\n".join(line.lstrip() for line in html_string.split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StormSentinel AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Space+Mono:wght@400;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    background-color: #05080e !important;
    color: #dde4ee;
    font-family: 'Inter', sans-serif;
}
.stApp { background-color: #05080e; }

/* Header */
.ss-header {
    background: #07090f;
    border-bottom: 1px solid #0e1929;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 1.5rem -1rem;
}
.ss-logo-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #dde4ee;
}
.ss-logo-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 9px;
    letter-spacing: 3px;
    color: #2e4156;
}
.ss-live {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #2e4156;
    letter-spacing: 1px;
}
.ss-live-dot {
    color: #22C55E;
    animation: blink 2s step-end infinite;
}
@keyframes blink { 0%,49%{opacity:1} 50%,100%{opacity:.15} }

/* Composite threat block */
.threat-block {
    background: #07090f;
    border: 1px solid #0e1929;
    border-radius: 10px;
    padding: 20px 26px;
    margin-bottom: 16px;
}
.threat-index-num {
    font-family: 'Space Mono', monospace;
    font-size: 58px;
    font-weight: 700;
    line-height: 1;
}
.threat-label-tag {
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2.5px;
    padding: 7px 20px;
    border-radius: 6px;
}
.threat-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 9px;
    letter-spacing: 3px;
    color: #2e4156;
}

/* Hazard cards */
.hazard-card {
    background: #080e1c;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.hazard-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1.5px;
}
.hazard-score {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
}
.hazard-badge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 3px 9px;
    border-radius: 3px;
}
.factor-row {
    background: #04080f;
    border-radius: 4px;
    padding: 5px 9px;
    margin: 3px 0;
    display: flex;
    justify-content: space-between;
}
.factor-label {
    font-size: 10px;
    color: #5a7080;
}
.factor-value {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #8fa0b0;
}

/* Weather strip */
.wx-strip {
    background: #07090f;
    border: 1px solid #0e1929;
    border-radius: 8px;
    padding: 10px 20px;
    display: flex;
    gap: 28px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 10px;
}
.wx-item-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 9px;
    letter-spacing: 1.5px;
    color: #2e4156;
}
.wx-item-value {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    color: #8fa0b0;
    font-weight: 700;
}
.ss-footer {
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-size: 8.5px;
    letter-spacing: 2px;
    color: #1a2a3a;
    margin-top: 20px;
}
</style>
""")


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CITIES = [
    {"city": "Los Angeles",   "state": "CA", "lat": 34.0522,  "lon": -118.2437, "zone": "wildfire"},
    {"city": "Sacramento",    "state": "CA", "lat": 38.5816,  "lon": -121.4944, "zone": "wildfire"},
    {"city": "San Diego",     "state": "CA", "lat": 32.7157,  "lon": -117.1611, "zone": "wildfire"},
    {"city": "Portland",      "state": "OR", "lat": 45.5152,  "lon": -122.6784, "zone": "wildfire"},
    {"city": "Eugene",        "state": "OR", "lat": 44.0521,  "lon": -123.0868, "zone": "wildfire"},
    {"city": "Seattle",       "state": "WA", "lat": 47.6062,  "lon": -122.3321, "zone": "wildfire"},
    {"city": "Spokane",       "state": "WA", "lat": 47.6588,  "lon": -117.4260, "zone": "wildfire"},
    {"city": "Tucson",        "state": "AZ", "lat": 32.2226,  "lon": -110.9747, "zone": "wildfire"},
    {"city": "Reno",          "state": "NV", "lat": 39.5296,  "lon": -119.8138, "zone": "wildfire"},
    {"city": "Oklahoma City", "state": "OK", "lat": 35.4676,  "lon": -97.5164,  "zone": "storm"},
    {"city": "Tulsa",         "state": "OK", "lat": 36.1540,  "lon": -95.9928,  "zone": "storm"},
    {"city": "Wichita",       "state": "KS", "lat": 37.6872,  "lon": -97.3301,  "zone": "storm"},
    {"city": "Topeka",        "state": "KS", "lat": 39.0473,  "lon": -95.6752,  "zone": "storm"},
    {"city": "Dallas",        "state": "TX", "lat": 32.7767,  "lon": -96.7970,  "zone": "storm"},
    {"city": "Amarillo",      "state": "TX", "lat": 35.2220,  "lon": -101.8313, "zone": "storm"},
    {"city": "Omaha",         "state": "NE", "lat": 41.2565,  "lon": -95.9345,  "zone": "storm"},
    {"city": "Lincoln",       "state": "NE", "lat": 40.8136,  "lon": -96.7026,  "zone": "storm"},
    {"city": "Des Moines",    "state": "IA", "lat": 41.5868,  "lon": -93.6250,  "zone": "storm"},
    {"city": "Cedar Rapids",  "state": "IA", "lat": 41.9779,  "lon": -91.6656,  "zone": "storm"},
    {"city": "Phoenix",       "state": "AZ", "lat": 33.4484,  "lon": -112.0740, "zone": "heat"},
    {"city": "Las Vegas",     "state": "NV", "lat": 36.1699,  "lon": -115.1398, "zone": "heat"},
    {"city": "Houston",       "state": "TX", "lat": 29.7601,  "lon": -95.3701,  "zone": "heat"},
    {"city": "San Antonio",   "state": "TX", "lat": 29.4241,  "lon": -98.4936,  "zone": "heat"},
    {"city": "Austin",        "state": "TX", "lat": 30.2672,  "lon": -97.7431,  "zone": "heat"},
    {"city": "Miami",         "state": "FL", "lat": 25.7617,  "lon": -80.1918,  "zone": "heat"},
    {"city": "Orlando",       "state": "FL", "lat": 28.5384,  "lon": -81.3789,  "zone": "heat"},
    {"city": "Tampa",         "state": "FL", "lat": 27.9506,  "lon": -82.4572,  "zone": "heat"},
    {"city": "Atlanta",       "state": "GA", "lat": 33.7490,  "lon": -84.3880,  "zone": "heat"},
    {"city": "Savannah",      "state": "GA", "lat": 32.0809,  "lon": -81.0912,  "zone": "heat"},
    {"city": "Chicago",       "state": "IL", "lat": 41.8781,  "lon": -87.6298,  "zone": "control"},
    {"city": "Denver",        "state": "CO", "lat": 39.7392,  "lon": -104.9903, "zone": "control"},
    {"city": "Minneapolis",   "state": "MN", "lat": 44.9778,  "lon": -93.2650,  "zone": "control"},
]

# All 32 preset training cities are US locations by construction — set this
# explicitly rather than relying on a .get() default, since geocoded search
# results carry an explicit is_us field and these should match that shape.
for _c in CITIES:
    _c["is_us"] = True
    _c["country_code"] = "US"
    _c["country_name"] = "United States"

HAZARDS = [
    # us_only=False: physically-grounded features (humidity/temp/wind) that
    # transfer reasonably across climates, even though training data was US.
    # us_only=True: trained exclusively on NOAA Storm Events / NOAA PDSI —
    # both US-only government data sources with zero equivalent signal
    # anywhere else in the world. These heads are extrapolating blind
    # outside the US, not just "less confident."
    {"key": "wildfire",          "label": "WILDFIRE",          "icon": "🔥", "color": "#F97316", "us_only": False},
    {"key": "tornado",           "label": "TORNADO",           "icon": "🌪️", "color": "#8B5CF6", "us_only": True},
    {"key": "hail",              "label": "HAIL",              "icon": "🧊", "color": "#22C55E", "us_only": True},
    {"key": "thunderstorm_wind", "label": "THUNDERSTORM WIND", "icon": "⚡", "color": "#3B82F6", "us_only": True},
    {"key": "flash_flood",       "label": "FLASH FLOOD",       "icon": "🌊", "color": "#06B6D4", "us_only": True},
    {"key": "heat",              "label": "EXTREME HEAT",      "icon": "🌡️", "color": "#EF4444", "us_only": False},
    {"key": "drought",           "label": "DROUGHT",           "icon": "☀️", "color": "#EAB308", "us_only": True},
]

THREAT_LEVELS = [
    {"label": "MINIMAL",  "max": 15,  "color": "#22C55E"},
    {"label": "LOW",      "max": 30,  "color": "#84CC16"},
    {"label": "MODERATE", "max": 50,  "color": "#EAB308"},
    {"label": "ELEVATED", "max": 65,  "color": "#F97316"},
    {"label": "HIGH",     "max": 80,  "color": "#EF4444"},
    {"label": "EXTREME",  "max": 100, "color": "#DC2626"},
]

STATE_CODES = {
    "002": "AZ", "004": "CA", "005": "CO", "008": "FL", "009": "GA",
    "011": "IL", "013": "IA", "014": "KS", "021": "MN", "025": "NE",
    "026": "NV", "034": "OK", "035": "OR", "041": "TX", "045": "WA",
}

# Full US state name -> abbreviation. Needed because a user-searched location
# can be anywhere in the US, not just the 15 states the model trained on —
# geocoding returns full state names, and we need abbreviations to check
# against the model's trained state dummy columns.
US_STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

# States the model actually trained on — used to warn users searching
# outside this footprint that predictions are extrapolating
TRAINED_STATES = set(STATE_CODES.values())

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "relative_humidity_2m_mean",
    "et0_fao_evapotranspiration", "shortwave_radiation_sum",
]

LABEL_COLS = [
    "wildfire_label", "tornado_label", "hail_label",
    "thunderstorm_wind_label", "flash_flood_label",
    "heat_label", "drought_label",
]
HEAD_ORDER = [c.replace("_label", "") for c in LABEL_COLS]

# Best thresholds from test_metrics.json — update after training
BEST_THRESHOLDS = {
    "wildfire":          0.70,
    "tornado":           0.80,
    "hail":              0.80,
    "thunderstorm_wind": 0.75,
    "flash_flood":       0.80,
    "heat":              0.75,
    "drought":           0.75,
}

# Tiered safety guidance per hazard, shown only at MODERATE severity or
# above — below that there's nothing actionable worth surfacing. Content
# follows standard public safety guidance patterns (NWS, CDC, Ready.gov)
# rather than anything hazard-model-specific; these are the same practical
# steps official sources recommend at each severity level.
RECOMMENDATIONS = {
    "wildfire": {
        "moderate": ["Monitor local air quality and fire updates", "Know your evacuation routes in advance"],
        "elevated": ["Avoid any outdoor burning", "Keep vehicles fueled with a go-bag packed", "Watch for local evacuation notices"],
        "high":     ["Prepare to evacuate on short notice", "Clear dry vegetation near structures", "Keep N95 masks on hand for smoke exposure"],
        "extreme":  ["Follow local evacuation orders immediately", "Avoid the area if at all possible", "Keep windows closed; use air purifiers indoors"],
    },
    "tornado": {
        "moderate": ["Review your household's tornado safety plan", "Keep a charged weather radio or phone alerts on"],
        "elevated": ["Stay alert to weather alerts through the day", "Identify your safe room — interior, lowest floor, no windows"],
        "high":     ["Postpone outdoor plans", "Keep shoes and a flashlight near your safe room"],
        "extreme":  ["Take shelter immediately if a warning is issued", "Stay away from windows and mobile homes"],
    },
    "hail": {
        "moderate": ["Move vehicles under cover if possible"],
        "elevated": ["Park vehicles in a garage or covered area", "Secure outdoor furniture and equipment"],
        "high":     ["Stay indoors, away from windows and skylights", "Avoid driving until the storm passes"],
        "extreme":  ["Take shelter immediately", "Do not go outside until conditions clear"],
    },
    "thunderstorm_wind": {
        "moderate": ["Secure loose outdoor items"],
        "elevated": ["Bring in patio furniture and anything that could become a projectile"],
        "high":     ["Avoid parking under trees or power lines", "Stay indoors"],
        "extreme":  ["Stay away from windows", "Be prepared for possible power outages"],
    },
    "flash_flood": {
        "moderate": ["Avoid low-lying areas after heavy rain"],
        "elevated": ["Know your route to higher ground", "Avoid spending time in basements during heavy rain"],
        "high":     ["Never drive through flooded roads — turn around, don't drown", "Move valuables to higher ground"],
        "extreme":  ["Evacuate low-lying areas immediately if instructed", "Never walk or drive through moving floodwater"],
    },
    "heat": {
        "moderate": ["Stay hydrated", "Limit strenuous outdoor activity during peak afternoon hours"],
        "elevated": ["Check on elderly neighbors and relatives", "Never leave children or pets in parked vehicles"],
        "high":     ["Limit outdoor exposure between 11am–4pm", "Seek air-conditioned spaces when possible", "Wear light, breathable clothing"],
        "extreme":  ["Avoid outdoor activity entirely if possible", "Watch for signs of heat exhaustion or heat stroke (confusion, rapid pulse, hot dry skin) and seek medical help immediately if they occur"],
    },
    "drought": {
        "moderate": ["Be mindful of water usage", "Check for local watering restrictions"],
        "elevated": ["Reduce non-essential water use — lawn watering, car washing", "Watch for local water advisories"],
        "high":     ["Follow local water conservation mandates", "Be alert to elevated wildfire risk, which often accompanies drought"],
        "extreme":  ["Comply with any mandatory water restrictions", "Monitor local emergency advisories closely"],
    },
}

_LEVEL_TO_TIER = {"MODERATE": "moderate", "ELEVATED": "elevated", "HIGH": "high", "EXTREME": "extreme"}


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def get_threat_level(score):
    for t in THREAT_LEVELS:
        if score <= t["max"]:
            return t
    return THREAT_LEVELS[-1]


def prob_to_score(prob):
    """Convert model probability to 0-100 display score."""
    return min(100, int(prob * 100))


# ── MODEL ─────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim); self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        return self.act(self.bn2(self.fc2(out)) + x)

class HazardHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class StormSentinelNet(nn.Module):
    def __init__(self, input_dim, trunk_dim=256, mid_dim=128, dropout=0.3):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, trunk_dim), nn.BatchNorm1d(trunk_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.res1 = ResBlock(trunk_dim, dropout)
        self.res2 = ResBlock(trunk_dim, dropout)
        self.mid_block = nn.Sequential(
            nn.Linear(trunk_dim, mid_dim), nn.BatchNorm1d(mid_dim),
            nn.GELU(), nn.Dropout(dropout * 0.6),
        )
        self.res3 = ResBlock(mid_dim, dropout * 0.6)
        self.head_wildfire          = HazardHead(mid_dim, 64)
        self.head_tornado           = HazardHead(mid_dim, 128)
        self.head_hail              = HazardHead(mid_dim, 64)
        self.head_thunderstorm_wind = HazardHead(mid_dim, 64)
        self.head_flash_flood       = HazardHead(mid_dim, 64)
        self.head_heat              = HazardHead(mid_dim, 64)
        self.head_drought           = HazardHead(mid_dim, 64)
    def forward(self, x):
        x = self.res2(self.res1(self.input_block(x)))
        x = self.res3(self.mid_block(x))
        return {
            "wildfire":          self.head_wildfire(x),
            "tornado":           self.head_tornado(x),
            "hail":              self.head_hail(x),
            "thunderstorm_wind": self.head_thunderstorm_wind(x),
            "flash_flood":       self.head_flash_flood(x),
            "heat":              self.head_heat(x),
            "drought":           self.head_drought(x),
        }


@st.cache_resource
def load_model_and_scaler():
    with open("feature_columns.json") as f:
        meta = json.load(f)
    scaler = joblib.load("feature_scaler.pkl")
    model = StormSentinelNet(len(meta["feature_columns"]))
    model.load_state_dict(torch.load("stormsentinel_model.pt", map_location="cpu"))
    model.eval()
    return model, scaler, meta


# ── GEOCODING ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def geocode_location(query):
    """
    Converts a free-text location into a list of matching candidates ANYWHERE
    in the world, using Open-Meteo's free geocoding API.

    Ambiguous names (e.g. "Paris" -> Paris FR, Paris TX, Paris KY, Paris IL,
    Paris ME...) return multiple entries sorted by population — largest
    first, purely to make the list easier to scan, never auto-picked.
    Disambiguation applies globally now, not just within the US, since
    cross-country name collisions are exactly as real as domestic ones.

    Non-US results are valid candidates. The app still runs full inference
    on them — the per-hazard "not validated outside US" distinction is
    handled at render time, not by blocking the search here.
    """
    if not query or not query.strip():
        return []

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query.strip(), "count": 20, "language": "en", "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        return []

    if not results:
        return []

    candidates = []
    for res in results:
        country_code = res.get("country_code", "")
        country_name = res.get("country", country_code)
        admin1 = res.get("admin1", "")
        admin2 = res.get("admin2", "")  # county, when available — helps disambiguate
        population = res.get("population")
        is_us = country_code == "US"

        # "state" field: US 2-letter abbreviation for US results, or the
        # region/province name for everywhere else — used both for display
        # and (for US results) matching against the model's trained state
        # dummy columns in feature engineering.
        state_label = US_STATE_ABBREV.get(admin1, admin1 or "??") if is_us else (admin1 or country_name)

        label = f"{res.get('name', query)}, {state_label}"
        if is_us and admin2:
            label += f" ({admin2})"
        elif not is_us:
            label += f", {country_name}"
        if population:
            label += f" — pop. {population:,}"

        display_name = f"{res.get('name', query)}, {state_label}"
        if not is_us:
            display_name += f", {country_name}"

        candidates.append({
            "city": res.get("name", query),
            "state": state_label,
            "country_code": country_code,
            "country_name": country_name,
            "lat": res.get("latitude"),
            "lon": res.get("longitude"),
            "is_us": is_us,
            "zone": "custom",
            "display_name": display_name,
            "disambiguation_label": label,
            "population": population or 0,
        })

    candidates.sort(key=lambda c: c["population"], reverse=True)
    return candidates


# ── DATA FETCHING ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def fetch_weather_lookback(lat, lon, days=395):
    """
    Fetches roughly 13 months of daily weather from Open-Meteo archive API.

    Extended from a 21-day window (Jul 2026) after live testing during an
    active heat wave revealed a self-referential baseline bug: with only 21
    days of history, an extended heat wave dominates its own comparison
    window, making temp_anomaly collapse toward zero exactly when it should
    spike. ~13 months lets engineer_features() compare today against the
    same calendar month from roughly a year ago — a genuine seasonal
    baseline that isn't contaminated by the current event. Short rolling
    features (heat_streak, humidity_7d_avg, etc.) still correctly use only
    the trailing ~21 days of this larger dataframe.
    """
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(start_date), "end_date": str(end_date),
        "daily": ",".join(DAILY_VARS), "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(ttl=21600)
def fetch_pdsi(state):
    """Fetches current PDSI for a state from NOAA's climdiv statewide file."""
    try:
        CLIMDIV_BASE = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv"
        r = requests.get(f"{CLIMDIV_BASE}/", timeout=15)
        r.raise_for_status()
        match = re.search(r"climdiv-pdsist-v\d+\.\d+\.\d+-\d{8}", r.text)
        if not match:
            return 0.0
        r2 = requests.get(f"{CLIMDIV_BASE}/{match.group(0)}", timeout=30)
        r2.raise_for_status()
        reverse_codes = {v: k for k, v in STATE_CODES.items()}
        state_code = reverse_codes.get(state)
        if not state_code:
            return 0.0
        now = datetime.utcnow()
        for line in r2.text.splitlines():
            if len(line) < 94:
                continue
            if line[0:3] == state_code and line[4:6] == "05":
                year = int(line[6:10])
                if year == now.year:
                    month_idx = now.month - 1
                    start = 10 + month_idx * 7
                    val_str = line[start:start+7].strip()
                    try:
                        val = float(val_str)
                        return val if val > -99 else 0.0
                    except ValueError:
                        return 0.0
        return 0.0
    except Exception:
        return 0.0


# ── FEATURE ENGINEERING (mirrors 07_feature_engineering.py exactly) ───────────

def engineer_features(wx_df, pdsi_val, city_info, meta):
    """
    Replicates the full feature engineering pipeline from training.
    Takes 21 days of weather history, returns a single feature vector
    for the most recent day (the one we're predicting).
    """
    df = wx_df.copy().sort_values("time").reset_index(drop=True)

    # ── Dewpoint (Magnus-Tetens) ───────────────────────────────────────────
    a, b = 17.27, 237.7
    T  = df["temperature_2m_mean"]
    RH = df["relative_humidity_2m_mean"].clip(lower=1, upper=100)
    alpha = np.log(RH / 100) + (a * T) / (b + T)
    df["dewpoint_c"] = (b * alpha) / (a - alpha)
    df["dewpoint_depression"] = df["temperature_2m_max"] - df["dewpoint_c"]

    # ── Wind direction cyclical encoding ──────────────────────────────────
    wind_rad = np.radians(df["winddirection_10m_dominant"])
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    # ── Gustiness ratio ───────────────────────────────────────────────────
    df["wind_gust_ratio"] = df["windgusts_10m_max"] / (df["windspeed_10m_max"] + 1.0)

    # ── Cyclical month encoding ───────────────────────────────────────────
    month = df["time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── Rolling / lag features ────────────────────────────────────────────
    df["precip_7d_sum"]  = df["precipitation_sum"].rolling(7,  min_periods=1).sum()
    df["precip_14d_sum"] = df["precipitation_sum"].rolling(14, min_periods=1).sum()
    df["et0_7d_sum"]     = df["et0_fao_evapotranspiration"].rolling(7, min_periods=1).sum()
    df["temp_change_1d"] = df["temperature_2m_max"].diff().fillna(0)

    # Humidity rolling averages (from EDA diagnostics)
    df["humidity_3d_avg"] = df["relative_humidity_2m_mean"].rolling(3, min_periods=1).mean()
    df["humidity_7d_avg"] = df["relative_humidity_2m_mean"].rolling(7, min_periods=1).mean()

    # Days since rain (cumsum-reset pattern from EDA)
    is_dry = df["precipitation_sum"] < 1
    df["days_since_rain"] = is_dry.groupby((~is_dry).cumsum()).cumsum()

    # Wind-humidity interaction (EDA's best engineered wildfire feature)
    df["wind_humidity_interaction"] = (
        df["windspeed_10m_max"] * (100 - df["relative_humidity_2m_mean"]) / 100
    )

    # Heat streak
    is_hot = df["apparent_temperature_max"] >= 35
    df["heat_streak"] = is_hot.groupby((~is_hot).cumsum()).cumsum()

    # Temperature anomaly (vs this city-month's climatological baseline)
    #
    # BUG FIX (Jul 2026): previously computed as today's temp minus the mean
    # of the same short window being analyzed — during an extended heat
    # wave, that self-referential baseline gets dragged up by the very
    # event we're trying to detect, so the anomaly collapses toward zero
    # exactly when it should spike. Confirmed via live testing: Chicago and
    # Dallas both showed near-zero heat scores during a verified, ongoing
    # heat wave (apparent temps ~38°C, heat streaks of 3-12 days).
    #
    # Fix: use days from the SAME calendar month, excluding the most recent
    # 21 days (the active event window), as the baseline instead. With the
    # ~13-month fetch, this mostly draws from ~last year's same month — a
    # genuine seasonal normal uncontaminated by whatever is happening right
    # now. Falls back to the old (imperfect) method only if there isn't
    # enough historical same-month data to compute a reliable baseline —
    # e.g., a location with very limited archive history available.
    current_month = df["time"].dt.month.iloc[-1]
    cutoff_date = df["time"].max() - pd.Timedelta(days=21)
    baseline_pool = df[(df["time"].dt.month == current_month) & (df["time"] < cutoff_date)]

    if len(baseline_pool) >= 5:
        baseline_temp = baseline_pool["temperature_2m_max"].mean()
    else:
        baseline_temp = df["temperature_2m_max"].mean()  # fallback: sparse history

    df["temp_anomaly"] = df["temperature_2m_max"] - baseline_temp

    # ── PDSI (current + lagged) ───────────────────────────────────────────
    df["pdsi"] = pdsi_val
    df["pdsi_change_1m"] = 0.0  # no prior month at inference time — zero-filled

    # ── Drop redundant cols (same as training) ────────────────────────────
    df = df.drop(columns=[
        "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_mean", "winddirection_10m_dominant",
        "time",
    ], errors="ignore")

    # ── State one-hot dummies ─────────────────────────────────────────────
    state_dummy_cols = meta.get("state_dummy_columns", [])
    for col in state_dummy_cols:
        df[col] = 0
    city_state_col = f"state_{city_info['state']}"
    if city_state_col in state_dummy_cols:
        df[city_state_col] = 1

    # ── Align to training feature order, take last row ────────────────────
    numeric_cols   = meta["numeric_columns"]
    all_feat_cols  = meta["feature_columns"]

    # Fill any missing columns with 0
    for col in all_feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    last_row = df.iloc[[-1]][all_feat_cols].fillna(0)
    return last_row, df.iloc[-1]   # feature row + raw weather row for display


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(feature_row, model, scaler, meta):
    numeric_cols = meta["numeric_columns"]
    feat = feature_row.copy()
    feat[numeric_cols] = scaler.transform(feat[numeric_cols])
    x = torch.tensor(feat.values.astype(np.float32))
    with torch.no_grad():
        outputs = model(x)
    probs = {name: torch.sigmoid(outputs[name]).item() for name in HEAD_ORDER}
    scores = {name: prob_to_score(probs[name]) for name in HEAD_ORDER}
    return probs, scores


# ── UI RENDERING ──────────────────────────────────────────────────────────────

def render_header(city_name, state, utc_time):
    st_html(f"""
    <div class="ss-header">
        <div style="display:flex;align-items:center;gap:12px">
            <div style="width:36px;height:36px;border-radius:8px;
                background:linear-gradient(135deg,#F97316,#8B5CF6);
                display:flex;align-items:center;justify-content:center;font-size:18px">⚡</div>
            <div>
                <div class="ss-logo-text">STORMSENTINEL AI</div>
                <div class="ss-logo-sub">MULTI-HAZARD · AI RISK INTELLIGENCE</div>
            </div>
        </div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:20px;
            font-weight:700;letter-spacing:2px;color:#dde4ee">
            {city_name.upper()}, {state}
        </div>
        <div class="ss-live">
            <span class="ss-live-dot">●</span> {utc_time} UTC
        </div>
    </div>
    """)


def render_composite(scores):
    composite = int(np.mean(list(scores.values())))
    lvl = get_threat_level(composite)
    drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = drivers[0][0].replace("_", " ").upper()

    bar_html = ""
    for t in THREAT_LEVELS:
        active = t["label"] == lvl["label"]
        bar_html += f"""
        <div style="flex:1;height:8px;border-radius:2px;
            background:{''+t['color'] if active else '#0c1520'};
            box-shadow:{'0 0 12px '+t['color']+'90' if active else 'none'};
            transition:all .4s ease"></div>"""

    labels_html = ""
    for t in THREAT_LEVELS:
        active = t["label"] == lvl["label"]
        labels_html += f"""<span style="font-size:8px;font-family:'Rajdhani',sans-serif;
            font-weight:{'700' if active else '400'};
            color:{''+t['color'] if active else '#1e2d40'}">{t['label']}</span>"""

    st_html(f"""
    <div class="threat-block">
        <div style="display:flex;align-items:center;gap:32px;flex-wrap:wrap">
            <div>
                <div class="threat-sub">COMPOSITE THREAT INDEX</div>
                <div class="threat-index-num" style="color:{lvl['color']};
                    text-shadow:0 0 28px {lvl['color']}55">{composite}</div>
                <span style="font-size:14px;color:#2e4156;font-family:'Space Mono',monospace">/100</span>
            </div>
            <div style="flex:1;min-width:180px">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">{labels_html}</div>
                <div style="display:flex;gap:3px">{bar_html}</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end">
                <div style="padding:8px 20px;border-radius:6px;
                    background:{lvl['color']}15;border:1px solid {lvl['color']}40;text-align:center">
                    <div class="threat-label-tag" style="color:{lvl['color']}">{lvl['label']}</div>
                    <div class="threat-sub">THREAT LEVEL</div>
                </div>
                <div style="font-size:9px;color:#2e4156;font-family:'Rajdhani',sans-serif;letter-spacing:1px">
                    PRIMARY DRIVER: <span style="color:#F97316;font-weight:700">{primary}</span>
                </div>
            </div>
        </div>
    </div>
    """)
    return composite


def render_hazard_card(h, score, wx_row, is_us=True):
    color = h["color"]
    lvl   = get_threat_level(score)
    is_high = score >= 65
    border  = f"1px solid {color}45" if is_high else "1px solid #111d2e"
    shadow  = f"0 0 30px {color}12" if is_high else "none"
    top_bar = f"linear-gradient(90deg,transparent,{color}{'ff' if is_high else '40'},transparent)"
    pct     = score / 100

    # Strong, hard-to-miss warning for the 5 hazards trained exclusively on
    # US-only data (NOAA Storm Events, NOAA PDSI) when shown for a non-US
    # location. The score is still displayed — just clearly flagged as
    # extrapolating with zero real grounding, not merely "lower confidence."
    show_unvalidated = h.get("us_only", False) and not is_us
    warning_banner = ""
    if show_unvalidated:
        warning_banner = """
        <div style="background:#78350f;border:1px solid #f59e0b;border-radius:6px;
            padding:7px 10px;margin-bottom:10px;display:flex;align-items:flex-start;gap:6px">
            <span style="font-size:13px;line-height:1.2">⚠️</span>
            <span style="font-size:9px;color:#fbbf24;font-family:'Rajdhani',sans-serif;
                font-weight:700;letter-spacing:0.3px;line-height:1.35">
                NOT VALIDATED OUTSIDE US — trained exclusively on US storm/drought data
            </span>
        </div>"""

    # SVG gauge
    r = 34; circ = 2 * 3.14159 * r
    offset = circ * (1 - pct)
    gauge_svg = f"""
    <svg width="88" height="88" viewBox="0 0 88 88">
        <circle cx="44" cy="44" r="{r}" fill="none" stroke="#111d2e" stroke-width="7"/>
        <circle cx="44" cy="44" r="{r}" fill="none" stroke="{color}" stroke-width="7"
            stroke-linecap="round" stroke-dasharray="{circ:.1f}"
            stroke-dashoffset="{offset:.1f}" transform="rotate(-90 44 44)"/>
        <text x="44" y="41" text-anchor="middle" fill="{color}"
            font-size="20" font-weight="700" font-family="Space Mono, monospace">{score}</text>
        <text x="44" y="55" text-anchor="middle" fill="#3d5068"
            font-size="9" font-family="Rajdhani, sans-serif">/100</text>
    </svg>"""

    # Contextual weather factors per hazard
    factors = []
    key = h["key"]
    if key == "wildfire":
        factors = [
            ("Humidity",    f"{wx_row.get('relative_humidity_2m_mean', 0):.0f}%",
             wx_row.get('relative_humidity_2m_mean', 100) < 30),
            ("Wind Speed",  f"{wx_row.get('windspeed_10m_max', 0):.0f} km/h",
             wx_row.get('windspeed_10m_max', 0) > 30),
            ("Days Dry",    f"{wx_row.get('days_since_rain', 0):.0f}",
             wx_row.get('days_since_rain', 0) > 7),
        ]
    elif key == "tornado":
        factors = [
            ("Dewpt Depress.", f"{wx_row.get('dewpoint_depression', 0):.1f}°C",
             wx_row.get('dewpoint_depression', 0) > 15),
            ("Wind Gusts",  f"{wx_row.get('windgusts_10m_max', 0):.0f} km/h",
             wx_row.get('windgusts_10m_max', 0) > 60),
            ("Gust Ratio",  f"{wx_row.get('wind_gust_ratio', 0):.2f}",
             wx_row.get('wind_gust_ratio', 0) > 2.0),
        ]
    elif key == "hail":
        factors = [
            ("Max Temp",    f"{wx_row.get('temperature_2m_max', 0):.1f}°C",
             wx_row.get('temperature_2m_max', 0) > 28),
            ("Wind Gusts",  f"{wx_row.get('windgusts_10m_max', 0):.0f} km/h",
             wx_row.get('windgusts_10m_max', 0) > 50),
            ("Dewpt Depress.", f"{wx_row.get('dewpoint_depression', 0):.1f}°C",
             wx_row.get('dewpoint_depression', 0) > 12),
        ]
    elif key == "thunderstorm_wind":
        factors = [
            ("Wind Gusts",  f"{wx_row.get('windgusts_10m_max', 0):.0f} km/h",
             wx_row.get('windgusts_10m_max', 0) > 55),
            ("Gust Ratio",  f"{wx_row.get('wind_gust_ratio', 0):.2f}",
             wx_row.get('wind_gust_ratio', 0) > 1.8),
            ("Humidity",    f"{wx_row.get('relative_humidity_2m_mean', 0):.0f}%",
             wx_row.get('relative_humidity_2m_mean', 0) > 60),
        ]
    elif key == "flash_flood":
        factors = [
            ("Precip Today", f"{wx_row.get('precipitation_sum', 0):.1f} mm",
             wx_row.get('precipitation_sum', 0) > 20),
            ("7-Day Precip", f"{wx_row.get('precip_7d_sum', 0):.1f} mm",
             wx_row.get('precip_7d_sum', 0) > 50),
            ("Rain Sum",    f"{wx_row.get('rain_sum', 0):.1f} mm",
             wx_row.get('rain_sum', 0) > 15),
        ]
    elif key == "heat":
        factors = [
            ("Apparent Max", f"{wx_row.get('apparent_temperature_max', 0):.1f}°C",
             wx_row.get('apparent_temperature_max', 0) >= 40),
            ("Heat Streak",  f"{wx_row.get('heat_streak', 0):.0f} days",
             wx_row.get('heat_streak', 0) > 3),
            ("Humidity",    f"{wx_row.get('relative_humidity_2m_mean', 0):.0f}%",
             wx_row.get('relative_humidity_2m_mean', 0) > 60),
        ]
    elif key == "drought":
        factors = [
            ("PDSI",        f"{wx_row.get('pdsi', 0):.2f}",
             wx_row.get('pdsi', 0) < -2),
            ("Days Dry",    f"{wx_row.get('days_since_rain', 0):.0f}",
             wx_row.get('days_since_rain', 0) > 14),
            ("ET₀ 7-Day",   f"{wx_row.get('et0_7d_sum', 0):.1f} mm",
             wx_row.get('et0_7d_sum', 0) > 30),
        ]

    factors_html = ""
    for fn, fv, fhi in factors:
        hi_color = color if fhi else "#111d2e"
        hi_val   = color if fhi else "#8fa0b0"
        factors_html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
            padding:5px 9px;background:#04080f;border-radius:4px;
            border-left:2px solid {hi_color};margin-top:4px">
            <span style="font-size:10px;color:#5a7080;font-family:Inter,sans-serif">{fn}</span>
            <span style="font-size:11px;font-family:Space Mono,monospace;
                color:{hi_val};font-weight:{'700' if fhi else '400'}">{fv}</span>
        </div>"""

    st_html(f"""
    <div class="hazard-card" style="border:{border};box-shadow:{shadow}">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
            background:{top_bar}"></div>
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
            <div>
                <div style="font-size:9px;color:#3a4e63;font-family:'Rajdhani',sans-serif;
                    letter-spacing:2.5px;margin-bottom:2px">{h['icon']} HAZARD</div>
                <div class="hazard-title" style="color:#dde4ee">{h['label']}</div>
            </div>
            <div style="font-size:9.5px;color:{lvl['color']};font-family:'Rajdhani',sans-serif;
                font-weight:700;letter-spacing:1.5px;padding:3px 9px;border-radius:3px;
                background:{lvl['color']}18;border:1px solid {lvl['color']}35">{lvl['label']}</div>
        </div>
        {warning_banner}
        <div style="display:flex;justify-content:center;margin-bottom:10px">{gauge_svg}</div>
        {factors_html}
    </div>
    """)


def render_recommendations(scores, is_us):
    """
    Shows actionable safety guidance for any hazard at MODERATE severity or
    above. For non-US locations, only Wildfire and Heat are eligible —
    generating "prepare your tornado safe room" advice from a score we've
    already flagged as unvalidated would undercut the honesty built into
    the hazard cards themselves.
    """
    eligible = []
    for h in HAZARDS:
        if h.get("us_only", False) and not is_us:
            continue
        score = scores[h["key"]]
        level = get_threat_level(score)
        tier = _LEVEL_TO_TIER.get(level["label"])
        if tier:
            eligible.append((h, score, level, tier))

    eligible.sort(key=lambda x: x[1], reverse=True)

    st_html("""
    <div style="margin:20px 0 10px 0">
        <span style="font-size:9px;color:#2e4156;font-family:'Rajdhani',sans-serif;
            letter-spacing:3px">RECOMMENDATIONS</span>
    </div>""")

    if not eligible:
        st_html("""
        <div style="background:#07090f;border:1px solid #0e1929;border-radius:10px;
            padding:16px 20px;display:flex;align-items:center;gap:10px">
            <span style="font-size:18px">✅</span>
            <span style="font-size:12px;color:#8fa0b0;font-family:'Inter',sans-serif">
                No elevated risks detected right now — no specific action needed beyond
                normal weather awareness.
            </span>
        </div>""")
        return

    for h, score, level, tier in eligible:
        color = h["color"]
        tips = RECOMMENDATIONS[h["key"]][tier]
        tips_html = "".join(f"""
        <div style="display:flex;align-items:flex-start;gap:8px;margin-top:6px">
            <span style="color:{color};font-size:12px;line-height:1.5">▸</span>
            <span style="font-size:11.5px;color:#c3ccd6;font-family:'Inter',sans-serif;
                line-height:1.5">{tip}</span>
        </div>""" for tip in tips)

        st_html(f"""
        <div style="background:#080e1c;border:1px solid {color}35;border-left:3px solid {color};
            border-radius:8px;padding:14px 18px;margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">
                <span style="font-size:13px;font-family:'Rajdhani',sans-serif;font-weight:700;
                    letter-spacing:1px;color:#dde4ee">{h['icon']} {h['label']}</span>
                <span style="font-size:9px;font-family:'Rajdhani',sans-serif;font-weight:700;
                    letter-spacing:1.5px;color:{level['color']};padding:2px 8px;border-radius:3px;
                    background:{level['color']}18">{level['label']} · {score}/100</span>
            </div>
            {tips_html}
        </div>""")


def render_wx_strip(wx_row):
    items = [
        ("TEMP MAX",  f"{wx_row.get('temperature_2m_max', 0):.1f}°C"),
        ("HUMIDITY",  f"{wx_row.get('relative_humidity_2m_mean', 0):.0f}%"),
        ("WIND MAX",  f"{wx_row.get('windspeed_10m_max', 0):.0f} km/h"),
        ("GUSTS",     f"{wx_row.get('windgusts_10m_max', 0):.0f} km/h"),
        ("PRECIP",    f"{wx_row.get('precipitation_sum', 0):.1f} mm"),
        ("ET₀",       f"{wx_row.get('et0_fao_evapotranspiration', 0):.1f} mm"),
    ]
    items_html = "".join(f"""
        <div style="display:flex;flex-direction:column;padding:0 16px;
            border-left:1px solid #0e1929">
            <span class="wx-item-label">{lbl}</span>
            <span class="wx-item-value">{val}</span>
        </div>""" for lbl, val in items)

    st_html(f"""
    <div class="wx-strip">
        <span style="font-size:8px;color:#2e4156;font-family:'Rajdhani',sans-serif;
            letter-spacing:2.5px;white-space:nowrap">SURFACE CONDITIONS</span>
        {items_html}
    </div>
    """)


# ── MAIN APP ──────────────────────────────────────────────────────────────────

def main():
    try:
        model, scaler, meta = load_model_and_scaler()
    except Exception as e:
        st.error(f"Could not load model files: {e}\n\nMake sure stormsentinel_model.pt, "
                 f"feature_scaler.pkl, and feature_columns.json are in the same directory.")
        st.stop()

    if "city_info" not in st.session_state:
        st.session_state.city_info = next(c for c in CITIES if c["city"] == "Phoenix")
    if "pending_candidates" not in st.session_state:
        st.session_state.pending_candidates = None

    # ── Sidebar controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚡ StormSentinel AI")
        st.markdown("**Search any US location**")

        with st.form("location_search"):
            query = st.text_input("City or place, anywhere", placeholder="e.g. Austin, TX or Nairobi, Kenya")
            searched = st.form_submit_button("🔍 Search")

        if searched and query:
            candidates = geocode_location(query)
            if not candidates:
                st.error(f"No match found for '{query}'. Try a different spelling.")
                st.session_state.pending_candidates = None
            elif len(candidates) == 1:
                # Unambiguous — proceed immediately, no need to make the user
                # confirm a single obvious match
                st.session_state.city_info = candidates[0]
                st.session_state.pending_candidates = None
                st.success(f"Showing: {candidates[0]['display_name']}")
            else:
                # Multiple places share this name (e.g. "Paris" -> Paris FR,
                # Paris TX, Paris KY, Paris IL, Paris ME) — never silently
                # guess which one the user meant. Hold the list and ask.
                st.session_state.pending_candidates = candidates

        # ── Disambiguation picker — only shown when a search was ambiguous ──
        if st.session_state.pending_candidates:
            n = len(st.session_state.pending_candidates)
            st.warning(f"Found {n} US places matching that name — which one did you mean?", icon="📍")
            options = [c["disambiguation_label"] for c in st.session_state.pending_candidates]
            picked_label = st.radio("Select the correct location:", options, index=0, label_visibility="collapsed")
            if st.button("✅ Confirm location"):
                picked = next(c for c in st.session_state.pending_candidates
                              if c["disambiguation_label"] == picked_label)
                st.session_state.city_info = picked
                st.session_state.pending_candidates = None
                st.rerun()

        st.markdown("---")
        st.markdown("**Or quick-select a trained city:**")
        city_names = [c["city"] for c in CITIES]
        selected_name = st.selectbox("Quick select", city_names, index=19, label_visibility="collapsed")
        if st.button("Use this city"):
            st.session_state.city_info = next(c for c in CITIES if c["city"] == selected_name)
            st.session_state.pending_candidates = None

        city_info = st.session_state.city_info

        st.markdown("---")
        st_html(f"""
        **Zone:** `{city_info.get('zone', 'custom').upper()}`  
        **State:** `{city_info['state']}`  
        **Coordinates:** `{city_info['lat']:.2f}°N, {city_info['lon']:.2f}°W`
        """)
        st.markdown("---")
        st.caption("Model: StormSentinelNet | 7-head multi-task PyTorch  \n"
                   "Weather: Open-Meteo Archive API  \n"
                   "Drought: NOAA PDSI climdiv  \n"
                   "Labels: NOAA Storm Events + NASA FIRMS")

    display_name = city_info.get("city", "Unknown")

    # ── Header ────────────────────────────────────────────────────────────
    utc_time = datetime.utcnow().strftime("%H:%M:%S")
    render_header(display_name, city_info["state"], utc_time)

    # ── Out-of-training-region notice ────────────────────────────────────
    # Three tiers, strongest to mildest:
    #  1. Non-US: 5 of 7 hazards have zero training signal outside the US
    #     (NOAA Storm Events + PDSI are US-only data sources). Say so plainly.
    #  2. US but outside the 15 trained states: general extrapolation caveat,
    #     same as before — all 7 heads still learned from US patterns.
    #  3. US, trained state, just not one of the 32 training cities: mild note.
    if not city_info.get("is_us", False):
        st.warning(
            f"🌍 **{display_name}** is outside the United States. StormSentinel trained "
            f"exclusively on US weather and hazard data. **Wildfire** and **Extreme Heat** "
            f"use physically-grounded features (humidity, wind, temperature) that transfer "
            f"reasonably across climates — those two are shown normally. But **Tornado, "
            f"Hail, Thunderstorm Wind, Flash Flood, and Drought** were trained only on "
            f"NOAA's US-only databases and have **no real grounding for this location** — "
            f"those 5 cards are flagged individually below. Treat them as a demonstration "
            f"of the model architecture, not a real risk assessment.",
            icon="🌍",
        )
    elif city_info["state"] not in TRAINED_STATES:
        st.warning(
            f"📍 **{display_name}, {city_info['state']}** is outside StormSentinel's core "
            f"training region (15 states across the wildfire, tornado-alley, and heat-corridor "
            f"zones). The model can still generate a prediction from live weather data, but "
            f"treat these numbers as **lower-confidence extrapolation** rather than a "
            f"validated forecast for this specific area.",
            icon="⚠️",
        )
    elif city_info.get("zone") == "custom":
        st.info(
            f"📍 Showing a searched location within StormSentinel's trained region "
            f"({city_info['state']}), but not one of the 32 cities used in training. "
            f"Predictions use the same model and live weather data as the trained cities.",
            icon="ℹ️",
        )

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Fetching weather data and running inference..."):
        try:
            wx_df = fetch_weather_lookback(city_info["lat"], city_info["lon"])
        except Exception as e:
            st.error(f"Weather data fetch failed: {e}")
            st.stop()

        pdsi_val = fetch_pdsi(city_info["state"])

        try:
            feat_row, last_wx = engineer_features(wx_df, pdsi_val, city_info, meta)
            probs, scores = run_inference(feat_row, model, scaler, meta)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("This usually means a feature column mismatch between training "
                    "and inference. Check that feature_columns.json matches what "
                    "07_feature_engineering.py produced.")
            st.stop()

    wx_row = dict(last_wx)
    wx_row["pdsi"] = pdsi_val

    # ── Composite threat ──────────────────────────────────────────────────
    render_composite(scores)

    # ── 7 hazard cards (2 rows: 4 + 3) ───────────────────────────────────
    row1 = HAZARDS[:4]
    row2 = HAZARDS[4:]

    cols1 = st.columns(4)
    for col, h in zip(cols1, row1):
        with col:
            render_hazard_card(h, scores[h["key"]], wx_row, is_us=city_info.get("is_us", False))

    cols2 = st.columns(3)
    for col, h in zip(cols2, row2):
        with col:
            render_hazard_card(h, scores[h["key"]], wx_row, is_us=city_info.get("is_us", False))

    # ── Recommendations ───────────────────────────────────────────────────
    render_recommendations(scores, is_us=city_info.get("is_us", False))

    # ── Surface conditions ────────────────────────────────────────────────
    render_wx_strip(wx_row)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""<div class="ss-footer">
        STORMSENTINELNET · 7-HEAD MULTI-TASK NEURAL NETWORK ·
        OPEN-METEO · NASA FIRMS · NOAA STORM EVENTS · NOAA PDSI
    </div>""")


if __name__ == "__main__":
    main()
