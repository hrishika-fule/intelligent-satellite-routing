"""
dashboard.py
============
Satellite Delay Predictor — Twitter-inspired White Minimalist Theme.
Run:
    streamlit run dashboard.py
"""

import pickle
import time
import warnings
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from datetime import datetime
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

HAS_MAP = True
try:
    from streamlit_folium import st_folium
    import folium
except ImportError:
    st_folium = None
    folium = None
    HAS_MAP = False

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
MODEL_PKL   = BASE_DIR / "model.pkl"
DATASET_CSV = BASE_DIR / "dataset.csv"

st.set_page_config(
    page_title="Satellite Link Delay Simulator",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide Streamlit menu and deploy button ───────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Global CSS — Twitter-inspired white minimalist palette ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: #f5f8fa !important;
    color: #0f1419 !important;
}
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e1e8ed !important;
    width: 380px !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Sidebar text & content visibility ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #0f1419 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ── */
header[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e1e8ed !important;
}
footer { display: none !important; }

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p, label {
    color: #536471 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Radio + Input text styling in sidebar ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] div,
[data-testid="stRadio"] span,
[data-testid="stRadio"] p,
[data-testid="stRadio"] .css-1wlv8u7,
[data-testid="stRadio"] .css-k1ih3n,
[data-testid="stRadio"] .css-1szy77x,
[data-testid="stRadio"] .stRadio > label,
.stRadio label p,
.stRadio span,
div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] p,
/* Catch-all: every p, span, div inside sidebar uses Inter */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] * {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

[data-testid="stRadio"] label p,
div[role="radiogroup"] label p {
    color: #0f1419 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] label {
    color: #0f1419 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 13px !important;
}

/* ── Number input field styling ── */
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid #e1e8ed !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
    color: #0f1419 !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: #1d9bf0 !important;
    box-shadow: 0 0 0 2px rgba(29, 155, 240, 0.1) !important;
}

/* ── Map container & iframe styling ── */
[data-testid="stIFrame"] {
    border-radius: 12px !important;
    border: 1px solid #e1e8ed !important;
    overflow: hidden !important;
}

/* ── Alert/Info/Warning styling ── */
.stAlert, [data-testid="stAlert"] {
    border-left: 4px solid #1d9bf0 !important;
    border-radius: 8px !important;
}

.stWarning, [data-testid="stWarning"] {
    border-left: 4px solid #f5a623 !important;
    border-radius: 8px !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: #e1e8ed !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #1d9bf0 !important;
}

/* ── Column spacing in sidebar ── */
[data-testid="stSidebar"] [data-testid="column"] {
    padding: 0 0.5rem !important;
}

/* ── Button ── */
div.stButton > button {
    background: #1d9bf0 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: background 0.2s, transform 0.1s !important;
    box-shadow: 0 2px 8px rgba(29,155,240,0.25) !important;
}
div.stButton > button:hover {
    background: #1a8cd8 !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #e1e8ed !important;
    color: #0f1419 !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #e1e8ed !important;
    margin: 1rem 0 !important;
}

/* ── Metric card ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e1e8ed;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card:hover {
    border-color: #1d9bf0;
    box-shadow: 0 4px 16px rgba(29,155,240,0.12);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 11px;
    font-weight: 700;
    color: #536471;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #0f1419;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-unit {
    font-size: 13px;
    font-weight: 400;
    color: #536471;
    margin-left: 4px;
}
.metric-sub {
    font-size: 12px;
    color: #8899a6;
    margin-top: 6px;
}

/* ── Hero prediction card ── */
.hero-card {
    background: linear-gradient(135deg, #1d9bf0 0%, #0a6ebd 100%);
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(29,155,240,0.25);
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    color: rgba(255,255,255,0.75);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-value {
    font-size: 4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1;
}
.hero-unit {
    font-size: 1.4rem;
    font-weight: 300;
    color: rgba(255,255,255,0.75);
    margin-left: 6px;
}
.hero-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin-top: 10px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.35);
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Section heading ── */
.section-head {
    font-size: 17px;
    font-weight: 800;
    color: #0f1419;
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e1e8ed;
}

/* ── Breakdown bar ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.bar-label {
    font-size: 12px;
    color: #536471;
    width: 110px;
    flex-shrink: 0;
    font-weight: 600;
}
.bar-track {
    flex: 1;
    height: 8px;
    background: #e1e8ed;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}
.bar-val {
    font-size: 12px;
    font-weight: 700;
    color: #0f1419;
    width: 85px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    font-size: 20px;
    font-weight: 800;
    color: #0f1419;
    margin-bottom: 2px;
    letter-spacing: -0.03em;
}
.sidebar-tagline {
    font-size: 11px;
    color: #8899a6;
    margin-bottom: 1.2rem;
    font-weight: 400;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f5f8fa;
    border: 1px solid #e1e8ed;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    color: #536471;
    margin-right: 6px;
    margin-bottom: 10px;
}
.dot-ok   { width:7px; height:7px; border-radius:50%; background:#00ba7c; display:inline-block; }
.dot-err  { width:7px; height:7px; border-radius:50%; background:#f4212e; display:inline-block; }
.dot-warn { width:7px; height:7px; border-radius:50%; background:#ffad1f; display:inline-block; }

/* ── Orbit badge ── */
.orbit-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.orbit-LEO { background: #e8f5fd; color: #1d9bf0; border: 1px solid #b3e0fc; }
.orbit-MEO { background: #f3e8fd; color: #7856ff; border: 1px solid #d4b3fc; }
.orbit-GEO { background: #fff8e1; color: #f5a623; border: 1px solid #ffe082; }

/* ── Sidebar section card ── */
.sb-section {
    background: #f5f8fa;
    border: 1px solid #e1e8ed;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.sb-section-title {
    font-size: 11px;
    font-weight: 800;
    color: #0f1419;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Sidebar info row ── */
.sb-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.sb-info-label {
    font-size: 11px;
    color: #536471;
    font-weight: 500;
}
.sb-info-val {
    font-size: 12px;
    color: #0f1419;
    font-weight: 700;
}

/* ── ML testing steps ── */
.ml-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
    color: #0f1419;
    background: #f5f8fa;
    border: 1px solid #e1e8ed;
    transition: all 0.3s;
}
.ml-step.done {
    background: #e8f5fd;
    border-color: #b3e0fc;
    color: #1d9bf0;
}
.ml-step.active {
    background: #fff8e1;
    border-color: #ffe082;
    color: #f5a623;
}
.ml-step-icon { font-size: 15px; }

/* ── Chart wrapper ── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e1e8ed;
    border-radius: 16px;
    padding: 1.2rem 1.2rem 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.chart-title {
    font-size: 14px;
    font-weight: 700;
    color: #0f1419;
    margin-bottom: 0.4rem;
}
.chart-sub {
    font-size: 11px;
    color: #8899a6;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not MODEL_PKL.exists():
        return None, None, None
    with open(MODEL_PKL, "rb") as f:
        p = pickle.load(f)
    return p["model"], p["feature_cols"], p.get("metrics", {})

@st.cache_data(show_spinner=False)
def load_dataset_cached():
    if not DATASET_CSV.exists():
        return None
    return pd.read_csv(DATASET_CSV)


# ── Utilities ──────────────────────────────────────────────────────────────────
def orbit_enc(alt):
    return 0 if alt <= 2000 else (1 if alt <= 35786 else 2)

def orbit_label(enc):
    return {0: "LEO", 1: "MEO", 2: "GEO"}.get(enc, "—")

def compute_distance(alt_km, az, el):
    from physics_engine import satellite_cartesian, ground_station_cartesian, euclidean_distance
    return euclidean_distance(
        satellite_cartesian(alt_km, az, el),
        ground_station_cartesian(0.0, 0.0)
    )

@st.cache_data(ttl=60, show_spinner=False)
def fetch_conditions_from_api(lat, lon):
    """Fetch real-time weather and deduce local time congestion via Open-Meteo API."""
    weather_factor = 0.2
    congestion_factor = 0.3
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if "current_weather" in data:
                cw = data["current_weather"]
                w_code = cw.get("weathercode", 0)
                
                if w_code == 0: weather_factor = 0.1
                elif w_code in [1, 2, 3]: weather_factor = 0.25
                elif w_code in [45, 48]: weather_factor = 0.4
                elif w_code in [51, 53, 55, 56, 57]: weather_factor = 0.5
                elif w_code in [61, 63, 65, 66, 67]: weather_factor = 0.65
                elif w_code in [71, 73, 75, 77]: weather_factor = 0.7
                elif w_code in [80, 81, 82, 85, 86]: weather_factor = 0.75
                elif w_code in [95, 96, 99]: weather_factor = 0.95
                else: weather_factor = 0.3
                
                time_str = cw.get("time")
                if time_str:
                    dt = datetime.fromisoformat(time_str)
                    offset = int(lon / 15.0)
                    hour = (dt.hour + offset) % 24
                    
                    if 0 <= hour < 6: congestion_factor = 0.15
                    elif 6 <= hour < 9: congestion_factor = 0.45
                    elif 9 <= hour < 12: congestion_factor = 0.75
                    elif 12 <= hour < 18: congestion_factor = 0.50
                    elif 18 <= hour < 22: congestion_factor = 0.85
                    else: congestion_factor = 0.30
    except Exception:
        pass
        
    return min(0.95, weather_factor), min(0.95, congestion_factor)

def estimate_weather_factor(lat, lon):
    w, c = fetch_conditions_from_api(lat, lon)
    return w

def estimate_congestion_factor(lat, lon):
    w, c = fetch_conditions_from_api(lat, lon)
    return c

# ── Chart colour palette ───────────────────────────────────────────────────────
TWITTER_BLUE  = "#1d9bf0"
CORAL         = "#ff6b6b"
EMERALD       = "#00ba7c"
AMBER         = "#f5a623"
VIOLET        = "#7856ff"
PINK          = "#f91880"
TEAL          = "#00bcd4"
MUTED         = "#8899a6"
BG_WHITE      = "#ffffff"
BG_LIGHT      = "#f5f8fa"
BORDER        = "#e1e8ed"
TEXT          = "#0f1419"

ORBIT_COLORS  = {"LEO": TWITTER_BLUE, "MEO": VIOLET, "GEO": AMBER}
COMP_COLORS   = {"Propagation": TWITTER_BLUE, "Transmission": CORAL, "Processing": VIOLET, "Queuing": AMBER}

def apply_white_theme(fig, height=320):
    fig.update_layout(
        paper_bgcolor=BG_WHITE,
        plot_bgcolor=BG_WHITE,
        font=dict(family="'Inter', sans-serif", color=MUTED, size=11),
        height=height,
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(
            gridcolor="#f0f3f4", linecolor=BORDER, zeroline=False,
            tickfont=dict(color=MUTED, size=10),
            title_font=dict(color=TEXT, size=12),
        ),
        yaxis=dict(
            gridcolor="#f0f3f4", linecolor=BORDER, zeroline=False,
            tickfont=dict(color=MUTED, size=10),
            title_font=dict(color=TEXT, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=10),
            orientation="h", y=1.12, x=0,
        ),
    )
    return fig


# ── Load data & model ──────────────────────────────────────────────────────────
model, feature_cols, train_metrics = load_model_cached()
df = load_dataset_cached()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">🛰 Satellite Link Delay Simulator</div>
    <div class="sidebar-tagline">Physics + ML Link Simulator</div>
    """, unsafe_allow_html=True)

    # Status pills
    model_ok = model is not None
    data_ok  = df is not None
    st.markdown(
        f'<span class="status-pill"><span class="{"dot-ok" if model_ok else "dot-err"}"></span>'
        f'{"Model ready" if model_ok else "No model"}</span>'
        f'<span class="status-pill"><span class="{"dot-ok" if data_ok else "dot-err"}"></span>'
        f'{f"{len(df):,} rows" if data_ok else "No data"}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Orbit Parameters / Geometry ──
    st.markdown('<div class="sb-section-title">📡 Orbit Parameters</div>', unsafe_allow_html=True)
    orientation_mode = st.radio(
        "Orientation selection",
        ["Manual satellite angles", "Auto from location"],
        index=1,
        help="Auto mode computes azimuth/elevation from ground station + satellite subpoint geometry",
    )

    altitude_km = st.slider("Altitude (km)", 500, 36000, 550, 50)

    if orientation_mode == "Manual satellite angles":
        azimuth_deg = st.slider("Azimuth (°)", 0, 360, 90, 1)
        elevation_deg = st.slider("Elevation (°)", 5, 90, 45, 1)
        ground_lat = 0.0
        ground_lon = 0.0
        sat_lat = 0.0
        sat_lon = 0.0

    else:
        if "ground_lat" not in st.session_state:
            st.session_state.ground_lat = 0.0
        if "ground_lon" not in st.session_state:
            st.session_state.ground_lon = 0.0
        if "sat_lat" not in st.session_state:
            st.session_state.sat_lat = 0.0
        if "sat_lon" not in st.session_state:
            st.session_state.sat_lon = 0.0

        # Get current values from session state
        ground_lat = st.session_state.ground_lat
        ground_lon = st.session_state.ground_lon
        sat_lat = st.session_state.sat_lat
        sat_lon = st.session_state.sat_lon


        if HAS_MAP and st_folium and folium:
            st.markdown('<div class="sb-section-title" style="margin-top:0.5rem;">🗺 Interactive Map</div>', unsafe_allow_html=True)

            # ── Placing mode toggle ──
            if "place_mode" not in st.session_state:
                st.session_state.place_mode = "Ground Station"
            place_mode = st.radio(
                "Click map to place →",
                ["📍 Ground Station", "🛰️ Satellite"],
                index=0 if st.session_state.place_mode == "Ground Station" else 1,
                horizontal=True,
                label_visibility="visible",
            )
            st.session_state.place_mode = "Ground Station" if "Ground" in place_mode else "Satellite"

            center = [(ground_lat + sat_lat) / 2, (ground_lon + sat_lon) / 2]
            map_obj = folium.Map(
                location=center,
                zoom_start=2,
                control_scale=True,
            )

            # Active marker highlight ring
            gs_highlight = st.session_state.place_mode == "Ground Station"

            # Ground station marker
            folium.Marker(
                [ground_lat, ground_lon],
                popup=folium.Popup(
                    f"<b>📍 Ground Station</b><br>Lat: {ground_lat:.4f}°<br>Lon: {ground_lon:.4f}°",
                    max_width=180
                ),
                tooltip="📍 Ground Station" + (" ← placing here" if gs_highlight else ""),
                icon=folium.Icon(
                    color="blue" if not gs_highlight else "darkblue",
                    icon="home", prefix="fa"
                ),
            ).add_to(map_obj)

            # Satellite subpoint marker
            sat_highlight = not gs_highlight
            folium.Marker(
                [sat_lat, sat_lon],
                popup=folium.Popup(
                    f"<b>🛰️ Satellite Subpoint</b><br>Lat: {sat_lat:.4f}°<br>Lon: {sat_lon:.4f}°",
                    max_width=180
                ),
                tooltip="🛰️ Satellite" + (" ← placing here" if sat_highlight else ""),
                icon=folium.Icon(
                    color="red" if not sat_highlight else "darkred",
                    icon="satellite", prefix="fa"
                ),
            ).add_to(map_obj)

            # Link line
            folium.PolyLine(
                [[ground_lat, ground_lon], [sat_lat, sat_lon]],
                color="#1d9bf0", weight=2, opacity=0.6, dash_array="5, 5"
            ).add_to(map_obj)

            # ── Render map — stable key prevents widget recreation ──
            try:
                map_data = st_folium(
                    map_obj, width=380, height=340,
                    key="satellite_map",
                    returned_objects=["last_clicked"],
                )
            except TypeError:
                map_data = st_folium(
                    map_obj, width=380, height=340,
                    key="satellite_map",
                )

            # ── Handle map click interaction ──
            if map_data and isinstance(map_data, dict):
                # Route click to the active marker — NO explicit st.rerun()
                # (st_folium interaction already triggers Streamlit's natural rerun)
                lc = map_data.get("last_clicked")
                if lc and isinstance(lc, dict):
                    clat = round(float(lc["lat"]), 6)
                    clon = round(float(lc["lng"]), 6)
                    if st.session_state.place_mode == "Ground Station":
                        if (abs(clat - st.session_state.ground_lat) > 1e-5 or
                                abs(clon - st.session_state.ground_lon) > 1e-5):
                            st.session_state.ground_lat = clat
                            st.session_state.ground_lon = clon
                    else:
                        if (abs(clat - st.session_state.sat_lat) > 1e-5 or
                                abs(clon - st.session_state.sat_lon) > 1e-5):
                            st.session_state.sat_lat = clat
                            st.session_state.sat_lon = clon

            # Read back current values (after any session_state update above)
            ground_lat = st.session_state.ground_lat
            ground_lon = st.session_state.ground_lon
            sat_lat    = st.session_state.sat_lat
            sat_lon    = st.session_state.sat_lon

            # ── Read-only coordinate chips ──
            st.markdown(f"""
            <div class="sb-section" style="margin-top:0.5rem;padding:0.6rem 0.9rem;">
                <div class="sb-info-row">
                    <span class="sb-info-label">📍 Ground Station</span>
                    <span class="sb-info-val">{ground_lat:.4f}°, {ground_lon:.4f}°</span>
                </div>
                <div class="sb-info-row" style="margin-bottom:0;">
                    <span class="sb-info-label">🛰️ Satellite</span>
                    <span class="sb-info-val">{sat_lat:.4f}°, {sat_lon:.4f}°</span>
                </div>
            </div>
            """, unsafe_allow_html=True)







        else:
            st.warning(
                "Interactive map click requires streamlit-folium. Install with: `pip install streamlit-folium`."
            )
            map_df = pd.DataFrame([
                {"lat": ground_lat, "lon": ground_lon, "name": "Ground station"},
                {"lat": sat_lat, "lon": sat_lon, "name": "Satellite nadir"},
            ])
            st.map(map_df)

        from physics_engine import azimuth_elevation_from_ground
        azimuth_deg, elevation_deg, computed_slant = azimuth_elevation_from_ground(
            ground_lat, ground_lon, sat_lat, sat_lon, altitude_km
        )

        st.markdown(f"""
        <div class='sb-section'>
            <div class='sb-info-row'><span class='sb-info-label'>Computed Azimuth</span>
                <span class='sb-info-val'>{azimuth_deg:.1f}°</span></div>
            <div class='sb-info-row'><span class='sb-info-label'>Computed Elevation</span>
                <span class='sb-info-val'>{elevation_deg:.1f}°</span></div>
            <div class='sb-info-row'><span class='sb-info-label'>Slant Range</span>
                <span class='sb-info-val'>{computed_slant/1e3:.1f} km</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Orbit type info box
    ol = orbit_label(orbit_enc(altitude_km))
    orbit_desc = {"LEO": "Low Earth Orbit · 500–2,000 km · fast passes, low latency",
                  "MEO": "Medium Earth Orbit · 2,000–35,786 km · GPS band",
                  "GEO": "Geostationary Orbit · ~35,786 km · fixed point, high latency"}
    orbit_emoji = {"LEO": "🟦", "MEO": "🟪", "GEO": "🟨"}
    st.markdown(f"""
    <div class="sb-section" style="margin-top:0.5rem;">
        <div class="sb-section-title">{orbit_emoji[ol]} Orbit Type</div>
        <div style="font-size:16px;font-weight:800;color:#0f1419;margin-bottom:4px;">{ol}</div>
        <div style="font-size:11px;color:#536471;">{orbit_desc[ol]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Payload ──
    st.markdown('<div class="sb-section-title">📦 Payload</div>', unsafe_allow_html=True)
    data_size_mb   = st.slider("Data size (MB)",    0.1, 1000.0,  50.0, 0.5)
    bandwidth_mbps = st.slider("Bandwidth (Mbps)",  1.0, 1000.0, 100.0, 1.0)

    # Efficiency = how hard the link is working: transfer demand vs max bandwidth
    # transfer_rate_mbps = data_size_mb * 8 / (data_size_mb / bandwidth_mbps) = bandwidth_mbps (tautology if 100%)
    # Better: ratio of data to time capacity = capped % of bandwidth utilisation
    transfer_time_s = (data_size_mb * 8) / bandwidth_mbps          # seconds to transfer
    # Link efficiency: how "loaded" the link is per second of capacity used
    efficiency = min(100, int((data_size_mb / (bandwidth_mbps * 10)) * 100))
    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-section-title">⚡ Link Efficiency</div>
        <div class="sb-info-row">
            <span class="sb-info-label">Utilization</span>
            <span class="sb-info-val">{efficiency}%</span>
        </div>
        <div style="height:6px;background:#e1e8ed;border-radius:99px;overflow:hidden;">
            <div style="width:{efficiency}%;height:100%;background:{'#00ba7c' if efficiency>70 else '#f5a623' if efficiency>40 else '#ff6b6b'};border-radius:99px;"></div>
        </div>
        <div class="sb-info-row" style="margin-top:8px;">
            <span class="sb-info-label">Transfer time</span>
            <span class="sb-info-val">{ f"{transfer_time_s*1000:.0f} ms" if transfer_time_s < 1 else (f"{transfer_time_s:.2f} s" if transfer_time_s < 60 else f"{transfer_time_s/60:.1f} min") }</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Conditions ──
    st.markdown('<div class="sb-section-title">🌦 Conditions</div>', unsafe_allow_html=True)
    
    conditions_mode = st.radio(
        "Conditions mode",
        ["Manual adjustment", "Auto from location"],
        index=1,
        help="Auto mode estimates weather/congestion based on geographic location",
    )
    
    if conditions_mode == "Manual adjustment":
        weather_factor    = st.slider("Weather factor",    0.0, 1.0, 0.2, 0.01)
        congestion_factor = st.slider("Congestion factor", 0.0, 1.0, 0.3, 0.01)
    else:
        # Auto mode — always read the very latest coords from session_state
        if orientation_mode == "Auto from location":
            _lat = st.session_state.get("ground_lat", 0.0)
            _lon = st.session_state.get("ground_lon", 0.0)
            # keep local vars in sync for downstream computation
            ground_lat, ground_lon = _lat, _lon
            weather_factor    = estimate_weather_factor(_lat, _lon)
            congestion_factor = estimate_congestion_factor(_lat, _lon)

            st.markdown(f"""
            <div class="sb-section">
                <div class="sb-info-row">
                    <span class="sb-info-label">Location</span>
                    <span class="sb-info-val">{_lat:.4f}°, {_lon:.4f}°</span>
                </div>
                <div class="sb-info-row">
                    <span class="sb-info-label">Weather Factor</span>
                    <span class="sb-info-val">{weather_factor:.3f}</span>
                </div>
                <div class="sb-info-row">
                    <span class="sb-info-label">Congestion Factor</span>
                    <span class="sb-info-val">{congestion_factor:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Manual orbit mode — show lat/lon inputs so user can still get real weather
            st.markdown(
                "<div style='font-size:11px;color:#8899a6;margin-bottom:0.5rem;'>"
                "Enter ground station coordinates for real-time weather fetch:"
                "</div>",
                unsafe_allow_html=True,
            )
            cond_col1, cond_col2 = st.columns(2)
            with cond_col1:
                cond_lat = st.number_input("Lat", -90.0, 90.0, 0.0, 0.1, key="cond_lat")
            with cond_col2:
                cond_lon = st.number_input("Lon", -180.0, 180.0, 0.0, 0.1, key="cond_lon")

            weather_factor    = estimate_weather_factor(cond_lat, cond_lon)
            congestion_factor = estimate_congestion_factor(cond_lat, cond_lon)
            st.markdown(f"""
            <div class="sb-section">
                <div class="sb-info-row">
                    <span class="sb-info-label">Location</span>
                    <span class="sb-info-val">{cond_lat:.4f}°, {cond_lon:.4f}°</span>
                </div>
                <div class="sb-info-row">
                    <span class="sb-info-label">Weather Factor</span>
                    <span class="sb-info-val">{weather_factor:.3f}</span>
                </div>
                <div class="sb-info-row">
                    <span class="sb-info-label">Congestion Factor</span>
                    <span class="sb-info-val">{congestion_factor:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Condition summary
    w_label = "☀️ Clear" if weather_factor < 0.3 else ("🌦 Moderate" if weather_factor < 0.7 else "⛈ Severe")
    c_label = "🟢 Low"   if congestion_factor < 0.3 else ("🟡 Medium"  if congestion_factor < 0.7 else "🔴 High")
    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-info-row">
            <span class="sb-info-label">Weather</span>
            <span class="sb-info-val">{w_label}</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-label">Congestion</span>
            <span class="sb-info-val">{c_label}</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-label">Signal impact</span>
            <span class="sb-info-val">{int((weather_factor*0.5+congestion_factor*0.5)*100)}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


# ── Minimal orbit label needed for title (no heavy computation yet) ──
orbit_e = orbit_enc(altitude_km)
ol      = orbit_label(orbit_e)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Page title row ─────────────────────────────────────────────────────────────
col_title, col_orbit = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <div style="margin-bottom:0.2rem;">
        <span style="font-size:11px;font-weight:700;color:#8899a6;
              letter-spacing:0.1em;text-transform:uppercase;">
            Satellite Link Simulator
        </span>
    </div>
    <div style="font-size:2.1rem;font-weight:800;color:#0f1419;
         letter-spacing:-0.03em;line-height:1.1;margin-bottom:0.15rem;">
        Signal Delay Analysis
    </div>
    <div style="font-size:13px;color:#8899a6;">
        {altitude_km:,} km altitude · {ol} orbit · Press 🚀 Predict to simulate
    </div>
    """, unsafe_allow_html=True)
with col_orbit:
    st.markdown(f"""
    <div style="text-align:right;padding-top:0.8rem;">
        <span class="orbit-badge orbit-{ol}">{ol}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

predict_clicked = st.sidebar.button("🚀 Predict Delay")

if not predict_clicked:
    st.markdown("""
    <div style='text-align:center; padding-top: 80px; padding-bottom: 80px;
                background: #ffffff; border-radius: 16px;
                border: 1px dashed #cfd9e0; margin-top: 20px;'>
        <div style='font-size:2.5rem;margin-bottom:0.75rem;'>🛰️</div>
        <h2 style='font-family: Inter, sans-serif; color: #0f1419;
                   margin-bottom: 10px; font-weight: 700;'>Ready to Simulate</h2>
        <p style='color: #536471; font-size: 15px;'>
            Configure orbit, payload and conditions in the sidebar,<br>
            then click <b>🚀 Predict Delay</b> to run the simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE — only runs after Predict is clicked
# ══════════════════════════════════════════════════════════════════════════════
if orientation_mode == "Auto from location":
    from physics_engine import azimuth_elevation_from_ground
    azimuth_deg, elevation_deg, dist_m = azimuth_elevation_from_ground(
        ground_lat, ground_lon, sat_lat, sat_lon, altitude_km
    )
else:
    dist_m = compute_distance(altitude_km, azimuth_deg, elevation_deg)

feat_row = {
    "altitude_km"       : altitude_km,
    "azimuth_deg"       : azimuth_deg,
    "elevation_deg"     : elevation_deg,
    "distance_m"        : dist_m,
    "weather_factor"    : weather_factor,
    "congestion_factor" : congestion_factor,
    "data_size_mb"      : data_size_mb,
    "bandwidth_mbps"    : bandwidth_mbps,
    "orbit_type_enc"    : orbit_e,
    "log_distance_m"    : float(np.log1p(dist_m)),
    "log_data_size_mb"  : float(np.log1p(data_size_mb)),
    "log_bandwidth_mbps": float(np.log1p(bandwidth_mbps)),
    "data_bw_ratio"     : float(data_size_mb / bandwidth_mbps),
    "log_data_bw_ratio" : float(np.log1p(data_size_mb / bandwidth_mbps)),
    "congestion_sq"     : float(congestion_factor ** 2),
    "weather_sq"        : float(weather_factor ** 2),
    "alt_sin_elev"      : float(altitude_km * np.sin(np.radians(elevation_deg))),
    "eff_distance_m"    : float(dist_m * (1 + weather_factor * 0.30)),
}

# ── ML Evaluation Animation — only on predict click ──────────────────────────
if model and feature_cols:
    eval_container = st.empty()
    steps = [
        ("🔍", "Loading model weights…"),
        ("📊", "Preparing feature vector…"),
        ("🧪", "Running inference on test samples…"),
        ("📈", "Computing R² and evaluation metrics…"),
        ("✅", "Prediction complete!"),
    ]
    for i, (icon, label) in enumerate(steps):
        with eval_container.container():
            st.markdown('<div class="section-head">🤖 ML Model Evaluation</div>', unsafe_allow_html=True)
            for j, (ic, lb) in enumerate(steps):
                cls = "done" if j < i else ("active" if j == i else "")
                tick = "✓" if j < i else (icon if j == i else "·")
                st.markdown(f'<div class="ml-step {cls}"><span class="ml-step-icon">{tick}</span>{lb}</div>',
                            unsafe_allow_html=True)
        time.sleep(0.45)
    eval_container.empty()

# ── Final prediction ──────────────────────────────────────────────────────────
if model and feature_cols:
    pred_df = pd.DataFrame([{c: feat_row[c] for c in feature_cols}])
    predicted_delay = float(model.predict(pred_df)[0])
    pred_method = "ML Model"
else:
    from physics_engine import total_delay as phys_total
    r = phys_total(dist_m, data_size_mb, bandwidth_mbps, weather_factor, congestion_factor)
    predicted_delay = r["total_delay_ms"]
    pred_method = "Physics"

from physics_engine import propagation_delay, transmission_delay, processing_delay, queuing_delay
t_prop  = propagation_delay(dist_m, weather_factor)
t_tx    = transmission_delay(data_size_mb, bandwidth_mbps)
t_proc  = processing_delay(congestion_factor)
t_queue = queuing_delay(congestion_factor)
t_total_phys = t_prop + t_tx + t_proc + t_queue

# ── Hero card + 4 metric cards ─────────────────────────────────────────────────
hero_col, cards_col = st.columns([1, 1.4])

with hero_col:
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-eyebrow">
            Total Predicted Delay
            <span class="hero-badge">{pred_method}</span>
        </div>
        <div style="margin:0.5rem 0;">
            <span class="hero-value">{predicted_delay:,.0f}</span>
            <span class="hero-unit">ms</span>
        </div>
        <div class="hero-sub">
            ≈ {predicted_delay/1000:.3f} s &nbsp;·&nbsp; {ol} orbit &nbsp;·&nbsp; {altitude_km:,} km
        </div>
    </div>
    """, unsafe_allow_html=True)

with cards_col:
    cc1, cc2 = st.columns(2)
    cc3, cc4 = st.columns(2)
    card_data = [
        (cc1, "Propagation",  t_prop,  "ms", TWITTER_BLUE, "Atmospheric + distance"),
        (cc2, "Transmission", t_tx,    "ms", CORAL,        "Data size / bandwidth"),
        (cc3, "Processing",   t_proc,  "ms", VIOLET,       "Transponder overhead"),
        (cc4, "Queuing",      t_queue, "ms", AMBER,        "Buffer queue delay"),
    ]
    for col, label, val, unit, color, sub in card_data:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
                <div class="metric-label">{label}</div>
                <div style="display:flex;align-items:baseline;gap:3px;">
                    <span class="metric-value" style="color:{color};">{val:,.1f}</span>
                    <span class="metric-unit">{unit}</span>
                </div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")


# ── Delay Breakdown Bar Chart ──────────────────────────────────────────────────
st.markdown('<div class="section-head">⚡ Delay Breakdown</div>', unsafe_allow_html=True)
components = [
    ("Propagation",  t_prop,  TWITTER_BLUE),
    ("Transmission", t_tx,    CORAL),
    ("Processing",   t_proc,  VIOLET),
    ("Queuing",      t_queue, AMBER),
]
mx = max(v for _, v, _ in components) or 1
for lbl, val, clr in components:
    pct   = int(val / mx * 100)
    share = (val / t_total_phys * 100) if t_total_phys > 0 else 0
    st.markdown(f"""
    <div class="bar-row">
        <div class="bar-label">{lbl}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{pct}%;background:{clr};"></div>
        </div>
        <div class="bar-val">{val:,.1f} ms &nbsp;<span style="color:#8899a6;font-weight:400;font-size:11px;">({share:.1f}%)</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">🎯 Optimization Insights</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:13px;color:#536471;margin:-0.5rem 0 1rem;'>"
    "What-if analysis — ranked by delay reduction potential.</p>",
    unsafe_allow_html=True
)

# Helper: compute total physics delay for any scenario
def scenario_delay(alt, d_m, wf, cf, ds, bw):
    tp  = propagation_delay(d_m, wf)
    ttx = transmission_delay(ds, bw)
    tpr = processing_delay(cf)
    tq  = queuing_delay(cf)
    return tp + ttx + tpr + tq

base_delay = t_total_phys  # use physics for fair comparison across scenarios

# ── Build scenarios ──────────────────────────────────────────────────────────
scenarios = []

# 1. Switch orbit to LEO 550 km
if altitude_km > 2000:
    from physics_engine import azimuth_elevation_from_ground, euclidean_distance, satellite_cartesian, ground_station_cartesian
    if orientation_mode == "Auto from location":
        _, _, leo_dist = azimuth_elevation_from_ground(ground_lat, ground_lon, sat_lat, sat_lon, 550)
    else:
        leo_dist = euclidean_distance(satellite_cartesian(550, azimuth_deg, elevation_deg),
                                      ground_station_cartesian(0, 0))
    leo_delay = scenario_delay(550, leo_dist, weather_factor, congestion_factor, data_size_mb, bandwidth_mbps)
    saving = base_delay - leo_delay
    if saving > 0:
        scenarios.append({
            "icon": "🛸", "title": "Switch to LEO (550 km)",
            "desc": f"Drop from {altitude_km:,} km → 550 km to slash propagation delay",
            "saving": saving, "tag": "Orbit Change", "tag_color": "#1d9bf0",
        })

# 2. Switch to MEO if currently GEO
if altitude_km > 35786:
    if orientation_mode == "Auto from location":
        _, _, meo_dist = azimuth_elevation_from_ground(ground_lat, ground_lon, sat_lat, sat_lon, 20200)
    else:
        meo_dist = euclidean_distance(satellite_cartesian(20200, azimuth_deg, elevation_deg),
                                      ground_station_cartesian(0, 0))
    meo_delay = scenario_delay(20200, meo_dist, weather_factor, congestion_factor, data_size_mb, bandwidth_mbps)
    saving = base_delay - meo_delay
    if saving > 0:
        scenarios.append({
            "icon": "🪐", "title": "Switch to MEO (20,200 km)",
            "desc": "Step down from GEO to MEO for lower propagation delay",
            "saving": saving, "tag": "Orbit Change", "tag_color": "#7856ff",
        })

# 3. Double the bandwidth
if bandwidth_mbps < 500:
    bw2 = min(bandwidth_mbps * 2, 1000)
    bw2_delay = scenario_delay(altitude_km, dist_m, weather_factor, congestion_factor, data_size_mb, bw2)
    saving = base_delay - bw2_delay
    if saving > 0:
        scenarios.append({
            "icon": "📡", "title": f"Double bandwidth ({bandwidth_mbps:.0f} → {bw2:.0f} Mbps)",
            "desc": "Upgrade your RF/modem to reduce transmission delay",
            "saving": saving, "tag": "Hardware", "tag_color": "#00ba7c",
        })

# 4. Compress data by 50%
if data_size_mb > 1:
    compress_delay = scenario_delay(altitude_km, dist_m, weather_factor, congestion_factor, data_size_mb * 0.5, bandwidth_mbps)
    saving = base_delay - compress_delay
    if saving > 0:
        scenarios.append({
            "icon": "🗜️", "title": "Apply 2× data compression",
            "desc": "Halve payload size via lossless compression before uplink",
            "saving": saving, "tag": "Software", "tag_color": "#f5a623",
        })

# 5. Clear weather (weather_factor → 0.05)
if weather_factor > 0.15:
    clr_delay = scenario_delay(altitude_km, dist_m, 0.05, congestion_factor, data_size_mb, bandwidth_mbps)
    saving = base_delay - clr_delay
    if saving > 0:
        scenarios.append({
            "icon": "☀️", "title": "Schedule during clear weather",
            "desc": f"Current weather factor {weather_factor:.2f} → 0.05 (clear sky)",
            "saving": saving, "tag": "Scheduling", "tag_color": "#f5a623",
        })

# 6. Off-peak congestion (congestion_factor → 0.1)
if congestion_factor > 0.2:
    offpeak_delay = scenario_delay(altitude_km, dist_m, weather_factor, 0.1, data_size_mb, bandwidth_mbps)
    saving = base_delay - offpeak_delay
    if saving > 0:
        scenarios.append({
            "icon": "🌙", "title": "Transmit during off-peak hours (02:00–06:00)",
            "desc": f"Congestion factor {congestion_factor:.2f} → 0.10 during low-traffic window",
            "saving": saving, "tag": "Scheduling", "tag_color": "#f5a623",
        })

# 7. Increase elevation (if low elevation)
if orientation_mode == "Manual satellite angles" and elevation_deg < 45:
    high_el = 80
    from physics_engine import satellite_cartesian, ground_station_cartesian, euclidean_distance
    hel_dist  = euclidean_distance(satellite_cartesian(altitude_km, azimuth_deg, high_el),
                                   ground_station_cartesian(0, 0))
    hel_delay = scenario_delay(altitude_km, hel_dist, weather_factor, congestion_factor, data_size_mb, bandwidth_mbps)
    saving = base_delay - hel_delay
    if saving > 0:
        scenarios.append({
            "icon": "📐", "title": f"Raise elevation angle ({elevation_deg}° → {high_el}°)",
            "desc": "Higher elevation = shorter slant range = less propagation delay",
            "saving": saving, "tag": "Geometry", "tag_color": "#1d9bf0",
        })

# Sort by saving, biggest first
scenarios.sort(key=lambda x: x["saving"], reverse=True)

if not scenarios:
    st.info("✅ Your configuration is already near-optimal for the current orbit type.")
else:
    opt_cols = st.columns(min(3, len(scenarios)))
    for i, s in enumerate(scenarios[:6]):
        col = opt_cols[i % 3]
        pct = (s["saving"] / base_delay * 100) if base_delay > 0 else 0
        bar_w = min(100, int(pct))
        bar_color = "#00ba7c" if pct > 40 else ("#f5a623" if pct > 15 else "#1d9bf0")
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e1e8ed;border-radius:16px;
                        padding:1.1rem 1.2rem;margin-bottom:0.9rem;
                        box-shadow:0 1px 4px rgba(0,0,0,0.05);
                        border-top:3px solid {s['tag_color']};">
                <div style="font-size:1.5rem;margin-bottom:0.3rem;">{s['icon']}</div>
                <div style="font-size:13px;font-weight:700;color:#0f1419;
                            margin-bottom:0.25rem;line-height:1.3;">{s['title']}</div>
                <div style="font-size:11px;color:#536471;margin-bottom:0.7rem;
                            line-height:1.4;">{s['desc']}</div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <div style="flex:1;height:5px;background:#e1e8ed;
                                border-radius:99px;overflow:hidden;">
                        <div style="width:{bar_w}%;height:100%;
                                    background:{bar_color};border-radius:99px;"></div>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:11px;font-weight:700;color:{bar_color};">
                        −{s['saving']:,.0f} ms saved
                    </span>
                    <span style="font-size:10px;background:{s['tag_color']}18;
                                 color:{s['tag_color']};font-weight:700;padding:2px 8px;
                                 border-radius:20px;">{s['tag']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Combined savings summary
    top3_saving = sum(s["saving"] for s in scenarios[:3])
    top3_pct    = (top3_saving / base_delay * 100) if base_delay > 0 else 0
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#e8f5fd 0%,#f0f7ff 100%);
                border:1px solid #b3e0fc;border-radius:14px;
                padding:0.9rem 1.2rem;margin-top:0.25rem;">
        <span style="font-size:12px;font-weight:700;color:#1d9bf0;">
            💡 Combining top 3 optimizations could reduce delay by
            <span style="font-size:15px;">−{top3_saving:,.0f} ms</span>
            ({top3_pct:.0f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")



if df is not None:
    orbit_map_rev = {0: "LEO", 1: "MEO", 2: "GEO"}
    df_plot = df.copy()
    df_plot["orbit"] = df_plot["orbit_type_enc"].map(orbit_map_rev).fillna("LEO")

    st.markdown('<div class="section-head">📊 Data Analysis</div>', unsafe_allow_html=True)

    # ── Row 1: Actual vs Predicted | Feature Importance ──────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="chart-card"><div class="chart-title">🎯 Actual vs Predicted Delay</div><div class="chart-sub">Scatter of model predictions against ground truth</div>', unsafe_allow_html=True)
        if model and feature_cols:
            n_sample = min(800, len(df))
            samp = df[feature_cols].sample(n=n_sample, random_state=7)
            y_a  = df.loc[samp.index, "total_delay_ms"].values
            y_p  = model.predict(samp)
            lim  = max(y_a.max(), y_p.max()) * 1.05
            ss_res = np.sum((y_a - y_p) ** 2)
            ss_tot = np.sum((y_a - y_a.mean()) ** 2)
            r2_samp = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_a, y=y_p, mode="markers",
                marker=dict(
                    color=y_a,
                    colorscale=[[0, TWITTER_BLUE], [0.5, VIOLET], [1, CORAL]],
                    size=5, opacity=0.65, line=dict(width=0),
                    showscale=True,
                    colorbar=dict(thickness=10, len=0.6, title="ms",
                                  tickfont=dict(size=9, color=MUTED)),
                ),
                hovertemplate="Actual: %{x:.0f} ms<br>Predicted: %{y:.0f} ms<extra></extra>",
                name=f"R²={r2_samp:.4f}",
            ))
            fig.add_trace(go.Scatter(
                x=[0, lim], y=[0, lim], mode="lines",
                line=dict(color=EMERALD, width=2, dash="dash"),
                name="Perfect fit",
            ))
            fig = apply_white_theme(fig, height=320)
            fig.update_xaxes(title_text="Actual (ms)")
            fig.update_yaxes(title_text="Predicted (ms)")
            fig.update_layout(title=dict(
                text=f"R² = {r2_samp:.5f}", font=dict(color=EMERALD, size=13), x=0.5
            ))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        else:
            st.info("Train a model to see predictions.")
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-card"><div class="chart-title">📌 Feature Importance</div><div class="chart-sub">Top predictors ranked by gradient boosting</div>', unsafe_allow_html=True)
        if model and feature_cols:
            fi = pd.DataFrame({"feature": feature_cols, "imp": model.feature_importances_}) \
                   .sort_values("imp", ascending=True)
            n = len(fi)
            gradient = [f"hsl({int(200 + 160 * i/max(n-1,1))}, 80%, 55%)" for i in range(n)]
            fig2 = go.Figure(go.Bar(
                x=fi["imp"], y=fi["feature"], orientation="h",
                marker=dict(color=gradient, line_width=0),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig2 = apply_white_theme(fig2, height=320)
            fig2.update_xaxes(title_text="Importance Score")
            fig2.update_yaxes(title_text="")
            st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})
        else:
            st.info("No model loaded.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Delay vs Altitude (full width) ────────────────────────────────────────
    st.markdown('<div class="chart-card"><div class="chart-title">🛸 Delay vs Altitude by Orbit Type</div><div class="chart-sub">Scatter + trend lines across orbital regimes</div>', unsafe_allow_html=True)
    samp2 = df_plot.sample(n=min(3000, len(df_plot)), random_state=42)
    fig3 = go.Figure()
    for orb, clr in ORBIT_COLORS.items():
        sub = samp2[samp2["orbit"] == orb]
        if len(sub) == 0: continue
        fig3.add_trace(go.Scatter(
            x=sub["altitude_km"], y=sub["total_delay_ms"], mode="markers",
            name=orb, marker=dict(color=clr, size=5, opacity=0.5, line=dict(width=0)),
            hovertemplate=f"<b>{orb}</b><br>Alt: %{{x:.0f}} km<br>Delay: %{{y:.0f}} ms<extra></extra>",
        ))
    for orb, clr in ORBIT_COLORS.items():
        sub = samp2[samp2["orbit"] == orb]
        if len(sub) < 5: continue
        z = np.polyfit(sub["altitude_km"], sub["total_delay_ms"], 1)
        x_line = np.linspace(sub["altitude_km"].min(), sub["altitude_km"].max(), 100)
        fig3.add_trace(go.Scatter(
            x=x_line, y=np.polyval(z, x_line), mode="lines",
            line=dict(color=clr, width=2.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    fig3 = apply_white_theme(fig3, height=340)
    fig3.update_xaxes(title_text="Altitude (km)")
    fig3.update_yaxes(title_text="Total Delay (ms)")
    st.plotly_chart(fig3, width='stretch', config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Distribution + Stacked Components ───────────────────────────────
    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown('<div class="chart-card"><div class="chart-title">🎻 Delay Distribution by Orbit</div><div class="chart-sub">Box plot with mean ± std across orbit types</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        for orb, clr in ORBIT_COLORS.items():
            sub = df_plot[df_plot["orbit"] == orb]["total_delay_ms"]
            if len(sub) == 0: continue
            r, g, b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
            fig4.add_trace(go.Box(
                y=sub, name=orb,
                marker_color=clr, line_color=clr,
                fillcolor=f"rgba({r},{g},{b},0.12)",
                whiskerwidth=0.6, boxmean="sd",
            ))
        fig4 = apply_white_theme(fig4, height=320)
        fig4.update_yaxes(title_text="Total Delay (ms)")
        st.plotly_chart(fig4, width='stretch', config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with ch4:
        st.markdown('<div class="chart-card"><div class="chart-title">📦 Avg Delay Components by Orbit</div><div class="chart-sub">Stacked breakdown: propagation, tx, processing, queue</div>', unsafe_allow_html=True)
        comp_rows = []
        for orb in ["LEO", "MEO", "GEO"]:
            sub = df_plot[df_plot["orbit"] == orb]
            if len(sub) == 0: continue
            comp_rows.append({
                "orbit"       : orb,
                "Propagation" : sub["propagation_delay_ms"].mean(),
                "Transmission": sub["transmission_delay_ms"].mean(),
                "Processing"  : sub["processing_delay_ms"].mean(),
                "Queuing"     : sub["queuing_delay_ms"].mean(),
            })
        cdf = pd.DataFrame(comp_rows)
        fig5 = go.Figure()
        for comp, clr in COMP_COLORS.items():
            fig5.add_trace(go.Bar(
                x=cdf["orbit"], y=cdf[comp], name=comp,
                marker_color=clr, marker_line_width=0,
            ))
        fig5.update_layout(barmode="stack")
        fig5 = apply_white_theme(fig5, height=320)
        fig5.update_yaxes(title_text="Delay (ms)")
        st.plotly_chart(fig5, width='stretch', config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    st.markdown('<div class="chart-card"><div class="chart-title">🔥 Feature Correlation Matrix</div><div class="chart-sub">Pearson r between key features and total delay</div>', unsafe_allow_html=True)
    corr_cols = ["altitude_km", "elevation_deg", "distance_m",
                 "weather_factor", "congestion_factor",
                 "data_size_mb", "bandwidth_mbps", "total_delay_ms"]
    corr_cols_present = [c for c in corr_cols if c in df.columns]
    if len(corr_cols_present) >= 3:
        corr = df[corr_cols_present].corr()
        labels = [c.replace("_", " ").replace(" ms", "").replace(" mb", "") for c in corr_cols_present]
        fig6 = go.Figure(go.Heatmap(
            z=corr.values,
            x=labels, y=labels,
            colorscale=[[0, "#cce4f7"], [0.5, TWITTER_BLUE], [1, VIOLET]],
            zmin=-1, zmax=1,
            hovertemplate="X: %{x}<br>Y: %{y}<br>r = %{z:.3f}<extra></extra>",
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#0f1419"),
        ))
        fig6 = apply_white_theme(fig6, height=340)
        fig6.update_layout(margin=dict(l=100, r=20, t=30, b=80))
        st.plotly_chart(fig6, width='stretch', config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")


# ── Model Performance ──────────────────────────────────────────────────────────
if train_metrics:
    st.markdown('<div class="section-head">🎯 Model Performance</div>', unsafe_allow_html=True)
    pm1, pm2, pm3, pm4 = st.columns(4)
    r2_val   = train_metrics.get('r2', 0)
    r2_color = EMERALD if r2_val >= 0.99 else (TWITTER_BLUE if r2_val >= 0.95 else AMBER)
    perf_data = [
        (pm1, "R² Score",   f"{r2_val:.5f}",                           "",      r2_color,     "Higher is better (max 1.0)"),
        (pm2, "MAE",        f"{train_metrics.get('mae',0):.2f}",       "ms",    CORAL,        "Mean absolute error"),
        (pm3, "RMSE",       f"{train_metrics.get('rmse',0):.2f}",      "ms",    VIOLET,       "Root mean sq. error"),
        (pm4, "Estimators", str(model.n_estimators) if model else "—", "trees", AMBER,        "Ensemble size"),
    ]
    for col, lbl, val, unit, color, sub in perf_data:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
                <div class="metric-label">{lbl}</div>
                <div style="display:flex;align-items:baseline;gap:3px;">
                    <span class="metric-value" style="font-size:1.6rem;color:{color};">{val}</span>
                    <span class="metric-unit">{unit}</span>
                </div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;
     padding:0.5rem 0 1rem;">
    <span style="font-size:12px;color:#8899a6;font-weight:500;">
        🛰 SatDelay · Physics-based + ML Satellite Link Simulator
    </span>
    <span style="font-size:12px;color:#8899a6;">
        c = 3×10⁸ m/s &nbsp;·&nbsp; GradientBoosting Regressor &nbsp;·&nbsp; Physics Engine v2
    </span>
</div>
""", unsafe_allow_html=True)