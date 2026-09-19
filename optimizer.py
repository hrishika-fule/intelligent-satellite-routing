"""
optimizer.py
============
Recommends the lowest-latency satellite route given a set of candidate
satellites and current link conditions.

The optimiser:
1. Accepts a list of candidate satellite configurations.
2. Uses the trained ML model to predict delay for each.
3. Falls back to the physics engine if the model is unavailable.
4. Returns the best (minimum delay) satellite and a full ranked table.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from physics_engine import (
    satellite_cartesian,
    ground_station_cartesian,
    euclidean_distance,
    total_delay,
    orbit_type_from_altitude,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_PKL = BASE_DIR / "model.pkl"

# ── Orbit encoding used during training ─────────────────────────────────────
ORBIT_ENC = {"LEO": 0, "MEO": 1, "GEO": 2}


def _load_model():
    """Lazy-load the trained RF model; returns None if not available."""
    if not MODEL_PKL.exists():
        return None
    try:
        import pickle
        with open(MODEL_PKL, "rb") as f:
            payload = pickle.load(f)
        return payload["model"]
    except Exception as e:
        print(f"[optimizer] Warning — could not load model: {e}")
        return None


def build_feature_row(altitude_km: float,
                       azimuth_deg: float,
                       elevation_deg: float,
                       ground_lat: float,
                       ground_lon: float,
                       weather_factor: float,
                       congestion_factor: float,
                       data_size_mb: float,
                       bandwidth_mbps: float) -> dict:
    """
    Compute derived features (distance, orbit encoding) and return a
    feature dict ready for model inference.
    """
    sat_pos = satellite_cartesian(altitude_km, azimuth_deg, elevation_deg)
    gnd_pos = ground_station_cartesian(ground_lat, ground_lon)
    dist_m  = euclidean_distance(sat_pos, gnd_pos)
    orbit   = orbit_type_from_altitude(altitude_km)

    return {
        "altitude_km"      : altitude_km,
        "azimuth_deg"      : azimuth_deg,
        "elevation_deg"    : elevation_deg,
        "distance_m"       : dist_m,
        "weather_factor"   : weather_factor,
        "congestion_factor": congestion_factor,
        "data_size_mb"     : data_size_mb,
        "bandwidth_mbps"   : bandwidth_mbps,
        "orbit_type_enc"   : ORBIT_ENC.get(orbit, 0),
        # --- physics fallback data (not used by model) ---
        "_distance_m"      : dist_m,
        "_orbit"           : orbit,
    }


def predict_delays(candidates: List[dict],
                   model=None) -> pd.DataFrame:
    """
    Predict (or compute) delay for each candidate satellite configuration.

    Parameters
    ----------
    candidates : List of dicts, each containing:
                    name, altitude_km, azimuth_deg, elevation_deg,
                    ground_lat, ground_lon, weather_factor,
                    congestion_factor, data_size_mb, bandwidth_mbps
    model      : Trained RandomForestRegressor (optional).

    Returns
    -------
    pd.DataFrame ranked by predicted_delay_ms ascending.
    """
    from train_model import FEATURE_COLS  # import here to avoid circular deps

    rows = []
    for c in candidates:
        feat = build_feature_row(
            altitude_km       = c["altitude_km"],
            azimuth_deg       = c["azimuth_deg"],
            elevation_deg     = c["elevation_deg"],
            ground_lat        = c["ground_lat"],
            ground_lon        = c["ground_lon"],
            weather_factor    = c["weather_factor"],
            congestion_factor = c["congestion_factor"],
            data_size_mb      = c["data_size_mb"],
            bandwidth_mbps    = c["bandwidth_mbps"],
        )

        if model is not None:
            # Use ML model for fast prediction
            input_df = pd.DataFrame([{col: feat[col] for col in FEATURE_COLS}])
            pred_delay = float(model.predict(input_df)[0])
            method = "ML"
        else:
            # Fallback: compute directly from physics
            delays = total_delay(
                distance_m        = feat["_distance_m"],
                data_size_mb      = c["data_size_mb"],
                bandwidth_mbps    = c["bandwidth_mbps"],
                weather_factor    = c["weather_factor"],
                congestion_factor = c["congestion_factor"],
            )
            pred_delay = delays["total_delay_ms"]
            method = "Physics"

        rows.append({
            "satellite_name"    : c.get("name", f"SAT-{len(rows)+1}"),
            "orbit_type"        : feat["_orbit"],
            "altitude_km"       : round(c["altitude_km"], 1),
            "distance_m"        : round(feat["_distance_m"] / 1_000, 2),  # stored as km
            "weather_factor"    : c["weather_factor"],
            "congestion_factor" : c["congestion_factor"],
            "predicted_delay_ms": round(pred_delay, 4),
            "method"            : method,
        })

    result = pd.DataFrame(rows).sort_values("predicted_delay_ms").reset_index(drop=True)
    return result


def recommend(candidates: List[dict],
              model=None,
              verbose: bool = True) -> dict:
    """
    Find and return the best satellite route (minimum predicted delay).

    Parameters
    ----------
    candidates : List of satellite configuration dicts (see predict_delays).
    model      : Optional trained model; uses physics fallback if None.
    verbose    : Print formatted recommendation report.

    Returns
    -------
    dict with keys: best_satellite, delay_ms, ranked_table.
    """
    if not candidates:
        raise ValueError("No candidates provided.")

    if model is None:
        model = _load_model()

    ranked = predict_delays(candidates, model)
    best   = ranked.iloc[0]

    if verbose:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  🛰  SATELLITE ROUTE RECOMMENDATION")
        print(sep)
        print(f"  Best satellite  : {best['satellite_name']}")
        print(f"  Orbit type      : {best['orbit_type']}")
        print(f"  Altitude        : {best['altitude_km']} km")
        print(f"  Path distance   : {best['distance_m']} km")
        print(f"  Weather factor  : {best['weather_factor']:.2f}")
        print(f"  Congestion      : {best['congestion_factor']:.2f}")
        print(f"  Predicted delay : {best['predicted_delay_ms']:.2f} ms")
        print(f"  Method used     : {best['method']}")
        print(f"\n  Full ranking:")
        print(ranked[["satellite_name", "orbit_type", "altitude_km",
                       "predicted_delay_ms"]].to_string(index=False))
        print(sep)

    return {
        "best_satellite": best["satellite_name"],
        "delay_ms"      : best["predicted_delay_ms"],
        "ranked_table"  : ranked,
    }


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: five hypothetical satellites visible from the same ground station
    GROUND = {"ground_lat": 28.6, "ground_lon": 77.2}   # near New Delhi
    COMMON = {
        "data_size_mb": 50.0,
        "bandwidth_mbps": 100.0,
        "weather_factor": 0.2,
        "congestion_factor": 0.3,
    }

    satellite_candidates = [
        {"name": "Starlink-L550",  "altitude_km": 550,    "azimuth_deg": 45,  "elevation_deg": 60, **GROUND, **COMMON},
        {"name": "Starlink-L1200", "altitude_km": 1200,   "azimuth_deg": 120, "elevation_deg": 45, **GROUND, **COMMON},
        {"name": "O3b-MEO",        "altitude_km": 8063,   "azimuth_deg": 200, "elevation_deg": 30, **GROUND, **COMMON},
        {"name": "GPS-MEO",        "altitude_km": 20200,  "azimuth_deg": 270, "elevation_deg": 55, **GROUND, **COMMON},
        {"name": "Intelsat-GEO",   "altitude_km": 35786,  "azimuth_deg": 180, "elevation_deg": 25, **GROUND, **COMMON},
    ]

    result = recommend(satellite_candidates, verbose=True)
    print(f"\n→ Recommended: {result['best_satellite']} "
          f"with {result['delay_ms']:.2f} ms delay")
