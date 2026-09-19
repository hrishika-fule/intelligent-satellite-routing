"""
physics_engine.py
=================
Core physics model for satellite communication delay simulation.

ASSUMPTIONS:
- Speed of light in vacuum (c = 3×10⁸ m/s) is used; atmospheric effects
  are captured via the weather factor multiplier instead of computing
  exact refractive indices.
- Earth is treated as a sphere (radius 6371 km) for ground station
  positioning.  This introduces <0.3% positional error vs. WGS-84.
- Processing delay is modelled as a fixed baseline (2 ms) plus a
  congestion-driven component; real hardware latency varies by vendor
  but 2–5 ms is representative of modern bent-pipe transponders.
- Queuing delay grows non-linearly with congestion to mimic M/M/1
  queue behaviour without requiring full queuing-theory simulation.
- Weather factor (0–1) scales propagation delay by up to 30% to
  approximate rain fade, tropospheric scintillation, and multipath.
- Transmission delay uses a simple size/bandwidth model (no TCP
  overhead, no ARQ retransmissions) — these layers are not simulated.
"""

import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
SPEED_OF_LIGHT = 3e8          # metres per second
EARTH_RADIUS_KM = 6371.0      # mean Earth radius in km
EARTH_RADIUS_M  = EARTH_RADIUS_KM * 1_000

# ── Orbit type altitude bands (km) ──────────────────────────────────────────
ORBIT_BANDS = {
    "LEO": (500,   2_000),
    "MEO": (2_000, 35_786),
    "GEO": (35_786, 35_786),
}

# ── Delay constants ─────────────────────────────────────────────────────────
PROCESSING_DELAY_BASE_MS = 2.0   # ms — fixed hardware/firmware overhead
MAX_WEATHER_MULTIPLIER   = 0.30  # weather can add up to 30% to prop delay
MAX_QUEUING_DELAY_MS     = 50.0  # ms — saturated buffer worst case


def satellite_cartesian(altitude_km: float,
                         azimuth_deg: float,
                         elevation_deg: float) -> np.ndarray:
    """
    Convert satellite position from spherical orbital params to 3-D
    Cartesian coordinates (metres) centred on Earth's core.

    Parameters
    ----------
    altitude_km   : Altitude above Earth's surface in km.
    azimuth_deg   : Azimuth angle from ground station (0–360°).
    elevation_deg : Elevation angle above horizon (0–90°).

    Returns
    -------
    np.ndarray shape (3,) — (x_s, y_s, z_s) in metres.
    """
    r = (EARTH_RADIUS_KM + altitude_km) * 1_000  # total radius in metres
    az  = np.radians(azimuth_deg)
    el  = np.radians(elevation_deg)

    # Standard spherical → Cartesian conversion
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.array([x, y, z])


def ground_station_cartesian(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Convert ground station geodetic coordinates to 3-D Cartesian (metres).

    Parameters
    ----------
    lat_deg : Geodetic latitude  in degrees (−90 to +90).
    lon_deg : Geodetic longitude in degrees (−180 to +180).

    Returns
    -------
    np.ndarray shape (3,) — (x_g, y_g, z_g) in metres.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r   = EARTH_RADIUS_M

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """Convert geodetic coordinates to ECEF Cartesian (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = (EARTH_RADIUS_KM + alt_km) * 1_000
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def satellite_ecef(altitude_km: float, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Compute satellite Earth-centred coordinates from orbit subpoint and altitude."""
    return geodetic_to_ecef(lat_deg, lon_deg, altitude_km)


def azimuth_elevation_from_ground(ground_lat: float, ground_lon: float,
                                 sat_lat: float, sat_lon: float,
                                 alt_km: float) -> tuple[float, float, float]:
    """Compute azimuth (deg), elevation (deg), and slant distance (m) from ground station to satellite."""
    gs = ground_station_cartesian(ground_lat, ground_lon)
    sat = satellite_ecef(alt_km, sat_lat, sat_lon)
    vec = sat - gs

    lat = np.radians(ground_lat)
    lon = np.radians(ground_lon)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

    e = float(np.dot(east, vec))
    n = float(np.dot(north, vec))
    u = float(np.dot(up, vec))

    horiz = np.hypot(e, n)
    slant = np.linalg.norm(vec)

    azimuth_deg = (np.degrees(np.arctan2(e, n)) + 360.0) % 360.0
    elevation_deg = np.degrees(np.arctan2(u, horiz))

    return azimuth_deg, elevation_deg, slant


def euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """
    3-D Euclidean distance between two Cartesian position vectors (metres).

    d = sqrt( (x_a-x_b)² + (y_a-y_b)² + (z_a-z_b)² )
    """
    delta = pos_a - pos_b
    return float(np.sqrt(np.dot(delta, delta)))


def propagation_delay(distance_m: float, weather_factor: float) -> float:
    """
    Compute one-way propagation delay (milliseconds).

    Formula:  t_prop = d / c
    Weather adds atmospheric latency (rain fade, scintillation):
              t_prop_effective = t_prop × (1 + weather_factor × MAX_WEATHER_MULT)

    Parameters
    ----------
    distance_m     : Signal path distance in metres.
    weather_factor : 0 (clear sky) → 1 (severe weather).

    Returns
    -------
    Propagation delay in milliseconds.
    """
    t_base = distance_m / SPEED_OF_LIGHT          # seconds
    # Weather increases effective path length / signal travel time
    weather_scaling = 1.0 + weather_factor * MAX_WEATHER_MULTIPLIER
    return t_base * weather_scaling * 1_000        # → ms


def transmission_delay(data_size_mb: float, bandwidth_mbps: float) -> float:
    """
    Compute transmission delay (milliseconds).

    Formula:  t_tx = DataSize / Bandwidth

    Parameters
    ----------
    data_size_mb   : Payload size in megabytes.
    bandwidth_mbps : Available channel bandwidth in Mb/s.

    Returns
    -------
    Transmission delay in milliseconds.
    """
    if bandwidth_mbps <= 0:
        raise ValueError("Bandwidth must be positive.")
    # Convert MB → Mb (×8), then divide by Mbps → seconds, then ×1000 → ms
    data_mb = data_size_mb * 8  # megabits
    return (data_mb / bandwidth_mbps) * 1_000


def processing_delay(congestion_factor: float) -> float:
    """
    Compute processing delay (milliseconds).

    Models on-board transponder and routing hardware latency, which
    increases under high congestion due to CPU/DSP load.

    Parameters
    ----------
    congestion_factor : 0 (idle) → 1 (fully congested).

    Returns
    -------
    Processing delay in milliseconds.
    """
    # Linear scaling: congestion adds up to 3× the base processing time
    return PROCESSING_DELAY_BASE_MS * (1.0 + 2.0 * congestion_factor)


def queuing_delay(congestion_factor: float) -> float:
    """
    Compute queuing delay (milliseconds).

    Approximates M/M/1 queue behaviour: delay grows super-linearly as
    the channel approaches saturation.

    Formula:  t_queue = MAX_QUEUING × congestion²
    (quadratic growth chosen as a simplified heavy-traffic approximation)

    Parameters
    ----------
    congestion_factor : 0 (idle) → 1 (fully congested).

    Returns
    -------
    Queuing delay in milliseconds.
    """
    return MAX_QUEUING_DELAY_MS * (congestion_factor ** 2)


def total_delay(distance_m: float,
                data_size_mb: float,
                bandwidth_mbps: float,
                weather_factor: float,
                congestion_factor: float) -> dict:
    """
    Compute the complete end-to-end communication delay.

    Total = Propagation + Transmission + Processing + Queuing

    Parameters
    ----------
    distance_m        : 3-D path distance in metres.
    data_size_mb      : Data payload in MB.
    bandwidth_mbps    : Channel bandwidth in Mbps.
    weather_factor    : 0–1 atmospheric severity.
    congestion_factor : 0–1 network load level.

    Returns
    -------
    dict with individual component delays and 'total_delay_ms'.
    """
    t_prop  = propagation_delay(distance_m, weather_factor)
    t_tx    = transmission_delay(data_size_mb, bandwidth_mbps)
    t_proc  = processing_delay(congestion_factor)
    t_queue = queuing_delay(congestion_factor)
    t_total = t_prop + t_tx + t_proc + t_queue

    return {
        "propagation_delay_ms" : round(t_prop,  4),
        "transmission_delay_ms": round(t_tx,    4),
        "processing_delay_ms"  : round(t_proc,  4),
        "queuing_delay_ms"     : round(t_queue, 4),
        "total_delay_ms"       : round(t_total, 4),
    }


def orbit_type_from_altitude(altitude_km: float) -> str:
    """Return the orbit category (LEO / MEO / GEO) for a given altitude."""
    if altitude_km <= 2_000:
        return "LEO"
    elif altitude_km <= 35_786:
        return "MEO"
    else:
        return "GEO"
