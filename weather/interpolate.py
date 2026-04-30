"""
weather/interpolate.py

Objective analysis: scattered station observations → regular lat/lon grid.
Supports Barnes (default) and Cressman weight functions.

Barnes:   w = exp(-d² / κ²)          — converges well with sparse networks
Cressman: w = (R²-d²) / (R²+d²)     — smoother, better for mission planning

Length scale κ (Barnes) and radius R (Cressman) are auto-computed from
mean inter-station separation; override via the `length_scale` parameter.

Path traversal additions (used by mission/traverse.py)
-------------------------------------------------------
interpolate_obs_at_time  — temporal interpolation across a station's obs list
query_point              — evaluate objective analysis at a single lat/lon
compute_length_scale     — precompute κ once for reuse across many point queries
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np


# ── weight functions ──────────────────────────────────────────────────────────

def _barnes(d: np.ndarray, kappa: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (kappa ** 2))


def _cressman(d: np.ndarray, R: float) -> np.ndarray:
    w = (R ** 2 - d ** 2) / (R ** 2 + d ** 2)
    return np.where(d < R, w, 0.0)


# ── main grid interface ───────────────────────────────────────────────────────

def interpolate_field(
    station_obs: dict,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    method: str = "barnes",
    length_scale: float | None = None,
) -> dict:
    """
    Interpolate a snapshot of station observations onto a 2-D grid.

    Parameters
    ----------
    station_obs  : {stn: obs_dict}  — obs_dict must have lat, lon, wdir, wspd;
                                      gust and temp are optional.
    grid_lat     : 2-D array of grid latitudes
    grid_lon     : 2-D array of grid longitudes
    method       : 'barnes' | 'cressman'
    length_scale : override automatic κ/R computation (degrees)

    Returns
    -------
    dict with 2-D arrays: U, V, Ug, Vg, T, speed, gust_speed
    U/V       wind components (FROM convention, m/s)
    Ug/Vg     gust components
    T         temperature (°C)
    speed     scalar wind speed
    gust_speed scalar gust speed
    """
    stns = list(station_obs.values())
    lats = np.array([s["lat"] for s in stns])
    lons = np.array([s["lon"] for s in stns])

    dirs = np.radians([s["wdir"] for s in stns])
    spds = np.array([s["wspd"] for s in stns])
    gusts = np.array([s.get("gust") or s["wspd"] for s in stns])
    temps = np.array([s.get("temp") if s.get("temp") is not None else 15.0 for s in stns])

    # Wind FROM → Cartesian (u points East, v points North)
    U  = -spds  * np.sin(dirs)
    V  = -spds  * np.cos(dirs)
    Ug = -gusts * np.sin(dirs)
    Vg = -gusts * np.cos(dirs)

    # Auto length scale: half mean inter-station distance
    if length_scale is None:
        n = len(lats)
        if n > 1:
            pairs = [
                np.sqrt((lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2)
                for i in range(n) for j in range(i + 1, n)
            ]
            kappa = float(np.mean(pairs)) * 0.5
        else:
            kappa = 1.0
    else:
        kappa = length_scale
    R = kappa * 2.0

    shape = grid_lat.shape
    out = {k: np.zeros(shape) for k in ("U", "V", "Ug", "Vg", "T")}

    for i in range(shape[0]):
        for j in range(shape[1]):
            d = np.sqrt(
                (lats - grid_lat[i, j]) ** 2 + (lons - grid_lon[i, j]) ** 2
            )
            w = _barnes(d, kappa) if method == "barnes" else _cressman(d, R)
            ws = w.sum()
            if ws < 1e-12:
                continue
            w /= ws
            out["U"][i, j]  = w @ U
            out["V"][i, j]  = w @ V
            out["Ug"][i, j] = w @ Ug
            out["Vg"][i, j] = w @ Vg
            out["T"][i, j]  = w @ temps

    out["speed"]      = np.hypot(out["U"],  out["V"])
    out["gust_speed"] = np.hypot(out["Ug"], out["Vg"])
    return out


def make_grid(extent: list, resolution: int = 20):
    """
    Build a regular lat/lon grid over an extent.

    Parameters
    ----------
    extent     : [lon_min, lon_max, lat_min, lat_max]
    resolution : points per degree

    Returns
    -------
    grid_lat, grid_lon as 2-D arrays
    """
    lons = np.linspace(extent[0], extent[1], int((extent[1] - extent[0]) * resolution))
    lats = np.linspace(extent[2], extent[3], int((extent[3] - extent[2]) * resolution))
    return np.meshgrid(lats, lons, indexing="ij")


# ── path traversal interface ──────────────────────────────────────────────────

def compute_length_scale(station_obs: dict, method: str = "barnes") -> float:
    """
    Compute the Barnes κ (or Cressman R) from station positions.
    Call once per station network; pass result to query_point for efficiency.
    """
    stns = list(station_obs.values())
    lats = np.array([s["lat"] for s in stns])
    lons = np.array([s["lon"] for s in stns])
    n = len(lats)
    if n < 2:
        return 1.0
    pairs = [
        np.sqrt((lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2)
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(pairs)) * 0.5


def _lerp_scalar(a, b, alpha: float):
    """Linear interpolation; returns None only if both inputs are None."""
    if a is None and b is None:
        return None
    a = a if a is not None else b
    b = b if b is not None else a
    return a + alpha * (b - a)


def _lerp_obs(o1: dict, o2: dict, alpha: float) -> dict:
    """
    Linearly interpolate between two obs dicts.
    alpha=0 → o1, alpha=1 → o2.

    Wind interpolated in U/V component space to handle directional wraparound.
    Sky condition (categorical) taken from the temporally nearer observation.
    """
    # Wind: U/V interpolation
    d1 = math.radians(o1["wdir"])
    d2 = math.radians(o2["wdir"])
    u1, v1 = -o1["wspd"] * math.sin(d1), -o1["wspd"] * math.cos(d1)
    u2, v2 = -o2["wspd"] * math.sin(d2), -o2["wspd"] * math.cos(d2)
    u = u1 + alpha * (u2 - u1)
    v = v1 + alpha * (v2 - v1)
    wspd = math.sqrt(u ** 2 + v ** 2)
    wdir = (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0

    # Gust: treat missing as equal to wind speed
    g1 = o1.get("gust") or o1["wspd"]
    g2 = o2.get("gust") or o2["wspd"]
    gust = g1 + alpha * (g2 - g1)

    # Sky conditions: categorical — use closer obs
    sky_src = o1 if alpha < 0.5 else o2

    return {
        "time":  o1["time"],
        "lat":   o1["lat"],
        "lon":   o1["lon"],
        "wdir":  wdir,
        "wspd":  wspd,
        "gust":  gust,
        "temp":  _lerp_scalar(o1.get("temp"),  o2.get("temp"),  alpha),
        "dwpt":  _lerp_scalar(o1.get("dwpt"),  o2.get("dwpt"),  alpha),
        "alti":  _lerp_scalar(o1.get("alti"),  o2.get("alti"),  alpha),
        "vsby":  _lerp_scalar(o1.get("vsby"),  o2.get("vsby"),  alpha),
        "skyc1": sky_src.get("skyc1"), "skyl1": sky_src.get("skyl1"),
        "skyc2": sky_src.get("skyc2"), "skyl2": sky_src.get("skyl2"),
        "skyc3": sky_src.get("skyc3"), "skyl3": sky_src.get("skyl3"),
    }


def interpolate_obs_at_time(obs_data: dict, t: datetime) -> dict:
    """
    For each station in obs_data, interpolate to time t.

    Parameters
    ----------
    obs_data : {stn: [obs_dict, ...]}  full time-series, sorted ascending
    t        : target UTC datetime

    Returns
    -------
    {stn: obs_dict}  — one interpolated snapshot per station.
    Clamps to the first/last obs if t is outside the data window.
    """
    snapshot: dict[str, dict] = {}

    for stn, obs_list in obs_data.items():
        if not obs_list:
            continue

        # Walk the sorted list to find the bracketing pair
        before: dict | None = None
        after:  dict | None = None

        for obs in obs_list:
            if obs["time"] <= t:
                before = obs
            else:
                after = obs
                break

        if before is None:
            # t is before all data — clamp to first obs
            snapshot[stn] = after  # type: ignore[assignment]
        elif after is None:
            # t is after all data — clamp to last obs
            snapshot[stn] = before
        else:
            dt_span = (after["time"] - before["time"]).total_seconds()
            alpha = (
                (t - before["time"]).total_seconds() / dt_span
                if dt_span > 0 else 0.0
            )
            snapshot[stn] = _lerp_obs(before, after, alpha)

    return snapshot


def query_point(
    station_snapshot: dict,
    lat: float,
    lon: float,
    method: str = "barnes",
    length_scale: float | None = None,
) -> dict:
    """
    Evaluate objective analysis at a single (lat, lon) point.

    More efficient than interpolate_field for path traversal — no grid
    allocation, O(n_stations) per call.

    Parameters
    ----------
    station_snapshot : {stn: obs_dict}  — one obs per station (use
                        interpolate_obs_at_time to get this)
    lat, lon         : query point (WGS84 degrees)
    method           : 'barnes' | 'cressman'
    length_scale     : precomputed κ (pass from compute_length_scale to avoid
                        recomputing on every call)

    Returns
    -------
    dict with scalar fields:
      U, V          wind components (m/s, NED: U=East-ish, V=North-ish — Cartesian)
      Ug, Vg        gust components
      speed         scalar mean wind (m/s)
      gust_speed    scalar gust speed (m/s)
      wdir          wind direction FROM (degrees, met convention)
      temp          temperature (°C)
      pressure_hpa  surface pressure (hPa)
      vsby          visibility (statute miles)
      ceiling_ft    lowest BKN/OVC layer (ft AGL), or None
    """
    stns = list(station_snapshot.values())
    if not stns:
        return {}

    lats = np.array([s["lat"] for s in stns])
    lons = np.array([s["lon"] for s in stns])

    dirs  = np.array([math.radians(s["wdir"]) for s in stns])
    spds  = np.array([s["wspd"] for s in stns])
    gusts = np.array([s.get("gust") or s["wspd"] for s in stns])
    temps = np.array([
        s.get("temp") if s.get("temp") is not None else 15.0 for s in stns
    ])
    # Pressure: altimeter (inHg) → hPa; fall back to standard atmosphere
    pressures = np.array([
        s.get("alti") * 33.8639 if s.get("alti") is not None else 1013.25
        for s in stns
    ])
    vsbys = np.array([
        s.get("vsby") if s.get("vsby") is not None else 10.0 for s in stns
    ])

    # Wind FROM → Cartesian
    U  = -spds  * np.sin(dirs)
    V  = -spds  * np.cos(dirs)
    Ug = -gusts * np.sin(dirs)
    Vg = -gusts * np.cos(dirs)

    # Length scale
    if length_scale is None:
        length_scale = compute_length_scale(station_snapshot, method)
    kappa = length_scale
    R     = kappa * 2.0

    # Distance from query point to each station
    d = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2)

    w = _barnes(d, kappa) if method == "barnes" else _cressman(d, R)
    ws = w.sum()

    if ws < 1e-12:
        # All stations are very distant — fall back to nearest
        nearest = int(np.argmin(d))
        u_q  = float(U[nearest])
        v_q  = float(V[nearest])
        ug_q = float(Ug[nearest])
        vg_q = float(Vg[nearest])
        t_q  = float(temps[nearest])
        p_q  = float(pressures[nearest])
        vis_q = float(vsbys[nearest])
        near_stn = stns[nearest]
    else:
        w    = w / ws
        u_q  = float(w @ U)
        v_q  = float(w @ V)
        ug_q = float(w @ Ug)
        vg_q = float(w @ Vg)
        t_q  = float(w @ temps)
        p_q  = float(w @ pressures)
        vis_q = float(w @ vsbys)
        # Nearest station for categorical fields
        near_stn = stns[int(np.argmin(d))]

    speed      = math.sqrt(u_q ** 2 + v_q ** 2)
    gust_speed = math.sqrt(ug_q ** 2 + vg_q ** 2)
    wdir       = (math.degrees(math.atan2(-u_q, -v_q)) + 360.0) % 360.0

    # Ceiling: lowest BKN or OVC layer from nearest station
    ceiling_ft = None
    for skyc_k, skyl_k in [("skyc1","skyl1"),("skyc2","skyl2"),("skyc3","skyl3")]:
        c = near_stn.get(skyc_k)
        h = near_stn.get(skyl_k)
        if c in ("BKN", "OVC") and h is not None:
            ceiling_ft = h * 100.0   # hundreds of feet → feet
            break

    return {
        "U":            u_q,
        "V":            v_q,
        "Ug":           ug_q,
        "Vg":           vg_q,
        "speed":        speed,
        "gust_speed":   gust_speed,
        "wdir":         wdir,
        "temp":         t_q,
        "pressure_hpa": p_q,
        "vsby":         vis_q,
        "ceiling_ft":   ceiling_ft,
    }
