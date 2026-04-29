"""
weather/interpolate.py

Objective analysis: scattered station observations → regular lat/lon grid.
Supports Barnes (default) and Cressman weight functions.

Barnes:   w = exp(-d² / κ²)          — converges well with sparse networks
Cressman: w = (R²-d²) / (R²+d²)     — smoother, better for mission planning

Length scale κ (Barnes) and radius R (Cressman) are auto-computed from
mean inter-station separation; override via the `length_scale` parameter.
"""

import numpy as np


# ── weight functions ──────────────────────────────────────────────────────────

def _barnes(d: np.ndarray, kappa: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (kappa ** 2))


def _cressman(d: np.ndarray, R: float) -> np.ndarray:
    w = (R ** 2 - d ** 2) / (R ** 2 + d ** 2)
    return np.where(d < R, w, 0.0)


# ── main interface ────────────────────────────────────────────────────────────

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
