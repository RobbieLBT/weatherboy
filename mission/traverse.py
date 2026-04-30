"""
mission/traverse.py

PathTraversal: walks a time-sampled path through the weather environment
and produces a ForcingRecord at each sample point.

Pipeline per sample
-------------------
1. Temporal interpolation — get obs snapshot at time t
2. Spatial query — evaluate objective analysis at (lat, lon)
3. Altitude correction — apply log wind profile if requested
4. Gust sampling — advance online OU process
5. Derived quantities — density, flight category, headwind/crosswind
6. Assemble ForcingRecord

Notes
-----
- Length scale is precomputed from the full obs_data (any snapshot has the
  same station positions), so it only changes if stations change.
- The OU gust sampler maintains state across samples; the gust field is
  spatially correlated along the path.
- If the path extends beyond the simulation time window, obs are clamped to
  the nearest edge. A warning is printed if this occurs.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from weather.interpolate import (
    compute_length_scale,
    interpolate_obs_at_time,
    query_point,
)
from weather.turbulence import GustSampler, log_wind_profile
from mission.forcing import ForcingRecord


# ── Atmosphere ────────────────────────────────────────────────────────────────

_RD = 287.05          # specific gas constant for dry air, J/(kg·K)
_METAR_Z_REF = 10.0   # METAR wind obs height, metres AGL


def _air_density(temp_c: float | None, pressure_hpa: float | None) -> float | None:
    if temp_c is None or pressure_hpa is None:
        return None
    T = temp_c + 273.15           # K
    P = pressure_hpa * 100.0      # Pa
    return P / (_RD * T)


def _flight_category(ceiling_ft: float | None, vsby_sm: float | None) -> str | None:
    if ceiling_ft is None and vsby_sm is None:
        return None
    c = ceiling_ft if ceiling_ft is not None else float("inf")
    v = vsby_sm    if vsby_sm    is not None else float("inf")
    if c < 500  or v < 1:  return "LIFR"
    if c < 1000 or v < 3:  return "IFR"
    if c < 3000 or v < 5:  return "MVFR"
    return "VFR"


# ── Mission-relative wind ─────────────────────────────────────────────────────

def _resolve_wind(
    u_e: float, v_n: float,
    heading_deg: float,
) -> tuple[float, float]:
    """
    Resolve wind (East=u, North=v) into headwind and crosswind components
    relative to vehicle heading.

    Returns (headwind_ms, crosswind_ms)
    headwind  : positive = opposing vehicle motion (headwind)
    crosswind : positive = wind from starboard (right side)
    """
    h_rad = math.radians(heading_deg)
    # Unit vector in direction of vehicle motion (NED, ignoring Down)
    fwd_n =  math.cos(h_rad)
    fwd_e =  math.sin(h_rad)
    # Right-perpendicular
    rgt_n = -math.sin(h_rad)
    rgt_e =  math.cos(h_rad)

    # Headwind: wind opposing forward motion → project wind onto -fwd
    headwind   = -(u_e * fwd_e + v_n * fwd_n)
    crosswind  =   u_e * rgt_e + v_n * rgt_n

    return headwind, crosswind


# ── Main traversal ────────────────────────────────────────────────────────────

def run_traversal(
    path_points: list[tuple[datetime, float, float, float, float]],
    obs_data:    dict,
    alt_agl_m:   float = 106.7,    # 350 ft AGL default
    cruise_speed_ms: float = 19.4, # for delta_beta proxy
    spatial_method: str  = "barnes",
    gust_mode:      str  = "envelope",
    log_profile:    bool = False,
    dt:             float = 10.0,
    seed:           int | None = None,
) -> list[ForcingRecord]:
    """
    Walk path_points through the weather environment, producing one
    ForcingRecord per sample.

    Parameters
    ----------
    path_points     : output of mission.path.interpolate_path
                      [(t, lat, lon, heading_deg, dist_m), ...]
    obs_data        : full time-series obs dict from fetch_metars
    alt_agl_m       : flight altitude AGL (metres)
    cruise_speed_ms : cruise airspeed for sideslip proxy (m/s)
    spatial_method  : 'barnes' | 'cressman'
    gust_mode       : 'envelope' (OU sampler) | 'reported' (raw METAR gust)
    log_profile     : apply log wind profile to scale obs from 10m to alt_agl_m
    dt              : traversal time step (used to seed GustSampler)
    seed            : RNG seed for reproducibility

    Returns
    -------
    List of ForcingRecord, one per path sample.
    """
    if not path_points:
        return []

    # ── Time window check ─────────────────────────────────────────────────────
    all_times = [
        obs["time"] for stn_obs in obs_data.values() for obs in stn_obs
    ]
    t_data_start = min(all_times)
    t_data_end   = max(all_times)
    t_path_start = path_points[0][0]
    t_path_end   = path_points[-1][0]

    if t_path_start < t_data_start or t_path_end > t_data_end:
        print(
            f"  [WARNING] Path time window {t_path_start:%Y-%m-%d %H:%MZ} → "
            f"{t_path_end:%Y-%m-%d %H:%MZ} extends beyond obs data "
            f"({t_data_start:%H:%MZ} → {t_data_end:%H:%MZ}). "
            f"Obs will be clamped to nearest edge."
        )

    # ── Precompute length scale from any snapshot (station positions fixed) ──
    first_snapshot = interpolate_obs_at_time(obs_data, path_points[0][0])
    length_scale   = compute_length_scale(first_snapshot, spatial_method)

    # ── Gust sampler ─────────────────────────────────────────────────────────
    gust_sampler = GustSampler(dt=dt, seed=seed)

    records: list[ForcingRecord] = []

    for (t, lat, lon, heading_deg, dist_m) in path_points:

        # 1. Temporal interpolation
        snapshot = interpolate_obs_at_time(obs_data, t)
        if not snapshot:
            continue

        # 2. Spatial query
        fld = query_point(snapshot, lat, lon, spatial_method, length_scale)
        if not fld:
            continue

        u_mean = fld["U"]       # Eastward component
        v_mean = fld["V"]       # Northward component
        ug     = fld["Ug"]
        vg     = fld["Vg"]
        mean_speed = fld["speed"]
        gust_speed = fld["gust_speed"]

        # 3. Log wind profile (optional) — scale from 10m obs to flight alt
        if log_profile and alt_agl_m > _METAR_Z_REF:
            scale = log_wind_profile(1.0, _METAR_Z_REF, alt_agl_m)
            u_mean *= scale
            v_mean *= scale
            ug     *= scale
            vg     *= scale
            mean_speed *= scale
            gust_speed *= scale

        # 4. OU gust sampler
        if gust_mode == "envelope":
            inst_speed = gust_sampler.step(mean_speed, gust_speed if gust_speed > mean_speed else None)
        else:
            # reported mode: use raw gust as instantaneous speed
            inst_speed = gust_speed

        # Gust NED components: scale mean direction to instantaneous magnitude
        if mean_speed > 1e-6:
            gust_scale = inst_speed / mean_speed
        else:
            gust_scale = 1.0
        gust_n = v_mean * gust_scale
        gust_e = u_mean * gust_scale

        # 5. Derived quantities
        headwind, crosswind = _resolve_wind(u_mean, v_mean, heading_deg)

        temp_c       = fld.get("temp")
        pressure_hpa = fld.get("pressure_hpa")
        density      = _air_density(temp_c, pressure_hpa)
        vsby         = fld.get("vsby")
        ceiling_ft   = fld.get("ceiling_ft")
        flight_cat   = _flight_category(ceiling_ft, vsby)

        # Sideslip proxy: atan2(crosswind, cruise_speed)
        if cruise_speed_ms > 0:
            delta_beta = math.degrees(math.atan2(crosswind, cruise_speed_ms))
        else:
            delta_beta = None

        # 6. Assemble record
        records.append(ForcingRecord(
            time               = t,
            lat_deg            = lat,
            lon_deg            = lon,
            alt_agl_m          = alt_agl_m,
            heading_deg        = heading_deg,
            dist_from_start_km = dist_m / 1000.0,
            wind_n_ms          = v_mean,
            wind_e_ms          = u_mean,
            wind_speed_ms      = mean_speed,
            wind_dir_deg       = fld["wdir"],
            gust_n_ms          = gust_n,
            gust_e_ms          = gust_e,
            gust_speed_ms      = inst_speed,
            headwind_ms        = headwind,
            crosswind_ms       = crosswind,
            delta_beta_deg     = delta_beta,
            delta_alpha_deg    = None,      # stub: no vertical wind model
            temp_c             = temp_c,
            pressure_hpa       = pressure_hpa,
            density_kgm3       = density,
            visibility_sm      = vsby,
            ceiling_ft         = ceiling_ft,
            flight_category    = flight_cat,
            interp_method      = spatial_method,
            gust_mode          = gust_mode,
            log_profile_applied = log_profile,
        ))

    return records
