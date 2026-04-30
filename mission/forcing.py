"""
mission/forcing.py

ForcingRecord: environmental state at a single path sample point.

This is the contract between the weather/environment layer and the
vehicle physics/control layer. Fields marked None are stubs that will
be populated once the VSPAero coupling is defined.

Coordinate conventions
----------------------
- Positions: WGS84, degrees
- Altitude: meters AGL
- Wind: NED frame (North-East-Down), m/s
- Heading/bearing: degrees true, 0=North, clockwise
- Wind direction: meteorological FROM convention (degrees true)
- Temperature: degrees Celsius
- Pressure: hPa
- Density: kg/m³
- Visibility: statute miles
- Ceiling: feet AGL
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, fields
from datetime import datetime


@dataclass
class ForcingRecord:
    # ── Position ──────────────────────────────────────────────────────────────
    time:               datetime    # UTC
    lat_deg:            float       # WGS84
    lon_deg:            float       # WGS84
    alt_agl_m:          float       # meters AGL

    # ── Path geometry ─────────────────────────────────────────────────────────
    heading_deg:        float       # degrees true (from path tangent)
    dist_from_start_km: float       # cumulative ground distance

    # ── Mean wind — NED frame, from spatial interpolation ────────────────────
    wind_n_ms:          float       # positive = from south (northward flow)
    wind_e_ms:          float       # positive = from west (eastward flow)
    wind_speed_ms:      float       # scalar |wind|
    wind_dir_deg:       float       # FROM, met convention

    # ── Gust perturbation — OU model applied to mean ──────────────────────────
    gust_n_ms:          float       # instantaneous N component
    gust_e_ms:          float       # instantaneous E component
    gust_speed_ms:      float       # scalar instantaneous speed

    # ── Mission-relative wind (resolved onto vehicle heading) ─────────────────
    headwind_ms:        float       # positive = headwind, negative = tailwind
    crosswind_ms:       float       # positive = from starboard (right)

    # ── VSPAero coupling stubs ────────────────────────────────────────────────
    # Populated once vehicle speed and full aero model are defined.
    # delta_alpha: requires vertical wind component (no model yet).
    # delta_beta: computed here as a sideslip proxy from crosswind/cruise_speed.
    delta_beta_deg:     float | None    # sideslip perturbation proxy
    delta_alpha_deg:    float | None    # AoA perturbation (None: no vertical wind)

    # ── Atmosphere ────────────────────────────────────────────────────────────
    temp_c:             float | None
    pressure_hpa:       float | None
    density_kgm3:       float | None    # derived: P / (Rd * T)

    # ── Operational environment ───────────────────────────────────────────────
    visibility_sm:      float | None
    ceiling_ft:         float | None
    flight_category:    str   | None    # VFR / MVFR / IFR / LIFR

    # ── Metadata ──────────────────────────────────────────────────────────────
    interp_method:      str  = "barnes"
    gust_mode:          str  = "envelope"
    log_profile_applied: bool = False


# ── CSV helpers ───────────────────────────────────────────────────────────────

def csv_header() -> list[str]:
    return [f.name for f in fields(ForcingRecord)]


def record_to_row(r: ForcingRecord) -> list:
    row = []
    for f in fields(r):
        v = getattr(r, f.name)
        if isinstance(v, datetime):
            row.append(v.strftime("%Y-%m-%dT%H:%M:%SZ"))
        elif isinstance(v, float):
            row.append(f"{v:.6f}")
        elif v is None:
            row.append("")
        else:
            row.append(str(v))
    return row
