"""
mission/path.py

Parse a KML LineString path and generate time-sampled position records
at a constant ground speed and time resolution.

Supports both VA-XC1-style Google Earth exports and future QGC exports
(QGC LineString KMLs have the same coordinate format).

Coordinate math uses flat-earth approximation for segment interpolation
(accurate to <0.1% for segment lengths under ~50 km) and haversine for
distances and bearings.
"""

from __future__ import annotations

import bisect
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


# ── KML parser ────────────────────────────────────────────────────────────────

def parse_kml_path(kml_path: str) -> list[tuple[float, float]]:
    """
    Parse the first LineString from a KML file.

    Returns
    -------
    List of (lon, lat) tuples in path order (WGS84 degrees).
    Altitude values in the KML are ignored — caller sets AGL.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = "http://www.opengis.net/kml/2.2"

    coords_el = root.find(f".//{{{ns}}}coordinates")
    if coords_el is None:
        coords_el = root.find(".//coordinates")
    if coords_el is None:
        raise ValueError(f"No <coordinates> element found in {kml_path}")

    pts: list[tuple[float, float]] = []
    for token in coords_el.text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))  # lon, lat

    if len(pts) < 2:
        raise ValueError(f"Path in {kml_path} has fewer than 2 waypoints.")

    return pts


# ── Geodetic utilities ────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters (WGS84 mean sphere)."""
    R = 6_371_000.0
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = (math.sin(d1 / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d2 / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from (lat1,lon1) to (lat2,lon2), degrees true [0,360)."""
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2r - lon1r
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


# ── Path interpolation ────────────────────────────────────────────────────────

def build_path_table(
    waypoints: list[tuple[float, float]],
) -> tuple[list, list[float], float]:
    """
    Pre-compute per-segment geometry.

    Returns
    -------
    segments     : list of (lon1,lat1, lon2,lat2, dist_m, bearing_deg)
    cum_dist     : cumulative distance at start of each segment [0, d1, d1+d2, ...]
                   length = len(segments) + 1, last value = total path distance
    total_dist_m : total path length in meters
    """
    segments = []
    cum_dist = [0.0]

    for i in range(len(waypoints) - 1):
        lon1, lat1 = waypoints[i]
        lon2, lat2 = waypoints[i + 1]
        d = haversine_m(lat1, lon1, lat2, lon2)
        b = bearing_deg(lat1, lon1, lat2, lon2)
        segments.append((lon1, lat1, lon2, lat2, d, b))
        cum_dist.append(cum_dist[-1] + d)

    return segments, cum_dist, cum_dist[-1]


def interpolate_path(
    waypoints:  list[tuple[float, float]],
    speed_ms:   float,
    dt:         float,
    t_start:    datetime,
) -> list[tuple[datetime, float, float, float, float]]:
    """
    Walk the path at constant ground speed, sampling at uniform time steps.

    Parameters
    ----------
    waypoints : [(lon, lat), ...] from parse_kml_path
    speed_ms  : ground speed in m/s
    dt        : time step in seconds
    t_start   : UTC datetime for waypoint[0]

    Returns
    -------
    List of (t, lat, lon, heading_deg, dist_m) tuples.
    One entry per dt interval plus the final waypoint.
    Heading is the bearing of the active path segment at that sample.
    """
    segments, cum_dist, total_dist_m = build_path_table(waypoints)
    total_time_s = total_dist_m / speed_ms
    n_samples = int(total_time_s / dt) + 1

    samples: list[tuple[datetime, float, float, float, float]] = []

    for k in range(n_samples):
        elapsed = k * dt
        d = min(elapsed * speed_ms, total_dist_m)
        t = t_start + timedelta(seconds=elapsed)

        # Find active segment via binary search
        seg_idx = bisect.bisect_right(cum_dist, d) - 1
        seg_idx = max(0, min(seg_idx, len(segments) - 1))

        lon1, lat1, lon2, lat2, seg_dist, seg_bearing = segments[seg_idx]
        seg_offset = d - cum_dist[seg_idx]
        frac = seg_offset / seg_dist if seg_dist > 0 else 0.0

        lat = lat1 + frac * (lat2 - lat1)
        lon = lon1 + frac * (lon2 - lon1)

        samples.append((t, lat, lon, seg_bearing, d))

    # Always include the exact final waypoint
    lon_f, lat_f = waypoints[-1]
    t_end = t_start + timedelta(seconds=total_time_s)
    samples.append((t_end, lat_f, lon_f, segments[-1][5], total_dist_m))

    return samples


# ── Summary ───────────────────────────────────────────────────────────────────

def path_summary(
    waypoints:  list[tuple[float, float]],
    speed_ms:   float,
    dt:         float,
) -> dict:
    """Return a dict of key path statistics for logging."""
    _, _, total_dist_m = build_path_table(waypoints)
    total_time_s = total_dist_m / speed_ms
    n_samples = int(total_time_s / dt) + 1
    return {
        "waypoints":      len(waypoints),
        "total_dist_km":  total_dist_m / 1000,
        "total_time_min": total_time_s / 60,
        "speed_kmh":      speed_ms * 3.6,
        "dt_s":           dt,
        "n_samples":      n_samples,
    }
