"""
run.py — Weatherboy CLI entry point

Standard usage (animation)
---------------------------
python run.py [options]

Mission traversal usage
-----------------------
python run.py --config config/virginia.xml --path config/paths/VA-XC1.kml [options]

Traversal writes a CSV to --output (default: output/traversal.csv) and prints
a summary. The animation still runs unless --no-animate is passed.

Options
-------
--config      PATH    Simulation config XML         (default: config/sim_config.xml)
--speed       FLOAT   Playback speed: sim-hrs/sec   (default: 1.0)
--interp      STR     Spatial interp: barnes|cressman (default: barnes)
--temporal    STR     Temporal mode: linear|step    (default: linear)
--gust        STR     Gust mode: envelope|reported  (default: envelope)
--map         STR     Basemap: topo|satellite|street|vfr|none (default: topo)
--voronoi            Enable Voronoi overlay
--blend              Smooth interpolated wind field

Traversal options (require --path)
-----------------------------------
--path        PATH    KML file with LineString path
--alt-agl     FLOAT   Flight altitude AGL, feet     (default: 350)
--speed-kmh   FLOAT   Cruise ground speed, km/h     (default: 70)
--traverse-dt FLOAT   Traversal time step, seconds  (default: 10)
--output      PATH    CSV output file               (default: output/traversal.csv)
--log-profile         Apply log wind profile for altitude correction
--no-animate         Skip animation after traversal

Config schema
-------------
<simulation>
  <start>2026-04-21T05:00:00Z</start>
  <end>2026-04-22T05:00:00Z</end>

  <!-- Option A: KML file -->
  <control_volume kml="config/maps/Virginia.kml"/>

  <!-- Option B: explicit vertices -->
  <control_volume>
    <vertex lon="-76.30" lat="36.85"/>
    ...
  </control_volume>

  <!-- Option C: omit — auto convex hull of stations -->

  <stations>
    <station icao="KCHO"/>
    ...
  </stations>
</simulation>
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import ConvexHull

from weather.fetch import fetch_metars
from weather.fetch_buoys import fetch_buoys
from viz.animate import run_animation


# ── KML parser ────────────────────────────────────────────────────────────────

def _parse_kml_polygon(path: str) -> np.ndarray:
    """Extract first polygon from KML. Returns (N,2) array of (lon, lat)."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns   = "http://www.opengis.net/kml/2.2"

    coords_el = root.find(f".//{{{ns}}}coordinates")
    if coords_el is None:
        coords_el = root.find(".//coordinates")
    if coords_el is None:
        raise ValueError(f"No <coordinates> element found in {path}")

    pts = []
    for token in coords_el.text.strip().split():
        parts = token.split(",")
        pts.append([float(parts[0]), float(parts[1])])   # lon, lat
    return np.array(pts)


# ── config parser ─────────────────────────────────────────────────────────────

def parse_config(path: str):
    """
    Returns (start, end, stations, buoys, cv_polygon)
      cv_polygon : (N,2) ndarray of (lon,lat) or None (→ auto hull)
    """
    tree = ET.parse(path)
    root = tree.getroot()

    def parse_dt(s):
        return datetime.fromisoformat(
            s.strip().replace("Z", "+00:00")
        ).astimezone(timezone.utc)

    start    = parse_dt(root.find("start").text)
    end      = parse_dt(root.find("end").text)
    stations = [el.get("icao") for el in root.findall(".//station")]

    if not stations:
        raise ValueError("No <station icao=...> elements found in config.")

    # ── control volume ────────────────────────────────────────────────────────
    cv_el = root.find("control_volume")
    cv_polygon = None

    if cv_el is not None:
        kml_path = cv_el.get("kml")
        if kml_path:
            cv_polygon = _parse_kml_polygon(kml_path)
        else:
            verts = cv_el.findall("vertex")
            if verts:
                cv_polygon = np.array(
                    [[float(v.get("lon")), float(v.get("lat"))] for v in verts]
                )

    buoys = [el.get("id") for el in root.findall(".//buoy")]
    return start, end, stations, buoys, cv_polygon


# ── traversal runner ──────────────────────────────────────────────────────────

def run_mission_traversal(args, obs_data: dict, t_start: datetime) -> None:
    """Load path, run traversal, write CSV, print summary."""
    from mission.path import parse_kml_path, interpolate_path, path_summary
    from mission.traverse import run_traversal
    from mission.logger import write_csv, print_summary

    print(f"  Loading path  : {args.path}")
    try:
        waypoints = parse_kml_path(args.path)
    except Exception as e:
        print(f"  [ERROR] Failed to parse path: {e}", file=sys.stderr)
        return

    speed_ms  = args.speed_kmh / 3.6
    alt_agl_m = args.alt_agl * 0.3048   # feet → metres

    summary = path_summary(waypoints, speed_ms, args.traverse_dt)
    print(f"  Waypoints     : {summary['waypoints']}")
    print(f"  Path length   : {summary['total_dist_km']:.1f} km")
    print(f"  Flight time   : {summary['total_time_min']:.1f} min  "
          f"({summary['total_time_min']/60:.2f} hr)")
    print(f"  Cruise speed  : {summary['speed_kmh']:.0f} km/h  "
          f"({speed_ms:.1f} m/s)")
    print(f"  Sample dt     : {summary['dt_s']:.0f} s  "
          f"→ {summary['n_samples']} samples")
    print(f"  Alt AGL       : {args.alt_agl:.0f} ft  ({alt_agl_m:.1f} m)")
    print(f"  Log profile   : {'yes' if args.log_profile else 'no'}")
    print()

    path_points = interpolate_path(waypoints, speed_ms, args.traverse_dt, t_start)

    print("  Running traversal...")
    records = run_traversal(
        path_points      = path_points,
        obs_data         = obs_data,
        alt_agl_m        = alt_agl_m,
        cruise_speed_ms  = speed_ms,
        spatial_method   = args.interp,
        gust_mode        = args.gust,
        log_profile      = args.log_profile,
        dt               = args.traverse_dt,
    )

    write_csv(records, args.output)
    print_summary(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Weatherboy — UAV weather environment simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Existing args ─────────────────────────────────────────────────────────
    p.add_argument("--config",   default="config/sim_config.xml")
    p.add_argument("--speed",    type=float, default=1.0,  metavar="FLOAT",
                   help="Animation playback speed (sim-hrs/sec)")
    p.add_argument("--interp",   choices=["barnes", "cressman"], default="barnes")
    p.add_argument("--temporal", choices=["linear", "step"],     default="linear")
    p.add_argument("--gust",     choices=["envelope", "reported"], default="envelope")
    p.add_argument("--map",      choices=["topo", "satellite", "street", "vfr", "none"],
                   default="topo")
    p.add_argument("--voronoi",  action="store_true")
    p.add_argument("--blend",    action="store_true",
                   help="Smooth interpolated wind field")

    # ── Traversal args ────────────────────────────────────────────────────────
    p.add_argument("--path",        default=None, metavar="KML",
                   help="KML LineString path for mission traversal")
    p.add_argument("--alt-agl",     type=float, default=350.0, metavar="FEET",
                   help="Flight altitude AGL in feet (default: 350)")
    p.add_argument("--speed-kmh",   type=float, default=70.0,  metavar="KMH",
                   help="Cruise ground speed in km/h (default: 70)")
    p.add_argument("--traverse-dt", type=float, default=10.0,  metavar="SEC",
                   help="Traversal time step in seconds (default: 10)")
    p.add_argument("--output",      default="output/traversal.csv", metavar="CSV",
                   help="CSV output path (default: output/traversal.csv)")
    p.add_argument("--log-profile", action="store_true",
                   help="Apply log wind profile to scale from 10m obs to flight alt")
    p.add_argument("--no-animate",  action="store_true",
                   help="Skip animation (useful when only running traversal)")

    args = p.parse_args()

    try:
        start, end, stations, buoys, cv_polygon = parse_config(args.config)
    except Exception as e:
        print(f"[ERROR] Config parse failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Stations  : {stations}")
    print(f"  Window    : {start.strftime('%Y-%m-%d %H:%MZ')}  ->  "
          f"{end.strftime('%Y-%m-%d %H:%MZ')}")
    print(f"  CV polygon: {len(cv_polygon)} vertices"
          if cv_polygon is not None
          else "  CV polygon: auto (convex hull)")
    print(f"  Interp    : {args.interp}")
    print(f"  Temporal  : {args.temporal}")
    print(f"  Gust mode : {args.gust}")
    if not args.path:
        print(f"  Speed     : {args.speed}x (sim-hrs/sec)")
    print()
    print("  Fetching METARs from Iowa State Mesonet...")

    try:
        obs_data = fetch_metars(stations, start, end)
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Observations loaded: { {s: len(v) for s, v in obs_data.items()} }")

    if buoys:
        print(f"  Fetching {len(buoys)} buoy station(s)...")
        buoy_data = fetch_buoys(buoys, start, end, metar_data=obs_data)
        obs_data.update(buoy_data)
        print(f"  Active obs sources: {list(obs_data.keys())}")
    print()

    # Auto convex hull if no CV provided
    if cv_polygon is None:
        coords = np.array(
            [[obs_data[s][0]["lon"], obs_data[s][0]["lat"]] for s in obs_data]
        )
        hull      = ConvexHull(coords)
        cv_polygon = coords[hull.vertices]
        print("  CV polygon: auto convex hull from station coordinates")

    # ── Mission traversal ─────────────────────────────────────────────────────
    if args.path:
        print()
        run_mission_traversal(args, obs_data, start)

    # ── Animation ─────────────────────────────────────────────────────────────
    if not args.no_animate:
        run_animation(
            obs_data       = obs_data,
            start          = start,
            end            = end,
            playback_speed = args.speed,
            spatial_method = args.interp,
            temporal_mode  = args.temporal,
            gust_mode      = args.gust,
            map_style      = args.map,
            voronoi        = args.voronoi,
            blend          = args.blend,
            cv_polygon     = cv_polygon,
        )


if __name__ == "__main__":
    main()
