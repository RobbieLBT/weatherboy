"""
run.py — Poseidon-01 CLI entry point

Usage
-----
python run.py [options]

Options
-------
--config    PATH   Simulation config XML  (default: config/sim_config.xml)
--speed     FLOAT  Playback speed: sim-hours per real second  (default: 1.0)
--interp    STR    Spatial interpolation: barnes | cressman   (default: barnes)
--temporal  STR    Temporal mode: linear | step               (default: linear)
--gust      STR    Gust mode: envelope | reported             (default: envelope)
--map       STR    Basemap: topo | satellite | street | vfr   (default: topo)
--voronoi          Enable Voronoi overlay

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
    Returns (start, end, stations, cv_polygon)
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

    return start, end, stations, cv_polygon


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Poseidon-01 — Weather environment simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config",   default="config/sim_config.xml")
    p.add_argument("--speed",    type=float, default=1.0,  metavar="FLOAT")
    p.add_argument("--interp",   choices=["barnes", "cressman"], default="barnes")
    p.add_argument("--temporal", choices=["linear", "step"],     default="linear")
    p.add_argument("--gust",     choices=["envelope", "reported"], default="envelope")
    p.add_argument("--map",      choices=["topo", "satellite", "street", "vfr", "none"], default="topo")
    p.add_argument("--voronoi",  action="store_true")
    p.add_argument("--blend",    action="store_true",
                   help="Smooth interpolated wind field; de-emphasises quivers")
    args = p.parse_args()

    try:
        start, end, stations, cv_polygon = parse_config(args.config)
    except Exception as e:
        print(f"[ERROR] Config parse failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Stations  : {stations}")
    print(f"  Window    : {start.strftime('%Y-%m-%d %H:%MZ')}  ->  {end.strftime('%Y-%m-%d %H:%MZ')}")
    print(f"  CV polygon: {len(cv_polygon)} vertices" if cv_polygon is not None else "  CV polygon: auto (convex hull)")
    print(f"  Interp    : {args.interp}")
    print(f"  Temporal  : {args.temporal}")
    print(f"  Gust mode : {args.gust}")
    print(f"  Speed     : {args.speed}x (sim-hrs/sec)")
    print()
    print("  Fetching METARs from Iowa State Mesonet...")

    try:
        obs_data = fetch_metars(stations, start, end)
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Observations loaded: { {s: len(v) for s, v in obs_data.items()} }")
    print()

    # Auto convex hull if no CV provided
    if cv_polygon is None:
        coords = np.array(
            [[obs_data[s][0]["lon"], obs_data[s][0]["lat"]] for s in obs_data]
        )
        hull   = ConvexHull(coords)
        cv_polygon = coords[hull.vertices]
        print("  CV polygon: auto convex hull from station coordinates")

    run_animation(
        obs_data=obs_data,
        start=start,
        end=end,
        playback_speed=args.speed,
        spatial_method=args.interp,
        temporal_mode=args.temporal,
        gust_mode=args.gust,
        map_style=args.map,
        voronoi=args.voronoi,
        blend=args.blend,
        cv_polygon=cv_polygon,
    )


if __name__ == "__main__":
    main()