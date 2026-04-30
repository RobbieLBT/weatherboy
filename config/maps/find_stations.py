"""
find_stations.py
----------------
Finds NDBC buoy stations and METAR/ASOS stations within a KML polygon.

Outputs (derived from KML filename, relative to project root):
  config/<ao_name>.xml          — simulation config (stations/buoys populated wholesale)
  data/stations/<ao_name>.csv   — full station table for reference/GIS
  data/stations/<ao_name>.kml   — point placemarks for GIS review

Project root defaults to CWD; override with --root.

Sources:
  NDBC buoys : https://www.ndbc.noaa.gov/data/stations/station_table.txt
  METAR/ASOS : Iowa State Mesonet -> aviationweather.gov (tiered fallback)
  Offline    : --metar-csv airports.csv  (ourairports.com, free ~7 MB)

Spatial filters (combinable, --grid applied before --min-dist):
  --grid DEGREES   Keep best station per N x N degree cell (METAR > NDBC, then alpha)
  --min-dist KM    Drop stations within KM km of a higher-priority kept station

Usage:
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --grid 0.5
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --min-dist 40
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --grid 0.5 --min-dist 40
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --metar-csv airports.csv
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --root /path/to/project
"""

import argparse
import csv
import io
import math
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from shapely.geometry import Point, Polygon
    import pandas as pd
except ImportError:
    sys.exit("Missing dependencies. Run:  pip install shapely pandas")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TYPE_PRIORITY = {"METAR": 0, "NDBC_BUOY": 1}   # lower = higher priority

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,text/plain,text/csv,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

IASTATE_STATE_NETWORKS = [
    "VA_ASOS", "MD_ASOS", "DE_ASOS", "NJ_ASOS", "PA_ASOS",
    "NC_ASOS", "NY_ASOS", "DC_ASOS", "SC_ASOS", "WV_ASOS",
    "MARITIME",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. KML PARSING
# ─────────────────────────────────────────────────────────────────────────────

def load_polygon_from_kml(kml_path: Path) -> Polygon:
    """Parse the first Polygon in a KML file and return a Shapely Polygon."""
    tree = ET.parse(kml_path)
    root = tree.getroot()

    coords_el = root.find(".//{http://www.opengis.net/kml/2.2}coordinates")
    if coords_el is None:
        coords_el = root.find(".//coordinates")
    if coords_el is None:
        raise ValueError(f"No <coordinates> element found in {kml_path}")

    points = []
    for triplet in coords_el.text.strip().split():
        parts = triplet.split(",")
        points.append((float(parts[0]), float(parts[1])))   # (lon, lat)

    if points[0] != points[-1]:
        points.append(points[0])

    poly = Polygon(points)
    if not poly.is_valid:
        raise ValueError("KML polygon is not valid geometry.")
    return poly


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_ndbc_stations() -> pd.DataFrame:
    print("  [NDBC]  Fetching station list ...", end=" ", flush=True)
    raw = _get("https://www.ndbc.noaa.gov/data/stations/station_table.txt")
    rows = []
    for line in raw.splitlines():
        if line.startswith(("#", "-")) or not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        lat, lon = _parse_ndbc_location(parts[6])
        if lat is None:
            continue
        rows.append({"station_id": parts[0], "name": parts[4],
                     "lat": lat, "lon": lon, "type": "NDBC_BUOY"})
    df = pd.DataFrame(rows).drop_duplicates(subset="station_id", keep="first").reset_index(drop=True)
    print(f"OK  ({len(df)} stations)")
    return df


def _parse_ndbc_location(loc: str):
    m = re.search(r"([\d.]+)\s*([NS])\s+([\d.]+)\s*([EW])", loc, re.I)
    if not m:
        return None, None
    lat = float(m.group(1)) * (1 if m.group(2).upper() == "N" else -1)
    lon = float(m.group(3)) * (1 if m.group(4).upper() == "E" else -1)
    return lat, lon


def fetch_metar_stations(metar_csv: str = None, airport_types: set = None) -> pd.DataFrame:
    if metar_csv:
        return _load_metar_csv(metar_csv, airport_types)

    print("  [METAR] Fetching station list ...")

    rows = _try_iastate("ASOS", "Iowa State national ASOS")
    if rows:
        return _metar_df(rows)

    print("    National network blocked; trying per-state ...")
    all_rows = []
    for net in IASTATE_STATE_NETWORKS:
        all_rows.extend(_try_iastate(net, net))
    if all_rows:
        return _metar_df(all_rows)

    print("    Trying aviationweather.gov API ...")
    rows = _try_avwx_api(30, -82, 45, -65)
    if rows:
        return _metar_df(rows)

    print("    Trying ADDS legacy dataserver ...")
    rows = _try_adds_xml(30, -82, 45, -65)
    if rows:
        return _metar_df(rows)

    raise RuntimeError(
        "All METAR sources failed.\n"
        "  Download https://ourairports.com/data/airports.csv "
        "and rerun with --metar-csv airports.csv"
    )


def _try_iastate(network: str, label: str) -> list:
    url = (
        f"https://mesonet.agron.iastate.edu/sites/networks.php"
        f"?network={network}&format=csv&noheader=on"
    )
    try:
        raw = _get(url)
    except Exception as e:
        print(f"    {label}: FAIL ({e})")
        return []

    # Iowa State returns an HTML error page (not an HTTP error code) when
    # a network code is unrecognised or the request is blocked
    if raw.lstrip().startswith("<"):
        print(f"    {label}: FAIL (HTML response — network blocked or invalid)")
        return []

    rows = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 3 or row[0].startswith("#"):
            continue
        try:
            rows.append({"station_id": row[0].strip(),
                         "name": row[4].strip() if len(row) > 4 else "",
                         "lat": float(row[1]), "lon": float(row[2]),
                         "type": "METAR"})
        except ValueError:
            continue

    if rows:
        print(f"    {label}: OK ({len(rows)} stations)")
    else:
        print(f"    {label}: empty response")
    return rows


def _try_avwx_api(min_lat, min_lon, max_lat, max_lon) -> list:
    url = (
        "https://aviationweather.gov/api/data/stationinfo"
        f"?bbox={min_lat},{min_lon},{max_lat},{max_lon}&format=csv"
    )
    try:
        raw = _get(url)
    except Exception as e:
        print(f"    aviationweather.gov: FAIL ({e})")
        return []
    rows = []
    for line in raw.splitlines():
        if not line.strip() or line.startswith(("!", "#")):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            rows.append({"station_id": parts[0].strip(),
                         "name": parts[3].strip() if len(parts) > 3 else "",
                         "lat": float(parts[1]), "lon": float(parts[2]),
                         "type": "METAR"})
        except ValueError:
            continue
    if rows:
        print(f"    aviationweather.gov: OK ({len(rows)} stations)")
    return rows


def _try_adds_xml(min_lat, min_lon, max_lat, max_lon) -> list:
    url = (
        "https://www.aviationweather.gov/adds/dataserver_current/httpparam"
        f"?dataSource=metars&requestType=retrieve&format=xml"
        f"&hoursBeforeNow=1&mostRecentForEachStation=true"
        f"&boundingBox={min_lon},{min_lat},{max_lon},{max_lat}"
    )
    try:
        raw = _get(url)
    except Exception as e:
        print(f"    ADDS XML: FAIL ({e})")
        return []
    rows = []
    try:
        root = ET.fromstring(raw)
        for metar in root.iter("METAR"):
            sid = metar.findtext("station_id")
            lat = metar.findtext("latitude")
            lon = metar.findtext("longitude")
            if not (sid and lat and lon):
                continue
            rows.append({"station_id": sid.strip(),
                         "name": (metar.findtext("site") or "").strip(),
                         "lat": float(lat), "lon": float(lon),
                         "type": "METAR"})
    except ET.ParseError as e:
        print(f"    ADDS XML parse error: {e}")
        return []
    if rows:
        print(f"    ADDS XML: OK ({len(rows)} stations)")
    return rows


def _metar_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.drop_duplicates(subset="station_id").reset_index(drop=True)
    print(f"  [METAR] {len(df)} stations loaded")
    return df


def _load_metar_csv(path: str, airport_types: set = None) -> pd.DataFrame:
    """
    Load METAR stations from a local CSV (e.g. OurAirports airports.csv).
    airport_types: set of OurAirports type strings to keep.
      Default: {'large_airport', 'medium_airport'} — the ones with ASOS/METAR.
      Pass None to skip filtering (keeps all rows).
    """
    if airport_types is None:
        airport_types = {"large_airport", "medium_airport"}

    print(f"  [METAR] Loading from {path} ...", end=" ", flush=True)
    df = pd.read_csv(path, low_memory=False)

    # Filter by airport type if the column exists
    if "type" in df.columns and airport_types:
        before = len(df)
        df = df[df["type"].isin(airport_types)]
        print(f"\n    type filter ({', '.join(sorted(airport_types))}): "
              f"{before} -> {len(df)} rows", end=" ", flush=True)

    rename = {}
    _STATION_PRIORITY = ["ident", "icao", "stid", "station_id", "id"]
    for priority in _STATION_PRIORITY:
        matches = [c for c in df.columns if c.lower() == priority]
        if matches:
            rename[matches[0]] = "station_id"
            break
    for c in df.columns:
        lc = c.lower()
        if lc in ("latitude_deg", "latitude", "lat"):            rename.setdefault(c, "lat")
        elif lc in ("longitude_deg", "longitude", "lon", "long"): rename.setdefault(c, "lon")
        elif lc in ("name", "airport_name", "site"):              rename.setdefault(c, "name")
        df = df.rename(columns=rename)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    missing = {"station_id", "lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    keep = ["station_id", "lat", "lon"] + (["name"] if "name" in df.columns else [])
    df = df[keep].copy()
    if "name" not in df.columns:
        df["name"] = ""
    df["lat"]  = pd.to_numeric(df["lat"],  errors="coerce")
    df["lon"]  = pd.to_numeric(df["lon"],  errors="coerce")
    df["type"] = "METAR"
    df = df.dropna(subset=["lat", "lon"]).drop_duplicates(subset="station_id").reset_index(drop=True)
    print(f"OK  ({len(df)} stations)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SPATIAL FILTER: polygon containment
# ─────────────────────────────────────────────────────────────────────────────

def filter_within_polygon(df: pd.DataFrame, poly: Polygon) -> pd.DataFrame:
    mask = df.apply(lambda r: poly.contains(Point(r["lon"], r["lat"])), axis=1)
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPATIAL RESOLUTION FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_grid_filter(df: pd.DataFrame, cell_deg: float) -> pd.DataFrame:
    """Keep the highest-priority station per cell_deg x cell_deg grid cell."""
    df = df.copy()
    df["_cell"] = (
        (df["lat"] / cell_deg).apply(math.floor).astype(str)
        + ","
        + (df["lon"] / cell_deg).apply(math.floor).astype(str)
    )
    df["_pri"] = df["type"].map(TYPE_PRIORITY).fillna(99)
    df = (df.sort_values(["_pri", "station_id"])
            .drop_duplicates(subset="_cell")
            .drop(columns=["_cell", "_pri"])
            .reset_index(drop=True))
    return df


def apply_min_dist_filter(df: pd.DataFrame, min_km: float) -> pd.DataFrame:
    """Greedy thinning: keep a station only if it is >= min_km from all kept stations."""
    df = df.copy()
    df["_pri"] = df["type"].map(TYPE_PRIORITY).fillna(99)
    df = df.sort_values(["_pri", "station_id"]).reset_index(drop=True)
    kept_idx, kept_coords = [], []
    for i, row in df.iterrows():
        if not any(_haversine_km(row["lat"], row["lon"], la, lo) < min_km
                   for la, lo in kept_coords):
            kept_idx.append(i)
            kept_coords.append((row["lat"], row["lon"]))
    return df.loc[kept_idx].drop(columns=["_pri"]).reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# 5. OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def write_config_xml(df: pd.DataFrame, path: Path, kml_rel: str,
                     grid_res: int, hours: int) -> None:
    """
    Write simulation config XML to config/<ao_name>.xml.
    <stations> and <buoys> blocks are replaced wholesale.
    <start>, <end>, <control_volume>, and <grid> are scaffolded as placeholders.

    Purely numeric station IDs (Iowa State internal identifiers) are dropped
    before writing — they pass metadata endpoint validation but return no data
    from the ASOS data endpoint, which requires ICAO codes.
    """
    import re
    from datetime import datetime, timezone, timedelta
    now   = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=hours)

    root = ET.Element("simulation")

    ET.SubElement(root, "start").text = start.isoformat()
    ET.SubElement(root, "end").text   = now.isoformat()
    ET.SubElement(root, "control_volume", kml=kml_rel.replace("\\", "/"))

    metar_df = df[df["type"] == "METAR"].sort_values("station_id")

    # Drop purely numeric IDs — Iowa State internal IDs that silently return
    # no data on the ASOS data endpoint (asos.py requires ICAO codes)
    valid_metar   = metar_df[~metar_df["station_id"].astype(str).str.fullmatch(r'\d+')]
    dropped_metar = metar_df[ metar_df["station_id"].astype(str).str.fullmatch(r'\d+')]  
        
    if not dropped_metar.empty:
        print(f"  [WARN] Dropping {len(dropped_metar)} numeric-only station ID(s) "
              f"(Iowa State internal — not valid for ASOS data endpoint): "
              f"{dropped_metar['station_id'].tolist()}")

    stations_el = ET.SubElement(root, "stations")
    for _, row in valid_metar.iterrows():
        attrs = {"icao": str(row["station_id"])}
        if str(row.get("name", "")).strip():
            attrs["description"] = str(row["name"]).strip()
        ET.SubElement(stations_el, "station", **attrs)

    buoy_df = df[df["type"] == "NDBC_BUOY"].sort_values("station_id")
    if not buoy_df.empty:
        buoys_el = ET.SubElement(root, "buoys")
        for _, row in buoy_df.iterrows():
            attrs = {"id": str(row["station_id"])}
            if str(row.get("name", "")).strip():
                attrs["description"] = str(row["name"]).strip()
            ET.SubElement(buoys_el, "buoy", **attrs)

    ET.SubElement(root, "grid", resolution=str(grid_res))

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        _indent_xml(root)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        ET.ElementTree(root).write(f, encoding="unicode", xml_declaration=False)

    print(f"  config  -> {path}")
    print(f"            ({len(valid_metar)} METAR stations, {len(buoy_df)} NDBC buoys)")

def write_station_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  csv     -> {path}")


def write_station_kml(df: pd.DataFrame, path: Path) -> None:
    """Write station placemarks as KML using ElementTree for proper XML escaping."""
    KML_NS = "http://www.opengis.net/kml/2.2"
    ET.register_namespace("", KML_NS)

    kml_el  = ET.Element(f"{{{KML_NS}}}kml")
    doc_el  = ET.SubElement(kml_el, f"{{{KML_NS}}}Document")

    # Styles
    for style_id, icon in (
        ("metar", "ylw-circle"),
        ("ndbc",  "blu-circle"),
    ):
        sty = ET.SubElement(doc_el, f"{{{KML_NS}}}Style", id=style_id)
        ico = ET.SubElement(sty, f"{{{KML_NS}}}IconStyle")
        ic  = ET.SubElement(ico, f"{{{KML_NS}}}Icon")
        ET.SubElement(ic, f"{{{KML_NS}}}href").text = (
            f"http://maps.google.com/mapfiles/kml/paddle/{icon}.png"
        )

    for _, r in df.iterrows():
        style    = "ndbc" if r["type"] == "NDBC_BUOY" else "metar"
        name     = str(r["station_id"])
        desc     = f"{r['type']} | {str(r.get('name', ''))}"

        pm = ET.SubElement(doc_el, f"{{{KML_NS}}}Placemark")
        ET.SubElement(pm, f"{{{KML_NS}}}name").text        = name
        ET.SubElement(pm, f"{{{KML_NS}}}description").text = desc
        ET.SubElement(pm, f"{{{KML_NS}}}styleUrl").text    = f"#{style}"
        pt = ET.SubElement(pm, f"{{{KML_NS}}}Point")
        ET.SubElement(pt, f"{{{KML_NS}}}coordinates").text = (
            f"{r['lon']},{r['lat']},0"
        )

    try:
        ET.indent(kml_el, space="  ")
    except AttributeError:
        _indent_xml(kml_el)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        ET.ElementTree(kml_el).write(f, encoding="unicode", xml_declaration=False)

    print(f"  kml     -> {path}")


def _indent_xml(elem, level=0):
    """Fallback indenter for Python < 3.9."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find weather stations within a KML polygon and write simulation config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
outputs (all derived from KML stem, relative to project root):
  config/<ao_name>.xml          simulation config with stations/buoys populated
  data/stations/<ao_name>.csv   full station table
  data/stations/<ao_name>.kml   point placemarks for GIS review

examples:
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --grid 0.5
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --min-dist 40
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --grid 0.5 --min-dist 40
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --metar-csv airports.csv
  python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --root /path/to/project
        """,
    )
    parser.add_argument("--kml",       required=True,
                        help="Input KML  (e.g. config/maps/Mid-Atlantic-Littoral.kml)")
    parser.add_argument("--root",      default=None,
                        help="Project root directory. Defaults to CWD.")
    parser.add_argument("--metar-csv", default="airports.csv",
                        help="Local METAR CSV — skips live fetch. "
                             "Accepts OurAirports airports.csv or any CSV "
                             "with station_id/lat/lon columns.")
    parser.add_argument("--grid",      type=float, default=None, metavar="DEGREES",
                        help="Keep best station per DEGREES x DEGREES grid cell.")
    parser.add_argument("--min-dist",  type=float, default=None, metavar="KM",
                        help="Drop stations within KM km of a higher-priority station.")
    parser.add_argument("--grid-res",  type=int,   default=20,
                        help="Interpolation resolution written to config XML "
                             "(points per degree). Default: 20")
    parser.add_argument("--airport-types", default="large_airport,medium_airport",
                        help="Comma-separated OurAirports type values to keep when using "
                             "--metar-csv airports.csv. Default: large_airport,medium_airport. "
                             "Use 'all' to skip filtering.")
    parser.add_argument("--hours",     type=int,   default=24,
                        help="Lookback window in hours written to config XML "
                             "(end=now, start=now-HOURS). Default: 24")
    args = parser.parse_args()

    # ── Resolve project root and paths ────────────────────────────────────────
    if args.root:
        root = Path(args.root).resolve()
    else:
        # Walk up from script location until we find a dir containing both
        # 'config' and 'data' subdirectories — that's the project root.
        # Falls back to CWD if not found.
        root = None
        for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
            if (candidate / "config").is_dir() and (candidate / "data").is_dir():
                root = candidate
                break
        if root is None:
            root = Path.cwd()
        print(f"      Project root: {root}")

    # KML may be given as a bare filename (running from config/maps/) or a path
    kml_arg = Path(args.kml)
    if kml_arg.is_absolute():
        kml_path = kml_arg.resolve()
    elif kml_arg.exists():
        kml_path = kml_arg.resolve()          # relative to CWD
    else:
        kml_path = (root / args.kml).resolve()  # relative to project root

    if not kml_path.exists():
        sys.exit(f"ERROR: KML not found: {kml_path}")

    ao_name    = kml_path.stem                              # e.g. Mid-Atlantic-Littoral
    config_xml = root / "config" / f"{ao_name}.xml"
    data_csv   = root / "data" / "stations" / f"{ao_name}.csv"
    data_kml   = root / "data" / "stations" / f"{ao_name}.kml"

    # KML path stored in config relative to project root
    try:
        kml_rel = str(kml_path.relative_to(root))
    except ValueError:
        kml_rel = str(kml_path)

    # ── Load polygon ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading polygon: {kml_rel}")
    poly = load_polygon_from_kml(kml_path)
    W, S, E, N = poly.bounds
    print(f"      Bounds:  W={W:.3f}  S={S:.3f}  E={E:.3f}  N={N:.3f}")

    # ── Fetch stations ────────────────────────────────────────────────────────
    print("\n[2/4] Fetching station data ...")
    frames = []
    try:
        frames.append(fetch_ndbc_stations())
    except Exception as e:
        print(f"  [NDBC]  WARN: {e}")
    airport_types = (
        None if args.airport_types.lower() == "all"
        else set(t.strip() for t in args.airport_types.split(","))
    )
    try:
        frames.append(fetch_metar_stations(metar_csv=args.metar_csv,
                                           airport_types=airport_types))
    except Exception as e:
        print(f"  [METAR] WARN: {e}")

    if not frames:
        sys.exit("ERROR: No station data could be fetched.")

    # Normalize every frame to exactly these four columns before concat
    COLS = ["station_id", "lat", "lon", "name", "type"]
    clean = []
    for df in frames:
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "name" not in df.columns:
            df["name"] = ""
        df = df.reindex(columns=COLS).reset_index(drop=True)
        clean.append(df)

    all_stations = pd.concat(clean, ignore_index=True)
    all_stations["name"] = all_stations["name"].fillna("")
    all_stations = all_stations.drop_duplicates(subset="station_id", keep="first")
    print(f"  Total loaded: {len(all_stations)}")

    # ── Polygon filter ────────────────────────────────────────────────────────
    print("\n[3/4] Filtering ...")
    inside = filter_within_polygon(all_stations, poly)
    print(f"  Within polygon: {len(inside)}"
          f"  (METAR: {(inside['type']=='METAR').sum()}, "
          f"NDBC: {(inside['type']=='NDBC_BUOY').sum()})")

    if args.grid is not None:
        before = len(inside)
        inside = apply_grid_filter(inside, args.grid)
        print(f"  After --grid {args.grid}deg:    {len(inside)}"
              f"  (removed {before - len(inside)})")

    if args.min_dist is not None:
        before = len(inside)
        inside = apply_min_dist_filter(inside, args.min_dist)
        print(f"  After --min-dist {args.min_dist}km:  {len(inside)}"
              f"  (removed {before - len(inside)})")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n[4/4] Writing outputs ...")
    write_config_xml(inside, config_xml, kml_rel, args.grid_res, args.hours)
    write_station_csv(inside, data_csv)
    write_station_kml(inside, data_kml)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n-- Station summary " + "-" * 55)
    pd.set_option("display.max_rows",    200)
    pd.set_option("display.max_colwidth", 40)
    print(inside[["station_id", "type", "lat", "lon", "name"]]
          .sort_values(["type", "station_id"])
          .to_string(index=False))
    print()


if __name__ == "__main__":
    main()