"""
weather/fetch_buoys.py

Pulls historical NDBC buoy observations and merges with nearest METAR data
for ceiling/visibility/sky condition fields.

Data source
-----------
Recent (< 45 days):  NDBC 45-day realtime stdmet file
Historical:          NDBC annual stdmet archive (.txt.gz)

NDBC stdmet columns
-------------------
YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE

Units (as-reported): WSPD/GST m/s, WVHT m, PRES hPa, ATMP/WTMP/DEWP degC,
                     VIS nautical miles, WDIR/MWD degrees

Missing value sentinels: 99(WDIR), 99.0(WSPD/GST/WVHT), 999(DPD/APD/MWD),
                         9999.0(PRES), 999.0(ATMP/WTMP/DEWP), 99.0(VIS)
"""

import gzip
import io
import math
from datetime import datetime, timedelta, timezone

import requests

# ── endpoints ─────────────────────────────────────────────────────────────────

NDBC_REALTIME   = "https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
NDBC_HISTORICAL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"
NDBC_STATIONS   = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"

_STATION_CACHE: dict = {}   # id -> {lat, lon, name}


# ── station metadata ──────────────────────────────────────────────────────────

def _fetch_station_meta(station_id: str) -> dict:
    """Return {lat, lon, name} for an NDBC station ID."""
    global _STATION_CACHE
    sid = station_id.lower()

    if sid in _STATION_CACHE:
        return _STATION_CACHE[sid]

    # Fetch the NDBC station table (pipe-delimited)
    resp = requests.get(NDBC_STATIONS, timeout=20)
    resp.raise_for_status()

    for line in resp.text.splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        row_id = parts[0].strip().lower()
        loc    = parts[6].strip()   # e.g. "36.611 N 74.836 W"
        name   = parts[4].strip()
        try:
            tokens = loc.split()
            lat = float(tokens[0]) * (1 if tokens[1] == "N" else -1)
            lon = float(tokens[2]) * (-1 if tokens[3] == "W" else 1)
            _STATION_CACHE[row_id] = {"lat": lat, "lon": lon, "name": name}
        except (IndexError, ValueError):
            continue

    if sid not in _STATION_CACHE:
        raise RuntimeError(
            f"Station {station_id} not found in NDBC station table."
        )
    return _STATION_CACHE[sid]


# ── NDBC data fetch ───────────────────────────────────────────────────────────

NDBC_MONTHLY = "https://www.ndbc.noaa.gov/data/stdmet/{mon}/{station}_{year}.txt.gz"

def _fetch_raw(station_id: str, start: datetime, end: datetime) -> str:
    """
    Fetch raw stdmet text for a station over a time window.
    Fetch strategy (in order):
      1. Realtime 45-day file  (most recent data)
      2. Monthly stdmet archive (current/recent months not yet in annual)
      3. Annual stdmet archive  (prior years)
    Collects all months spanned by [start, end] and concatenates.
    """
    sid  = station_id.lower()
    now  = datetime.now(tz=timezone.utc)

    # Build set of (year, month) tuples spanned by the window
    months = set()
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        months.add((cur.year, cur.month))
        # advance one month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    all_lines = []
    header    = None

    for yr, mo in sorted(months):
        text = None

        # Try 1: realtime (only useful if this month is current and recent)
        if yr == now.year and mo == now.month:
            try:
                url  = NDBC_REALTIME.format(station=sid)
                resp = requests.get(url, timeout=30)
                if resp.ok:
                    text = resp.text
            except Exception:
                pass

        # Try 2: monthly archive
        if text is None:
            mon_str = datetime(yr, mo, 1).strftime("%b")   # "Jan", "Feb", ...
            url  = NDBC_MONTHLY.format(mon=mon_str, station=sid, year=yr)
            resp = requests.get(url, timeout=60)
            if resp.ok:
                try:
                    text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
                except Exception:
                    text = resp.text

        # Try 3: annual archive
        if text is None:
            url  = NDBC_HISTORICAL.format(station=sid, year=yr)
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                raise RuntimeError(
                    f"No NDBC data found for {station_id} {yr}-{mo:02d}. "
                    f"Check station ID or adjust the time window."
                )
            resp.raise_for_status()
            try:
                text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
            except Exception:
                text = resp.text

        file_lines = text.splitlines()
        if header is None:
            # Keep header rows (lines starting with #)
            for line in file_lines:
                if line.startswith("#"):
                    all_lines.append(line)
                else:
                    break
            header = True
        # Append data rows only
        all_lines.extend(
            l for l in file_lines if l.strip() and not l.startswith("#")
        )

    if not all_lines:
        raise RuntimeError(f"No data retrieved for buoy {station_id}.")

    return "\n".join(all_lines)


# ── parser ────────────────────────────────────────────────────────────────────

def _missing(val: float, sentinel: float, tol: float = 1.0) -> bool:
    return abs(val - sentinel) < tol


def _parse_stdmet(raw: str, lat: float, lon: float,
                  start: datetime, end: datetime) -> list:
    """Parse NDBC stdmet text into obs dicts filtered to [start, end]."""
    obs_list = []
    lines    = [l for l in raw.splitlines()
                if l.strip() and not l.startswith("#")]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            # Handle both 4-digit and 2-digit year
            yr = int(parts[0])
            if yr < 100:
                yr += 2000
            t = datetime(yr, int(parts[1]), int(parts[2]),
                         int(parts[3]), int(parts[4]),
                         tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue

        if not (start <= t <= end):
            continue

        def _g(idx, sentinel, tol=1.0):
            try:
                v = float(parts[idx])
                return None if _missing(v, sentinel, tol) else v
            except (IndexError, ValueError):
                return None

        wdir  = _g(5,  99,   0.5)
        wspd  = _g(6,  99.0, 0.5)
        gust  = _g(7,  99.0, 0.5)
        wvht  = _g(8,  99.0, 0.5)
        dpd   = _g(9,  99,   0.5)
        pres  = _g(12, 9999.0, 1.0)
        atmp  = _g(13, 999.0, 0.5)
        wtmp  = _g(14, 999.0, 0.5)
        dewp  = _g(15, 999.0, 0.5)
        vis   = _g(16, 99.0, 0.5)

        if wdir is None or wspd is None:
            continue

        obs_list.append({
            "time":        t,
            "lat":         lat,
            "lon":         lon,
            "wdir":        wdir,
            "wspd":        wspd,                          # already m/s
            "gust":        gust,
            "temp":        atmp,
            "dwpt":        dewp,
            "alti":        pres * 0.02953 if pres else None,   # hPa → inHg
            "vsby":        vis  * 1.15078 if vis  else None,   # nm  → sm
            "wave_ht":     wvht,                          # m
            "wave_period": dpd,                           # s (dominant period)
            "sst":         wtmp,                          # °C
            # Inherited from nearest METAR — filled later
            "skyc1": None, "skyl1": None,
            "skyc2": None, "skyl2": None,
            "skyc3": None, "skyl3": None,
            "source": "buoy",
        })

    return sorted(obs_list, key=lambda x: x["time"])


# ── METAR inheritance ─────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R  = 6371.0
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a  = (math.sin(d1 / 2) ** 2
          + math.cos(math.radians(lat1))
          * math.cos(math.radians(lat2))
          * math.sin(d2 / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _inherit_metar_fields(buoy_obs: list, metar_data: dict) -> list:
    """
    For each buoy observation, find the temporally nearest obs from the
    spatially nearest METAR station and copy ceiling/visibility/sky fields.
    """
    if not metar_data or not buoy_obs:
        return buoy_obs

    buoy_lat = buoy_obs[0]["lat"]
    buoy_lon = buoy_obs[0]["lon"]

    # Find nearest METAR station by position
    nearest_stn = min(
        metar_data.keys(),
        key=lambda s: _haversine(
            buoy_lat, buoy_lon,
            metar_data[s][0]["lat"], metar_data[s][0]["lon"]
        )
    )
    metar_obs = metar_data[nearest_stn]

    for obs in buoy_obs:
        # Find nearest METAR obs in time
        best = min(metar_obs,
                   key=lambda m: abs((m["time"] - obs["time"]).total_seconds()))
        for k in ("skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3"):
            obs[k] = best.get(k)
        # Inherit visibility only if buoy didn't report it
        if obs["vsby"] is None:
            obs["vsby"] = best.get("vsby")

    return buoy_obs


# ── public interface ──────────────────────────────────────────────────────────

def fetch_buoys(
    station_ids: list[str],
    start:       datetime,
    end:         datetime,
    metar_data:  dict | None = None,
) -> dict:
    """
    Fetch NDBC buoy observations for a list of station IDs.

    Parameters
    ----------
    station_ids : list of NDBC station IDs, e.g. ['44014', '44025']
    start / end : timezone-aware datetimes (UTC)
    metar_data  : existing METAR obs dict — used to inherit ceiling/vis/sky fields

    Returns
    -------
    {station_id: [obs_dict, ...]}  same schema as fetch_metars output,
    plus wave_ht, wave_period, sst fields.
    """
    result = {}

    for sid in station_ids:
        print(f"  Fetching buoy {sid}...")
        try:
            meta = _fetch_station_meta(sid)
            raw  = _fetch_raw(sid, start, end)
            obs  = _parse_stdmet(raw, meta["lat"], meta["lon"], start, end)
            if not obs:
                print(f"    [WARNING] Buoy {sid} skipped: no valid observations in window.")
                continue
            if metar_data:
                obs = _inherit_metar_fields(obs, metar_data)
            result[f"BUOY_{sid}"] = obs
            print(f"    {len(obs)} obs loaded  "
                  f"({meta['name']}, {meta['lat']:.2f}N {abs(meta['lon']):.2f}W)")
        except Exception as e:
            print(f"    [WARNING] Buoy {sid} skipped: {e}")

    return result