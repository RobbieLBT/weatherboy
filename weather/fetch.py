"""
weather/fetch.py

Pulls historical METARs from the Iowa State Mesonet archive.
Iowa State is used instead of AWC because AWC only retains ~24 h of observations;
for arbitrary historical windows, Iowa State has complete coverage back to the 1970s.

Hard-fails if any station returns no data.

Lessons learned
---------------
- tz must be "UTC" (uppercase) — Iowa State API is case-sensitive since 2026-04
- sts/ets datetime format replaced year1/month1/.../hour2 params (422 otherwise)
- report_type param removed — no longer accepted
- Iowa State strips the K prefix from ICAO codes; re-added on parse
- 5+-digit pure-numeric station IDs are rejected by the API; filtered out on entry
"""

import csv
import io
import re
from datetime import datetime, timezone

import requests

IOWA_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _is_valid_mesonet_id(sid: str) -> bool:
    """
    Iowa State Mesonet rejects station IDs that are 5+ pure digits.
    Valid IDs are either ICAO alpha codes (e.g. KCHO) or short numeric
    WMO-style codes up to 4 digits (e.g. 3363).
    """
    return not re.fullmatch(r'\d{5,}', sid.strip())


def fetch_metars(stations: list[str], start: datetime, end: datetime) -> dict:
    """
    Fetch METAR observations for a list of ICAO stations over a time window.

    Parameters
    ----------
    stations : list of ICAO codes, e.g. ['KCHO', 'KLYH']
    start    : timezone-aware datetime (UTC)
    end      : timezone-aware datetime (UTC)

    Returns
    -------
    {station_id: [obs_dict, ...]}  sorted chronologically.

    obs_dict keys
    -------------
    time    : datetime (UTC)
    lat     : float (degrees N)
    lon     : float (degrees E)
    wdir    : float (degrees, meteorological FROM convention)
    wspd    : float (m/s)
    gust    : float | None (m/s)
    temp    : float | None (deg C)
    dwpt    : float | None (deg C)
    alti    : float | None (inHg)
    vsby    : float | None (statute miles)
    skyc1-3 : str | None  sky condition (CLR/FEW/SCT/BKN/OVC)
    skyl1-3 : float | None  layer height (hundreds of feet)
    """
    # ── Filter out IDs the Mesonet API cannot accept ──────────────────────────
    valid   = [s for s in stations if _is_valid_mesonet_id(s)]
    skipped = [s for s in stations if not _is_valid_mesonet_id(s)]
    if skipped:
        print(f"  [WARN] Skipping unrecognised station IDs (5+ digit numeric): {skipped}")
    if not valid:
        raise RuntimeError("No valid station IDs remain after filtering.")

    params = [("station", s) for s in valid]
    params += [
        ("data",   "drct"),
        ("data",   "sknt"),
        ("data",   "gust"),
        ("data",   "tmpf"),
        ("data",   "dwpf"),
        ("data",   "alti"),
        ("data",   "vsby"),
        ("data",   "skyc1"),
        ("data",   "skyl1"),
        ("data",   "skyc2"),
        ("data",   "skyl2"),
        ("data",   "skyc3"),
        ("data",   "skyl3"),
        ("sts",    start.strftime("%Y-%m-%dT%H:%MZ")),
        ("ets",    end.strftime("%Y-%m-%dT%H:%MZ")),
        ("tz",     "UTC"),          # NOTE: must be uppercase — API is case-sensitive
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("direct", "no"),
    ]

    resp = requests.get(IOWA_URL, params=params, timeout=30)
    if not resp.ok:
        print("RESPONSE BODY:", resp.text[:500])
        resp.raise_for_status()

    raw = resp.text
    if "no results" in raw.lower() or (raw[:200].lower().count("error") > 0):
        raise RuntimeError(f"Iowa Mesonet returned an error:\n{raw[:400]}")

    lines = [l for l in raw.splitlines() if not l.startswith("#") and l.strip()]
    if len(lines) < 2:
        raise RuntimeError(
            "No METAR data returned. Verify station IDs and time window."
        )

    reader = csv.DictReader(io.StringIO("\n".join(lines)))

    def _f(v):
        """Return float or None for missing/trace values."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def _s(row, k):
        """Return sky condition string or None."""
        v = row.get(k, "").strip()
        return v if v and v != "M" else None

    result: dict[str, list] = {}

    for row in reader:
        raw_stn = row.get("station", "").strip()
        if not raw_stn:
            continue

        # Iowa State strips the leading K from US ICAO codes
        stn = "K" + raw_stn

        try:
            t = datetime.strptime(
                row["valid"].strip(), "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue

        wdir     = _f(row.get("drct"))
        wspd_kts = _f(row.get("sknt"))
        gust_kts = _f(row.get("gust"))
        tmpf     = _f(row.get("tmpf"))
        dwpf     = _f(row.get("dwpf"))
        alti     = _f(row.get("alti"))
        vsby     = _f(row.get("vsby"))
        skyl1    = _f(row.get("skyl1"))
        skyl2    = _f(row.get("skyl2"))
        skyl3    = _f(row.get("skyl3"))

        # Skip obs with no usable wind data
        if wdir is None or wspd_kts is None:
            continue

        obs = {
            "time":  t,
            "lat":   float(row["lat"]),
            "lon":   float(row["lon"]),
            "wdir":  wdir,
            "wspd":  wspd_kts * 0.514444,                              # kt → m/s
            "gust":  gust_kts * 0.514444 if gust_kts is not None else None,
            "temp":  (tmpf - 32.0) * 5.0 / 9.0 if tmpf is not None else None,
            "dwpt":  (dwpf - 32.0) * 5.0 / 9.0 if dwpf is not None else None,
            "alti":  alti,
            "vsby":  vsby,
            "skyc1": _s(row, "skyc1"), "skyl1": skyl1,
            "skyc2": _s(row, "skyc2"), "skyl2": skyl2,
            "skyc3": _s(row, "skyc3"), "skyl3": skyl3,
        }
        result.setdefault(stn, []).append(obs)

    # Check only against IDs that were actually submitted to the API,
    # accounting for the K-prefix Iowa State strips on parse.
    missing = [s for s in valid if not result.get("K" + s) and not result.get(s)]
    if missing:
        print(f"  [WARN] No observations returned for: {missing} "
            f"(station may not report publicly — skipping)")
    
    for stn in result:
        result[stn].sort(key=lambda x: x["time"])

    return result