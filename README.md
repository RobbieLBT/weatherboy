# Weatherboy

"wouldn't you like to know?"

Spatiotemporal weather environment simulator for UAV mission analysis.

Pulls real METAR observations, interpolates them onto a spatial grid, and animates the result over a georeferenced basemap. Also walks a defined flight path through the environment, logging the environmental forcing at each sample point. Built as the weather layer of a larger autonomous vehicle mission analysis framework.

---

## What it does

- Fetches historical METAR data (wind, temperature, pressure, visibility, ceiling, dewpoint) from the Iowa State Mesonet archive for any set of ICAO stations over any time window
- Fetches NDBC buoy observations and merges them with the METAR network for coastal and offshore AOs
- Interpolates observations onto a spatial grid using Barnes or Cressman objective analysis
- Models sub-hourly wind fluctuations using an Ornstein-Uhlenbeck process anchored to observed gust spread
- Animates a 2D wind field over a georeferenced basemap with a timeline slider and play/pause control
- Overlays Voronoi cells showing station ownership boundaries, shaded by any met parameter
- Renders a smooth blended interpolation field as an alternative to or alongside Voronoi shading
- Clips the analysis domain to a KML-defined control volume (or convex hull of stations)
- Derives flight category (VFR / MVFR / IFR / LIFR) from ceiling and visibility
- **Traverses a KML flight path through the environment**, sampling the interpolated field at each point and logging a complete forcing record to CSV
- **Visualizes traversal output** as a static figure: path colored by headwind, summary stats, and time series subplots

**Interactive shading parameters** (selectable at runtime via radio panel):
Wind Speed · Gust Delta · Temperature · Pressure · Dewpoint Depression · Visibility · Ceiling · Flight Category

---

## Where this is going

Weatherboy is the environmental layer of a larger UAV mission analysis framework. It is kept as a standalone repo with a clean interface boundary — downstream tools consume the forcing CSV or call the `mission` package directly.

**Current state — Phase 2 complete**
The interpolated forcing vector field is queryable at any (lat, lon, t) point. The `mission` package walks a flight path through this field and exports a structured forcing record per sample.

**Phase 3 — Vehicle dynamics integration**
Feed the forcing CSV into a vehicle physics model (VSPAero for aero derivatives, PX4 SITL for control response). The `ForcingRecord` schema is the contract: `wind_ned`, `gust_ned`, `delta_beta_deg`, `density_kgm3`, and stubs for `delta_alpha_deg` pending a vertical wind model. Key terms to resolve before integration: Cₗα, Cmα, Cnβ from VSPAero; dynamic pressure q = ½ρv²; angle-of-attack perturbation from vertical wind component (no model yet).

**Phase 4 — Closed-loop path adaptation**
Path traversal is currently open-loop (predetermined route, no vehicle response). The next step is an adaptive loop where vehicle deviations trigger a re-query of the forcing field at the actual position rather than the planned position.

**Phase 5 — Mission analysis products**
Bulk statistical analysis of a given AO: wind roses, gust exceedance curves, seasonal severity distributions, storm frequency. Lifecycle loads prediction from operational area statistics.

**Phase 6 — Urban / complex terrain**
Extend the spatial model to account for wind shear and channeling in obstructed environments for urban mission planning. Requires DEM integration.

---

## Setup

### Prerequisites (Linux / WSL2)

```bash
sudo apt install libgeos-dev libproj-dev proj-data proj-bin python3-tk python3.12-venv
```

### Environment

```bash
python3 -m venv ~/envs/weatherboy
source ~/envs/weatherboy/bin/activate
pip install -r requirements.txt
```

### VS Code (WSL2)

Set the Python interpreter to `~/envs/weatherboy/bin/python3` via the command palette (`Ctrl+Shift+P` → *Python: Select Interpreter*).

---

## Usage

### Weather animation

```bash
python3 run.py [options]
```

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--config` | path | `config/sim_config.xml` | Simulation config XML |
| `--speed` | float | `1.0` | Playback: sim-hours per real second |
| `--interp` | `barnes` / `cressman` | `barnes` | Spatial interpolation method |
| `--temporal` | `linear` / `step` | `linear` | Temporal interpolation between obs |
| `--gust` | `envelope` / `reported` | `envelope` | Gust: OU envelope or raw METAR |
| `--map` | `topo` / `satellite` / `street` / `vfr` / `none` | `topo` | Basemap style |
| `--voronoi` | flag | off | Voronoi cell overlay |
| `--blend` | flag | off | Smooth interpolated field |

```bash
# Default run
python3 run.py

# Best combined view
python3 run.py --blend --voronoi --map satellite --config config/virginia.xml

# Clean analysis view, no tiles
python3 run.py --blend --voronoi --map none --config config/virginia.xml

# Fast playback, raw METAR gusts
python3 run.py --speed 6.0 --gust reported --map topo
```

### Mission traversal

Traverse a KML flight path through the weather environment and log the forcing record at each sample point.

```bash
python3 run.py --config config/virginia.xml \
               --path config/paths/VA-XC1.kml \
               --alt-agl 350 \
               --speed-kmh 70 \
               --traverse-dt 10 \
               --output output/VA-XC1.csv \
               --no-animate
```

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | — | KML LineString path (Google Earth or QGC export) |
| `--alt-agl` | `350` | Flight altitude AGL, feet |
| `--speed-kmh` | `70` | Cruise ground speed, km/h |
| `--traverse-dt` | `10` | Sample interval, seconds (~190m spacing at 70 km/h) |
| `--output` | `output/traversal.csv` | CSV output path |
| `--log-profile` | off | Apply log wind profile to scale obs from 10m to flight altitude |
| `--no-animate` | off | Skip animation after traversal |

The output CSV contains one row per sample: position, heading, mean wind (NED), instantaneous gust (OU), headwind/crosswind, delta-beta proxy, temperature, density, visibility, ceiling, and flight category. See `mission/forcing.py` for the full schema.

**Temporal resolution note:** the OU gust correlation time is ~50s (θ = 0.02 s⁻¹). The default dt = 10s gives ~5 samples per correlation length, which adequately resolves gust texture. The METAR spatial grid node spacing (~5.5 km at 20 pts/degree) is the binding constraint — there is no physical value in dt below ~5s for this network density.

### Traversal visualization

```bash
python -m mission.visualize output/VA-XC1.csv --map topo
```

Produces a static figure: path on a Cartopy basemap colored by headwind intensity, summary stats table, and two time series subplots (wind speed + gust envelope; headwind + crosswind). Saves a PNG alongside the CSV.

| Flag | Options | Default |
|------|---------|---------|
| `--map` | `topo` / `satellite` / `street` / `none` | `topo` |

---

## Station config generation

Config XMLs are generated by `find_stations.py`, which finds all METAR and NDBC buoy stations within a KML polygon and writes a simulation config.

```bash
python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml
python find_stations.py --kml config/maps/Mid-Atlantic-Littoral.kml --grid 0.5 --min-dist 20
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kml` | required | Input KML polygon |
| `--metar-csv` | `airports.csv` | Local OurAirports CSV — skips live fetch (recommended) |
| `--grid` | off | Keep best station per N×N degree cell |
| `--min-dist` | off | Drop stations within N km of a higher-priority station |
| `--hours` | `24` | Lookback window written to config (end=now, start=now-N) |
| `--grid-res` | `20` | Interpolation grid resolution (points per degree) |
| `--airport-types` | `large_airport,medium_airport` | OurAirports type filter |

**METAR source fallback chain:** Iowa State `networks.php` → per-state Iowa State networks → aviationweather.gov API → ADDS legacy XML → local `airports.csv`. All live sources are currently blocked in many environments; using `--metar-csv airports.csv` is the reliable path.

Download `airports.csv` from [ourairports.com/data/airports.csv](https://ourairports.com/data/airports.csv) (~7 MB) and keep it in the project root.

**Known issue — Iowa State internal IDs:** when live Iowa State sources are used, some stations (particularly military fields) are assigned internal numeric IDs (e.g. `3363`, `19843`) rather than ICAO codes. These pass metadata validation but return no data from the ASOS data endpoint. `find_stations.py` now drops them automatically with a warning. The `airports.csv` path avoids this entirely.

Outputs (all derived from KML stem):
- `config/<ao_name>.xml` — simulation config
- `data/stations/<ao_name>.csv` — full station table
- `data/stations/<ao_name>.kml` — point placemarks for GIS review

---

## Forcing record schema

The `ForcingRecord` dataclass (`mission/forcing.py`) defines the environmental state contract between Weatherboy and downstream vehicle models. Key fields:

| Field | Unit | Notes |
|-------|------|-------|
| `wind_n_ms`, `wind_e_ms` | m/s | Mean wind, NED frame, from spatial interpolation |
| `gust_n_ms`, `gust_e_ms` | m/s | Instantaneous wind, OU process applied to mean |
| `headwind_ms` | m/s | Positive = opposing vehicle motion |
| `crosswind_ms` | m/s | Positive = from starboard |
| `delta_beta_deg` | deg | Sideslip perturbation proxy: atan2(crosswind, cruise_speed) |
| `delta_alpha_deg` | deg | AoA perturbation — stub, None until vertical wind model added |
| `density_kgm3` | kg/m³ | Derived: P / (Rd × T) |
| `flight_category` | — | VFR / MVFR / IFR / LIFR |

Coordinate conventions: WGS84, UTC, altitude in meters AGL, wind direction in meteorological FROM convention (degrees true).

---

## Config format

```xml
<simulation>
  <start>2026-04-21T05:00:00Z</start>
  <end>2026-04-22T05:00:00Z</end>

  <!-- Option A: KML polygon file -->
  <control_volume kml="config/maps/Virginia.kml"/>

  <!-- Option B: explicit vertices -->
  <control_volume>
    <vertex lon="-76.30" lat="36.85"/>
    <vertex lon="-79.97" lat="37.32"/>
  </control_volume>

  <!-- Option C: omit — auto convex hull of stations -->

  <stations>
    <station icao="KCHO"/>
    <station icao="KLYH"/>
    <station icao="KROA"/>
  </stations>

  <!-- Optional: NDBC buoy stations -->
  <buoys>
    <buoy id="44014"/>
  </buoys>
</simulation>
```

Station IDs must be ICAO codes (e.g. `KACY`, `KBWI`). Purely numeric IDs from Iowa State's internal station list are rejected at config-write time and at fetch time. The control volume clips Voronoi cells and sets the map extent. Stations may extend beyond the CV boundary to improve interpolation accuracy at edges.

---

## Known issues and workarounds

**Iowa State METAR sources blocked:** Iowa State's `networks.php` endpoint returns HTML (bot detection) in many environments. Use `--metar-csv airports.csv` as the reliable fallback. Set `--metar-csv airports.csv` as the default in `find_stations.py` if this is consistently your environment.

**Stations with no Iowa State coverage:** some military and municipal fields (e.g. `KFTJ`, `KOJG`) are not published to Iowa State's ASOS archive. `fetch.py` now warns and skips these rather than hard-failing. Remove them from the config if they consistently produce no data.

**VFR tile 404 errors:** `--map vfr` tile fetches may return 404 for some zoom levels or tile providers. These are non-fatal — the animation continues with missing tiles. Use `--map topo` or `--map satellite` as alternatives.

---

## Data sources

**METAR observations:** [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu) ASOS archive — full historical coverage back to the 1970s, no API key required. The Aviation Weather Center (AWC) is not used because it only retains ~24 hours of observations.

**Buoy observations:** [NOAA NDBC](https://www.ndbc.noaa.gov) stdmet archive. Recent data (< 45 days) pulled from the 45-day realtime file; historical data from monthly and annual archives.

**Station list (offline):** [OurAirports airports.csv](https://ourairports.com/data/airports.csv) — free, ~7 MB, updated regularly. Preferred over live Iowa State fetch for METAR station discovery.

**Note:** The Iowa State API strips the leading `K` from US ICAO codes in responses. This is handled automatically in `fetch.py`.

---

## Repo structure

```
weatherboy/
├── config/
│   ├── sim_config.xml          default AO (KCHO-KLYH-KROA-KSHD)
│   ├── virginia.xml            statewide Virginia config
│   ├── Mid-Atlantic-Littoral.xml  coastal Mid-Atlantic + Chesapeake config
│   ├── paths/
│   │   └── VA-XC1.kml          Hampton Roads → Staunton cross-country
│   └── maps/
│       ├── Virginia.kml        control volume polygon
│       └── Mid-Atlantic-Littoral.kml
├── data/
│   ├── stations/               station CSVs and KMLs (auto-populated)
│   └── tiles/                  basemap tile cache (auto-populated)
├── weather/
│   ├── fetch.py                Iowa State Mesonet METAR retrieval;
│   │                           filters invalid station IDs, warns on missing
│   ├── fetch_buoys.py          NDBC buoy retrieval + METAR field inheritance
│   ├── interpolate.py          Barnes / Cressman objective analysis;
│   │                           point query and temporal interpolation
│   │                           for path traversal
│   └── turbulence.py           OU sub-hourly model + log wind profile;
│                               GustSampler for online path traversal
├── mission/
│   ├── forcing.py              ForcingRecord dataclass — vehicle forcing contract
│   ├── path.py                 KML path parser, haversine geometry, path sampler
│   ├── traverse.py             PathTraversal engine — walks path, queries field,
│   │                           builds ForcingRecord list
│   ├── logger.py               CSV writer + summary printer
│   └── visualize.py            Static Matplotlib figure: map + time series
├── viz/
│   └── animate.py              Matplotlib + Cartopy animation
├── find_stations.py            Station discovery tool; writes config XML,
│                               station CSV, and station KML from a KML polygon
├── airports.csv                OurAirports station list (local fetch fallback)
├── run.py                      CLI entry point
└── requirements.txt
```
