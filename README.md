# Weatherboy

Spatiotemporal weather environment simulator for UAV mission analysis.

Pulls real METAR observations, interpolates them onto a spatial grid, and animates the result over a georeferenced basemap. Built as a forcing-vector layer for eventual integration with PX4 SITL and a broader conceptual design tool.

---

## What it does

- Fetches historical METAR data (wind, temperature, pressure, visibility, ceiling, dewpoint) from the Iowa State Mesonet archive for any set of ICAO stations over any time window
- Interpolates observations onto a spatial grid using Barnes or Cressman objective analysis
- Models sub-hourly wind fluctuations using an Ornstein-Uhlenbeck process anchored to observed gust spread
- Animates a 2D wind field over a georeferenced basemap with a timeline slider and play/pause control
- Overlays Voronoi cells showing station ownership boundaries, shaded by any met parameter
- Renders a smooth blended interpolation field as an alternative to or alongside Voronoi shading
- Clips the analysis domain to a KML-defined control volume (or convex hull of stations)
- Derives flight category (VFR / MVFR / IFR / LIFR) from ceiling and visibility

**Interactive shading parameters** (selectable at runtime via radio panel):
Wind Speed · Gust Delta · Temperature · Pressure · Dewpoint Depression · Visibility · Ceiling · Flight Category

---

## Where this is going

Weatherboy is the environmental layer of a larger UAV mission analysis framework. The near-term roadmap:

**Phase 2 — Spatial field output**
Export the interpolated forcing vector field as a structured array that an arbitrary vehicle position can be dropped into and acted upon. Clean interface for external dynamics models.

**Phase 3 — PX4 SITL integration**
Pipe the forcing field into PX4 Software-In-The-Loop simulation as a real-world wind disturbance input. Enables hardware-representative flight testing against actual historical weather.

**Phase 4 — Mission analysis**
- Virtual flight path analysis: subject a defined route to the weather environment and characterize conditions along track
- Bulk statistical analysis of a given AO: wind roses, gust exceedance curves, seasonal severity distributions, storm frequency
- Sensor performance estimation under IMC conditions
- Lifecycle loads prediction based on operational area statistics

**Phase 5 — Urban / complex terrain**
Extend the spatial model to account for wind shear and channeling in obstructed environments for urban mission planning.

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

```bash
python3 run.py [options]
```

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--config` | path | `config/sim_config.xml` | Simulation config XML |
| `--speed` | float | `1.0` | Playback: sim-hours per real second |
| `--interp` | `barnes` / `cressman` | `barnes` | Spatial interpolation method |
| `--temporal` | `linear` / `step` | `linear` | Temporal interpolation between obs |
| `--gust` | `envelope` / `reported` | `envelope` | Gust: OU p95 envelope or raw METAR |
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
</simulation>
```

The control volume clips Voronoi cells and sets the map extent. Stations can extend beyond the CV boundary — this is intentional, it improves interpolation accuracy at the edges.

---

## Repo structure

```
weatherboy/
├── config/
│   ├── sim_config.xml        default AO (KCHO-KLYH-KROA-KSHD)
│   ├── virginia.xml          statewide Virginia config
│   └── maps/
│       └── Virginia.kml      control volume polygon
├── data/                     tile cache (auto-populated)
├── weather/
│   ├── fetch.py              Iowa State Mesonet METAR retrieval
│   ├── interpolate.py        Barnes / Cressman objective analysis
│   └── turbulence.py         OU sub-hourly model + log wind profile
├── viz/
│   └── animate.py            Matplotlib + Cartopy animation
├── run.py                    CLI entry point
└── requirements.txt
```

---

## Data source

All observations pulled from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu) ASOS archive — full historical coverage back to the 1970s, no API key required. The Aviation Weather Center (AWC) is not used because it only retains ~24 hours of observations.

**Note:** The Iowa State API strips the leading `K` from US ICAO codes in responses. This is handled automatically in `fetch.py`.