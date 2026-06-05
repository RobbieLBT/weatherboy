"""
viz/animate.py

Animated surface wind field with timeline slider, multiple basemap options,
and optional Voronoi spatial overlay.

Flags
-----
--map       topo | satellite | street | vfr      (default: topo)
--voronoi   enable Voronoi cell overlay
--save-gif  PATH  save animation to GIF instead of displaying interactively
"""

from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")

from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, to_rgba, ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPoly, MultiPolygon
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

PROJ              = ccrs.PlateCarree()
SUBSTEPS_PER_HOUR = 4
TILE_ZOOM         = 9


# ── shade registry ────────────────────────────────────────────────────────────

_FC_CMAP = ListedColormap(["magenta", "red", "#4488ff", "green"])
_FC_NORM = BoundaryNorm([0, 1, 2, 3, 4], 4)

SHADE_OPTS = {
    "wspd":      {"label": "Wind Speed",   "vmin": 0,     "vmax": 15,   "cmap": "plasma",   "unit": "m/s"},
    "gust":      {"label": "Gust Delta",   "vmin": 0,     "vmax": 8,    "cmap": "hot_r",    "unit": "m/s"},
    "temp":      {"label": "Temperature",  "vmin": -5,    "vmax": 35,   "cmap": "RdYlBu_r", "unit": "C"},
    "pressure":  {"label": "Pressure",     "vmin": 29.5,  "vmax": 30.8, "cmap": "viridis",  "unit": "inHg"},
    "dewdep":    {"label": "Dewpt Dep.",   "vmin": 0,     "vmax": 20,   "cmap": "YlOrBr",   "unit": "C"},
    "visibility":{"label": "Visibility",   "vmin": 0,     "vmax": 10,   "cmap": "Blues_r",  "unit": "sm"},
    "ceiling":   {"label": "Ceiling",      "vmin": 0,     "vmax": 5000, "cmap": "cool",     "unit": "ft"},
    "flightcat": {"label": "Flight Cat.",  "vmin": 0,     "vmax": 4,    "cmap": _FC_CMAP,   "unit": ""},
}
_LABEL_TO_KEY = {v["label"]: k for k, v in SHADE_OPTS.items()}


def _get_ceiling(obs: dict) -> float:
    for i in range(1, 4):
        skyc = obs.get(f"skyc{i}")
        skyl = obs.get(f"skyl{i}")
        if skyc in ("BKN", "OVC") and skyl is not None:
            return float(skyl) * 100.0
    return 9999.0


def _flight_cat(obs: dict) -> float:
    ceil = _get_ceiling(obs)
    vis  = obs.get("vsby") or 10.0
    if ceil < 500  or vis < 1.0: return 0.5   # LIFR
    if ceil < 1000 or vis < 3.0: return 1.5   # IFR
    if ceil < 3000 or vis < 5.0: return 2.5   # MVFR
    return 3.5                                  # VFR


def _shade_value(obs: dict, key: str) -> float:
    if key == "wspd":
        return obs.get("wspd") or 0.0
    if key == "gust":
        g = obs.get("gust") or obs.get("wspd") or 0.0
        return max(0.0, g - (obs.get("wspd") or 0.0))
    if key == "temp":
        return obs.get("temp") if obs.get("temp") is not None else 15.0
    if key == "pressure":
        return obs.get("alti") or 29.92
    if key == "dewdep":
        t = obs.get("temp") if obs.get("temp") is not None else 15.0
        d = obs.get("dwpt") if obs.get("dwpt") is not None else t
        return max(0.0, t - d)
    if key == "visibility":
        return min(obs.get("vsby") or 10.0, 10.0)
    if key == "ceiling":
        return min(_get_ceiling(obs), 5000.0)
    if key == "flightcat":
        return _flight_cat(obs)
    return 0.0


# ── tile sources ──────────────────────────────────────────────────────────────

class _OpenTopoMap(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://tile.opentopomap.org/{z}/{x}/{y}.png"

class _Satellite(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

class _Street(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"

class _VFR(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return (
            f"https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/services"
            f"/VFR_Sectional/MapServer/tile/{z}/{y}/{x}"
        )

_TILE_SOURCES = {
    "topo":      _OpenTopoMap,
    "satellite": _Satellite,
    "street":    _Street,
    "vfr":       _VFR,
}
_BG_COLORS   = {"topo": "#0d1117", "satellite": "#0a0a0a", "street": "#f0f0f0", "vfr": "#d0e8f0"}
_TEXT_COLORS = {"topo": "white",   "satellite": "white",   "street": "#111111", "vfr": "#111111"}


# ── Voronoi helpers ───────────────────────────────────────────────────────────

def _clip_to_cv(verts: np.ndarray, cv_polygon: np.ndarray) -> np.ndarray | None:
    """Clip a Voronoi cell polygon to the control volume using shapely."""
    try:
        cell  = ShapelyPoly(verts)
        ao    = ShapelyPoly(cv_polygon)
        inter = cell.intersection(ao)
        if inter.is_empty or inter.area < 1e-10:
            return None
        if isinstance(inter, MultiPolygon):
            inter = max(inter.geoms, key=lambda g: g.area)
        coords = np.array(inter.exterior.coords)
        return coords if len(coords) >= 3 else None
    except Exception:
        return None

def _build_voronoi_collection(coords: dict, cv_polygon: np.ndarray, ax, cmap, norm):
    from matplotlib.collections import PolyCollection

    lon0, lon1 = cv_polygon[:,0].min(), cv_polygon[:,0].max()
    lat0, lat1 = cv_polygon[:,1].min(), cv_polygon[:,1].max()
    cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    R = max(lon1 - lon0, lat1 - lat0) * 5

    station_names = list(coords.keys())
    pts = np.array([[coords[s][1], coords[s][0]] for s in station_names])
    mirrors = np.array([
        [cx - R, cy], [cx + R, cy], [cx, cy - R], [cx, cy + R],
        [cx - R, cy - R], [cx + R, cy - R], [cx - R, cy + R], [cx + R, cy + R],
    ])
    vor = Voronoi(np.vstack([pts, mirrors]))

    verts_list, station_order = [], []
    for i, stn in enumerate(station_names):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) == 0:
            continue
        clipped = _clip_to_cv(vor.vertices[region], cv_polygon)
        if clipped is None:
            continue
        verts_list.append(clipped)
        station_order.append(stn)

    if not verts_list:
        return [], None

    init_colors = [(*to_rgba(cmap(norm(0.0)))[:3], 0.32)] * len(verts_list)
    col = PolyCollection(
        verts_list,
        facecolors=init_colors,
        edgecolors="white",
        linewidths=0.8,
        transform=PROJ,
        zorder=3,
    )
    ax.add_collection(col)
    return station_order, col


# ── temporal helpers ──────────────────────────────────────────────────────────

def _lerp_angle(a1: float, a2: float, alpha: float) -> float:
    diff = ((a2 - a1 + 180.0) % 360.0) - 180.0
    return (a1 + alpha * diff) % 360.0


def _obs_at(obs_list: list, t, mode: str) -> dict | None:
    if not obs_list:
        return None
    before = [o for o in obs_list if o["time"] <= t]
    after  = [o for o in obs_list if o["time"] >  t]

    if mode == "step" or not after:
        return before[-1] if before else obs_list[0]
    if not before:
        return obs_list[0]

    o1, o2  = before[-1], after[0]
    dt_total = (o2["time"] - o1["time"]).total_seconds()
    if dt_total == 0:
        return o1
    alpha = (t - o1["time"]).total_seconds() / dt_total

    def _lv(k, default=0.0):
        v1 = o1.get(k) if o1.get(k) is not None else default
        v2 = o2.get(k) if o2.get(k) is not None else default
        return v1 + alpha * (v2 - v1)

    return {
        "time": t, "lat": o1["lat"], "lon": o1["lon"],
        "wdir": _lerp_angle(o1["wdir"], o2["wdir"], alpha),
        "wspd": _lv("wspd"),
        "gust": _lv("gust") if (o1.get("gust") and o2.get("gust")) else None,
        "temp": _lv("temp", 15.0),
    }


def _to_uv(wdir: float, wspd: float) -> tuple[float, float]:
    r = np.radians(wdir)
    return -wspd * np.sin(r), -wspd * np.cos(r)


# ── main entry ────────────────────────────────────────────────────────────────

def run_animation(
    obs_data:       dict,
    start,
    end,
    playback_speed: float       = 1.0,
    spatial_method: str         = "barnes",
    temporal_mode:  str         = "linear",
    gust_mode:      str         = "envelope",
    map_style:      str         = "topo",
    voronoi:        bool        = False,
    blend:          bool        = False,
    cv_polygon:     object      = None,
    save_gif:       str | None  = None,
) -> animation.FuncAnimation:

    # ── timeline ──────────────────────────────────────────────────────────────
    total_hours = (end - start).total_seconds() / 3600.0
    n_steps     = max(int(total_hours * SUBSTEPS_PER_HOUR), 1)
    times       = [start + timedelta(hours=i * total_hours / n_steps)
                   for i in range(n_steps + 1)]
    n_frames    = len(times)

    stations = list(obs_data.keys())
    coords   = {s: (obs_data[s][0]["lat"], obs_data[s][0]["lon"]) for s in stations}

    pad = 0.25
    if cv_polygon is not None:
        import numpy as _np
        extent = [
            float(cv_polygon[:,0].min()) - pad, float(cv_polygon[:,0].max()) + pad,
            float(cv_polygon[:,1].min()) - pad, float(cv_polygon[:,1].max()) + pad,
        ]
    else:
        all_lats = [c[0] for c in coords.values()]
        all_lons = [c[1] for c in coords.values()]
        extent   = [min(all_lons) - pad, max(all_lons) + pad,
                    min(all_lats) - pad, max(all_lats) + pad]

    bg   = _BG_COLORS.get(map_style, "#0d1117")
    tc   = _TEXT_COLORS.get(map_style, "white")
    dark = tc == "white"

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10), facecolor=bg)
    fig.subplots_adjust(bottom=0.13)
    ax  = fig.add_axes([0.05, 0.15, 0.72, 0.80], projection=PROJ)
    ax.set_extent(extent, crs=PROJ)
    ax.set_facecolor(bg)

    if map_style != "none":
        tile_cls = _TILE_SOURCES.get(map_style, _OpenTopoMap)
        print(f"  Loading {map_style} tiles (first run may be slow)...")
        ax.add_image(tile_cls(), TILE_ZOOM)

    edge_c = "#cccccc" if dark else "#555555"
    ax.add_feature(cfeature.STATES.with_scale("10m"),    linewidth=0.8, edgecolor=edge_c)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor=edge_c)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"),    linewidth=0.4, edgecolor="#5599bb", alpha=0.5)

    # ── blend field (interpolated grid) ──────────────────────────────────────
    _blend_mesh = [None]
    _blend_grid_lat = _blend_grid_lon = None
    if blend:
        from weather.interpolate import make_grid
        _blend_grid_lat, _blend_grid_lon = make_grid(extent, resolution=15)

    # ── colormap (dynamic — updated when shade changes) ───────────────────────
    _shade  = ["wspd"]
    _opt    = SHADE_OPTS["wspd"]
    cmap    = plt.get_cmap(_opt["cmap"]) if isinstance(_opt["cmap"], str) else _opt["cmap"]
    norm    = Normalize(vmin=_opt["vmin"], vmax=_opt["vmax"])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.01)
    cb.set_label(f"{_opt['label']}  ({_opt['unit']})", color=tc, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=tc)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=tc)

    def _update_shade_style(key: str):
        opt  = SHADE_OPTS[key]
        _cm  = plt.get_cmap(opt["cmap"]) if isinstance(opt["cmap"], str) else opt["cmap"]
        _nm  = (_FC_NORM if key == "flightcat"
                else Normalize(vmin=opt["vmin"], vmax=opt["vmax"]))
        sm.set_cmap(_cm)
        sm.set_norm(_nm)
        cb.update_normal(sm)
        cb.set_label(f"{opt['label']}  ({opt['unit']})", color=tc, fontsize=9)
        if _blend_mesh[0] is not None:
            _blend_mesh[0].set_cmap(_cm)
            _blend_mesh[0].set_norm(_nm)
        return _cm, _nm

    # ── Voronoi patches (built before station markers so markers sit on top) ──
    vor_order, vor_col = [], None
    if voronoi and len(stations) >= 3:
        print("  Building Voronoi cells...")
        _cv = cv_polygon if cv_polygon is not None else __import__("numpy").array(
            [[extent[0], extent[2]], [extent[1], extent[2]],
             [extent[1], extent[3]], [extent[0], extent[3]]]
        )
        vor_order, vor_col = _build_voronoi_collection(coords, _cv, ax, cmap, norm)
    elif voronoi:
        print("  Warning: need >= 3 stations for Voronoi. Skipping.")

    # ── CV boundary ──────────────────────────────────────────────────────────
    if cv_polygon is not None:
        cv_closed = np.vstack([cv_polygon, cv_polygon[0]])
        ax.plot(
            cv_closed[:,0], cv_closed[:,1],
            color="#00d4ff", linewidth=1.8, linestyle="--",
            transform=PROJ, zorder=7, alpha=0.85,
            label="AO boundary",
        )

    # ── station markers ───────────────────────────────────────────────────────
    for stn, (lat, lon) in coords.items():
        ax.plot(lon, lat, "o", color=tc, ms=5, transform=PROJ, zorder=8)
        ax.text(lon + 0.06, lat + 0.05, stn,
                fontsize=8, color=tc, fontweight="bold",
                transform=PROJ, zorder=9,
                bbox=dict(facecolor=bg, alpha=0.6, boxstyle="round,pad=0.25"))

    # ── clock ─────────────────────────────────────────────────────────────────
    clock = ax.text(
        0.02, 0.97, "",
        transform=ax.transAxes,
        fontsize=11, color=tc, va="top", ha="left", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor=bg, alpha=0.85, edgecolor="#444"),
        zorder=10,
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker=">", color=tc,    label="Wind", markersize=8, linewidth=0),
        Line2D([0],[0], marker=">", color="cyan", label="Gust", markersize=8, linewidth=0, alpha=0.5),
    ]
    if voronoi:
        from matplotlib.patches import Patch
        legend_handles.append(
            Patch(facecolor=cmap(norm(7.5)), alpha=0.3,
                  edgecolor="white", label="Wind speed (cell)")
        )
    ax.legend(handles=legend_handles, loc="lower right",
              facecolor=bg, edgecolor="#555", labelcolor=tc, fontsize=8)

    ax.set_title(
        f"POSEIDON-01  |  Surface Wind Field  [{map_style.upper()}]"
        + ("  [VORONOI]" if voronoi else ""),
        color=tc, fontsize=13, pad=8, fontfamily="monospace",
    )

    # ── slider + button (skipped when saving GIF — non-interactive) ───────────
    if not save_gif:
        sl_ax  = fig.add_axes([0.10, 0.05, 0.68, 0.025],
                              facecolor="#1e2a35" if dark else "#dde")
        slider = Slider(sl_ax, "", 0, n_frames - 1, valinit=0, valstep=1,
                        color="#4a90d9")
        slider.valtext.set_visible(False)
        sl_ax.set_xlabel("Timeline", color=tc, fontsize=8, labelpad=2)

        btn_ax = fig.add_axes([0.82, 0.04, 0.07, 0.045])
        btn    = Button(btn_ax, "|| Pause",
                        color="#1e2a35" if dark else "#ccd",
                        hovercolor="#2e3a45" if dark else "#aab")
        btn.label.set_color(tc)
        btn.label.set_fontsize(9)

        # ── shade selector (RadioButtons) ─────────────────────────────────────
        radio_ax = fig.add_axes(
            [0.79, 0.22, 0.19, 0.58],
            facecolor="#111820" if dark else "#e8eaf0"
        )
        radio_ax.set_title("Shading", color=tc, fontsize=8, pad=4)
        radio = RadioButtons(
            radio_ax,
            labels=[SHADE_OPTS[k]["label"] for k in SHADE_OPTS],
            active=0,
        )
        for lbl in radio.labels:
            lbl.set_fontsize(8)
            lbl.set_color(tc)

    # ── mutable state ─────────────────────────────────────────────────────────
    _qw              = [None]
    _qg              = [None]
    _playing         = [True]
    _frame           = [0]
    _slider_dragging = [False]
    _render_cmap     = [cmap]
    _render_norm     = [norm]

    # ── shade callback (only wired when interactive) ──────────────────────────
    if not save_gif:
        def _on_shade(label):
            key = _LABEL_TO_KEY[label]
            _shade[0] = key
            new_cmap, new_norm = _update_shade_style(key)
            _render_cmap[0] = new_cmap
            _render_norm[0] = new_norm
            _render(_frame[0])
            fig.canvas.draw_idle()

        radio.on_clicked(_on_shade)

    # ── OU envelopes ──────────────────────────────────────────────────────────
    ou_p95: dict[str, list[float]] = {}
    if gust_mode == "envelope":
        from weather.turbulence import gust_envelope
        for stn in stations:
            ou_p95[stn] = [gust_envelope(o, n=60, dt=60.0)["p95"]
                           for o in obs_data[stn]]

    def _gust_speed(stn, obs, t):
        if gust_mode == "reported":
            return obs.get("gust") or obs["wspd"]
        obs_list = obs_data[stn]
        before   = [i for i, o in enumerate(obs_list) if o["time"] <= t]
        idx      = before[-1] if before else 0
        return max(ou_p95[stn][idx], obs["wspd"])

    # ── render ────────────────────────────────────────────────────────────────
    def _render(frame_idx: int):
        for ref in (_qw, _qg):
            if ref[0] is not None:
                try:
                    ref[0].remove()
                except Exception:
                    pass
                ref[0] = None

        t = times[frame_idx]
        lons_w, lats_w, U_w, V_w, C_w = [], [], [], [], []
        lons_g, lats_g, U_g, V_g      = [], [], [], []
        spd_by_stn: dict[str, float]   = {}

        for stn in stations:
            obs = _obs_at(obs_data[stn], t, temporal_mode)
            if obs is None or obs.get("wspd") is None or obs.get("wdir") is None:
                spd_by_stn[stn] = 0.0
                continue

            spd_by_stn[stn] = obs["wspd"]
            lat, lon = coords[stn]
            u, v = _to_uv(obs["wdir"], obs["wspd"])
            lons_w.append(lon); lats_w.append(lat)
            U_w.append(u);      V_w.append(v);     C_w.append(obs["wspd"])

            gs = _gust_speed(stn, obs, t)
            if gs > obs["wspd"] * 1.05:
                ug, vg = _to_uv(obs["wdir"], gs)
                lons_g.append(lon); lats_g.append(lat)
                U_g.append(ug);     V_g.append(vg)

        # Update blend field
        if blend and _blend_grid_lat is not None:
            from weather.interpolate import interpolate_field
            snapshot = {s: obs for s, obs in [
                (stn, _obs_at(obs_data[stn], t, temporal_mode))
                for stn in stations
            ] if obs is not None}
            if len(snapshot) >= 2:
                field = interpolate_field(snapshot, _blend_grid_lat, _blend_grid_lon,
                                          method=spatial_method)
                sk = _shade[0]
                if sk == "wspd":
                    grid_vals = field["speed"]
                elif sk == "gust":
                    grid_vals = (field["gust_speed"] - field["speed"]).clip(0)
                elif sk == "temp":
                    grid_vals = field["T"]
                else:
                    import numpy as _np2
                    from scipy.interpolate import RBFInterpolator
                    raw_vals = _np2.array([_shade_value(obs_s, sk)
                                           for obs_s in snapshot.values()])
                    _lats = _np2.array([o["lat"] for o in snapshot.values()])
                    _lons = _np2.array([o["lon"] for o in snapshot.values()])
                    rbf = RBFInterpolator(
                        _np2.column_stack([_lats.ravel(), _lons.ravel()]),
                        raw_vals, smoothing=0.5
                    )
                    grid_vals = rbf(
                        _np2.column_stack([_blend_grid_lat.ravel(),
                                           _blend_grid_lon.ravel()])
                    ).reshape(_blend_grid_lat.shape)

                _cm, _nm = _render_cmap[0], _render_norm[0]
                if _blend_mesh[0] is None:
                    _blend_mesh[0] = ax.pcolormesh(
                        _blend_grid_lon, _blend_grid_lat, grid_vals,
                        cmap=_cm, norm=_nm, alpha=0.60,
                        transform=PROJ, zorder=4, shading="nearest",
                    )
                else:
                    _blend_mesh[0].set_array(grid_vals.ravel())
                    _blend_mesh[0].set_cmap(_cm)
                    _blend_mesh[0].set_norm(_nm)

        # Update Voronoi cell colors
        if vor_col is not None:
            _cm, _nm = _render_cmap[0], _render_norm[0]
            new_colors = []
            for s in vor_order:
                obs_s = _obs_at(obs_data[s], t, temporal_mode)
                val   = _shade_value(obs_s, _shade[0]) if obs_s else 0.0
                rgba  = list(to_rgba(_cm(_nm(val))))
                rgba[3] = 0.32
                new_colors.append(rgba)
            vor_col.set_facecolors(new_colors)
            vor_col.set_edgecolors("white")

        if lons_w:
            _qw[0] = ax.quiver(
                lons_w, lats_w, U_w, V_w, C_w,
                cmap=("gray" if blend else _render_cmap[0]),
                norm=(None if blend else _render_norm[0]),
                color=("gray" if blend else None),
                alpha=(0.45 if blend else 1.0),
                scale=(100 if blend else 50),
                width=(0.0025 if blend else 0.005),
                headwidth=3, headlength=4, headaxislength=3.5,
                transform=PROJ, zorder=6,
            )
        if lons_g:
            _qg[0] = ax.quiver(
                lons_g, lats_g, U_g, V_g,
                color=("dimgray" if blend else "cyan"),
                alpha=(0.30 if blend else 0.45),
                scale=(100 if blend else 50),
                width=(0.002 if blend else 0.004),
                headwidth=2, headlength=3, headaxislength=2.5,
                transform=PROJ, zorder=5,
            )

        clock.set_text(t.strftime("UTC  %Y-%m-%d  %H:%M"))

    # ── tick / slider / button (interactive only) ─────────────────────────────
    if not save_gif:
        def _tick(i):
            if _slider_dragging[0]:
                return []
            if _playing[0]:
                _frame[0] = (_frame[0] + 1) % n_frames
            _render(_frame[0])
            slider.eventson = False
            slider.set_val(_frame[0])
            slider.eventson = True
            fig.canvas.draw_idle()
            return []

        def _on_slider(val):
            _frame[0] = int(val)
            _render(_frame[0])
            fig.canvas.draw_idle()

        slider.on_changed(_on_slider)
        fig.canvas.mpl_connect("button_press_event",
                               lambda e: _slider_dragging.__setitem__(0, e.inaxes == sl_ax))
        fig.canvas.mpl_connect("button_release_event",
                               lambda e: _slider_dragging.__setitem__(0, False))

        def _toggle(event):
            _playing[0] = not _playing[0]
            btn.label.set_text("|| Pause" if _playing[0] else ">  Play")
            fig.canvas.draw_idle()

        btn.on_clicked(_toggle)

    else:
        # Non-interactive tick for GIF export
        def _tick(i):
            _render(i)
            return []

    fps         = n_steps / (total_hours / playback_speed)
    interval_ms = max(50, int(1000.0 / fps))

    ani = animation.FuncAnimation(
        fig, _tick, frames=n_frames,
        interval=interval_ms, blit=False, repeat=True,
    )

    # ── output ────────────────────────────────────────────────────────────────
    if save_gif:
        print(f"  Saving GIF → {save_gif}")
        print(f"  Frames: {n_frames}  |  FPS: {max(10, int(1000.0 / interval_ms))}")
        print("  This may take several minutes for long windows...")
        writer = animation.PillowWriter(fps=max(10, int(1000.0 / interval_ms)))
        ani.save(save_gif, writer=writer)
        print("  Done.")
        plt.close(fig)
    else:
        plt.show()

    return ani
