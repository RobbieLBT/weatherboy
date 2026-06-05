"""
mission/visualize.py

Plot a traversal CSV as a path overlay on a Cartopy basemap,
with headwind/gust time series subplots below.

Usage
-----
python -m mission.visualize output/VA-XC1.csv [--map topo|satellite|street|none]

Produces a static figure (saved as PNG alongside the CSV and shown interactively).
No animation — use run.py for the animated weather field.
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("  [INFO] Cartopy not found — map panel will use plain axes.")


MS_TO_KTS = 1.94384
KM_TO_NMI = 0.539957


# ── Tile sources ──────────────────────────────────────────────────────────────

class _OpenTopoMap(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://tile.opentopomap.org/{z}/{x}/{y}.png"

class _Satellite(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return (
            f"https://server.arcgisonline.com/ArcGIS/rest/services"
            f"/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )

class _Street(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"

_TILE_SOURCES = {
    "topo":      _OpenTopoMap,
    "satellite": _Satellite,
    "street":    _Street,
}


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def _f(key):
        return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])

    return {
        "lat":       _f("lat_deg"),
        "lon":       _f("lon_deg"),
        "dist_km":   _f("dist_from_start_km"),
        "heading":   _f("heading_deg"),
        "wind_spd":  _f("wind_speed_ms"),
        "gust_spd":  _f("gust_speed_ms"),
        "headwind":  _f("headwind_ms"),
        "crosswind": _f("crosswind_ms"),
        "temp":      _f("temp_c"),
        "density":   _f("density_kgm3"),
        "time": [r["time"] for r in rows],
    }


def headwind_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "headwind", ["#2878c8", "#f0f0f0", "#d04020"]
    )


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_traversal(csv_path: str, map_style: str = "topo") -> None:
    d = load_csv(csv_path)

    dist     = d["dist_km"]  * KM_TO_NMI
    wind_spd = d["wind_spd"] * MS_TO_KTS
    gust_spd = d["gust_spd"] * MS_TO_KTS
    hw       = d["headwind"] * MS_TO_KTS
    cw       = d["crosswind"]* MS_TO_KTS

    hw_max = max(abs(np.nanmax(hw)), abs(np.nanmin(hw)), 1.0)
    norm   = mcolors.Normalize(vmin=-hw_max, vmax=hw_max)
    cmap   = headwind_cmap()

    # ── Figure: use GridSpec for the top block ────────────────────────────────
    # Top block occupies rows 0, bottom two time-series occupy rows 1 and 2.
    # We use a plain figure + manually placed axes so we have pixel-level control.
    #
    # Figure: 14 × 9 inches at 150 dpi = 2100 × 1350 px
    # Top block  : left=0.02  right=0.98  bottom=0.32  top=0.93
    #   map      : left=0.02  width=0.50
    #   colorbar : left=0.53  width=0.02
    #   stats    : left=0.56  width=0.42
    # TS panels  : left=0.07  width=0.89  (leaves room for rotated "kts" label)

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle("Mission Environment", color="#111111", fontsize=12, y=0.97)

    TB  = 0.32   # top block bottom
    TT  = 0.93   # top block top
    TH  = TT - TB

    ML  = 0.02
    MW  = 0.50
    CBL = 0.53
    CBW = 0.018
    SL  = 0.565
    SW  = 0.42

    # ── Map ───────────────────────────────────────────────────────────────────
    if HAS_CARTOPY:
        proj   = ccrs.PlateCarree()
        ax_map = fig.add_axes([ML, TB, MW, TH], projection=proj)
        tile_cls = _TILE_SOURCES.get(map_style)
        if tile_cls is not None:
            try:
                ax_map.add_image(tile_cls(), 7)
            except Exception:
                pass
        ax_map.set_facecolor("#e8f0f8")
        pad = 0.4
        ax_map.set_extent(
            [d["lon"].min()-pad, d["lon"].max()+pad,
             d["lat"].min()-pad, d["lat"].max()+pad],
            crs=proj,
        )
        ax_map.gridlines(draw_labels=False, linewidth=0.4,
                         color="#333333", alpha=0.3, linestyle="--")
    else:
        ax_map = fig.add_axes([ML, TB, MW, TH], facecolor="#e8f0f8")
        ax_map.set_xlim(d["lon"].min()-0.3, d["lon"].max()+0.3)
        ax_map.set_ylim(d["lat"].min()-0.3, d["lat"].max()+0.3)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for sp in ax_map.spines.values():
            sp.set_edgecolor("#cccccc")

    for i in range(len(dist) - 1):
        color = cmap(norm(hw[i]))
        kw    = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        ax_map.plot(
            [d["lon"][i], d["lon"][i+1]],
            [d["lat"][i], d["lat"][i+1]],
            color=color, linewidth=2.0, solid_capstyle="round", **kw,
        )
    mkw = {"transform": ccrs.PlateCarree(), "zorder": 5} if HAS_CARTOPY else {"zorder": 5}
    ax_map.plot(d["lon"][0],  d["lat"][0],  "o", color="#1d9e75", ms=7, **mkw)
    ax_map.plot(d["lon"][-1], d["lat"][-1], "s", color="#d85a30", ms=7, **mkw)
    ax_map.set_title("path colored by headwind", color="#555555", fontsize=9, loc="left")

    # ── Colorbar — explicit axes, exactly same bottom/height as map ───────────
    ax_cb = fig.add_axes([CBL, TB, CBW, TH])
    sm    = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("headwind (kts)\n+head / −tail", color="#444444", fontsize=8,
                 labelpad=6)
    cb.ax.yaxis.set_tick_params(color="#444444", labelcolor="#444444", labelsize=8)

    # ── Stats panel — same bottom/height as map ────────────────────────────────
    ax_txt = fig.add_axes([SL, TB, SW, TH], facecolor="#f5f7fa")
    ax_txt.axis("off")

    try:
        t0 = datetime.fromisoformat(d["time"][0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(d["time"][-1].replace("Z", "+00:00"))
        dur_min = (t1 - t0).total_seconds() / 60
    except Exception:
        dur_min = float("nan")

    stats = [
        ("distance",      f"{d['dist_km'][-1] * KM_TO_NMI:.1f} nmi"),
        ("duration",      f"{dur_min:.0f} min"),
        ("mean wind",     f"{np.nanmean(wind_spd):.1f} kts"),
        ("peak gust",     f"{np.nanmax(gust_spd):.1f} kts"),
        ("mean headwind", f"{np.nanmean(hw):.1f} kts"),
        ("max headwind",  f"{np.nanmax(hw):.1f} kts"),
        ("max tailwind",  f"{abs(np.nanmin(hw)):.1f} kts"),
        ("max crosswind", f"{np.nanmax(np.abs(cw)):.1f} kts"),
        ("mean temp",     f"{np.nanmean(d['temp']):.1f} °C"),
        ("mean density",  f"{np.nanmean(d['density']):.4f} kg/m³"),
    ]
    for i, (label, val) in enumerate(stats):
        y = 0.94 - i * 0.086
        ax_txt.text(0.04, y, label, color="#666666", fontsize=9,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.96, y, val, color="#111111", fontsize=9,
                    fontweight="bold", ha="right", transform=ax_txt.transAxes)
    ax_txt.set_title("Summary", color="#444444", fontsize=9, loc="left")

    # ── Time series panels ────────────────────────────────────────────────────
    # Left edge 0.07 gives ~70px for the rotated "kts" ylabel at 150dpi.
    # Both panels share the same left/width so ylabels are in the same column.

    TSL  = 0.07
    TSW  = 0.91
    TSH  = 0.11
    A1B  = 0.185   # ax1 bottom
    A2B  = 0.03    # ax2 bottom  (gap = 0.185 - 0.03 - 0.11 = 0.045 fig frac ≈ 12px)

    ax1 = fig.add_axes([TSL, A1B, TSW, TSH], facecolor="white")
    ax1.fill_between(dist, wind_spd, gust_spd, alpha=0.12, color="#e65100")
    ax1.plot(dist, wind_spd, color="#1565c0", linewidth=1.2, label="wind spd")
    ax1.plot(dist, gust_spd, color="#e65100", linewidth=1.0,
             linestyle="--", label="gust")
    ax1.set_ylabel("kts", color="#444444", fontsize=8, rotation=90, labelpad=4)
    ax1.set_title("wind speed  /  gust", color="#444444", fontsize=9, loc="left")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.8,
               labelcolor="#111111", facecolor="white", edgecolor="#cccccc")
    ax1.tick_params(colors="#444444", labelbottom=False)
    for sp in ax1.spines.values(): sp.set_edgecolor("#cccccc")
    ax1.grid(axis="y", color="#e0e0e0", linewidth=0.5)
    ax1.set_xlim(dist[0], dist[-1])

    ax2 = fig.add_axes([TSL, A2B, TSW, TSH], facecolor="white")
    ax2.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax2.fill_between(dist, hw, 0, where=(hw >= 0), alpha=0.10, color="#d85a30")
    ax2.fill_between(dist, hw, 0, where=(hw <  0), alpha=0.10, color="#1565c0")
    ax2.plot(dist, hw, color="#d85a30", linewidth=1.2, label="headwind")
    ax2.plot(dist, cw, color="#1d9e75", linewidth=1.0,
             linestyle="--", label="crosswind")
    ax2.set_xlabel("distance (nmi)", color="#444444", fontsize=8)
    ax2.set_ylabel("kts", color="#444444", fontsize=8, rotation=90, labelpad=4)
    ax2.set_title("headwind (+) / tailwind (−)  ·  crosswind",
                  color="#444444", fontsize=9, loc="left")
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.8,
               labelcolor="#111111", facecolor="white", edgecolor="#cccccc")
    ax2.tick_params(colors="#444444")
    for sp in ax2.spines.values(): sp.set_edgecolor("#cccccc")
    ax2.grid(axis="y", color="#e0e0e0", linewidth=0.5)
    ax2.set_xlim(dist[0], dist[-1])

    plt.rcParams["axes.labelcolor"] = "#444444"
    plt.rcParams["xtick.color"]     = "#444444"
    plt.rcParams["ytick.color"]     = "#444444"

    out_path = csv_path.replace(".csv", "_viz.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize traversal CSV")
    p.add_argument("csv", help="Path to traversal CSV")
    p.add_argument("--map", choices=["topo", "satellite", "street", "none"],
                   default="topo")
    args = p.parse_args()
    plot_traversal(args.csv, map_style=args.map)