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


# ── Colormap for headwind ─────────────────────────────────────────────────────

def headwind_cmap():
    """Blue (tailwind) → white (neutral) → red (headwind)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "headwind", ["#2878c8", "#e8e8e8", "#d04020"]
    )


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_traversal(csv_path: str, map_style: str = "topo") -> None:
    d = load_csv(csv_path)
    dist = d["dist_km"]
    hw   = d["headwind"]
    hw_max = max(abs(np.nanmax(hw)), abs(np.nanmin(hw)), 1.0)
    norm  = mcolors.Normalize(vmin=-hw_max, vmax=hw_max)
    cmap  = headwind_cmap()

    fig = plt.figure(figsize=(12, 8), facecolor="#0d1117")
    fig.suptitle(
        os.path.basename(csv_path).replace(".csv", "")
        + "  —  mission environment",
        color="#c0c8d0", fontsize=11, y=0.98
    )

    # ── Map panel ─────────────────────────────────────────────────────────────
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        if map_style == "satellite":
            tiles = cimgt.GoogleTiles(style="satellite")
        elif map_style == "street":
            tiles = cimgt.OpenStreetMap()
        else:
            tiles = cimgt.StadiaMapsTiles(style="stamen_terrain") \
                if map_style == "topo" else None

        ax_map = fig.add_axes([0.03, 0.35, 0.60, 0.60], projection=proj)
        if tiles:
            try:
                ax_map.add_image(tiles, 7)
            except Exception:
                pass
        ax_map.set_facecolor("#0d1a26")

        pad = 0.4
        lon_min, lon_max = d["lon"].min() - pad, d["lon"].max() + pad
        lat_min, lat_max = d["lat"].min() - pad, d["lat"].max() + pad
        ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax_map.gridlines(draw_labels=True, linewidth=0.4, color="#ffffff",
                         alpha=0.15, linestyle="--")
    else:
        ax_map = fig.add_axes([0.03, 0.35, 0.60, 0.60], facecolor="#0d1a26")
        ax_map.set_xlim(d["lon"].min() - 0.3, d["lon"].max() + 0.3)
        ax_map.set_ylim(d["lat"].min() - 0.3, d["lat"].max() + 0.3)
        ax_map.tick_params(colors="#607080")
        for spine in ax_map.spines.values():
            spine.set_edgecolor("#1a3050")

    # Colored path segments
    for i in range(len(dist) - 1):
        x0, y0 = d["lon"][i],   d["lat"][i]
        x1, y1 = d["lon"][i+1], d["lat"][i+1]
        color  = cmap(norm(hw[i]))
        kw = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        ax_map.plot([x0, x1], [y0, y1], color=color, linewidth=2.0,
                    solid_capstyle="round", **kw)

    # Start / end markers
    kw = {"transform": ccrs.PlateCarree(), "zorder": 5} if HAS_CARTOPY else {"zorder": 5}
    ax_map.plot(d["lon"][0],  d["lat"][0],  "o", color="#1d9e75", ms=7, **kw)
    ax_map.plot(d["lon"][-1], d["lat"][-1], "s", color="#d85a30", ms=7, **kw)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, orientation="vertical",
                      fraction=0.025, pad=0.02)
    cb.set_label("headwind  m/s  (+head / −tail)", color="#9090a0", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#9090a0", labelcolor="#9090a0")

    ax_map.set_title("path colored by headwind", color="#8090a0",
                     fontsize=9, loc="left")

    # ── Stats table (right of map) ────────────────────────────────────────────
    ax_txt = fig.add_axes([0.66, 0.35, 0.32, 0.60], facecolor="#07111c")
    ax_txt.axis("off")
    total_dist = dist[-1]
    total_min  = (dist[-1] / dist[-1]) * ((d["lon"].shape[0]-1) * 1)
    # Derive mission duration from timestamps if available
    try:
        t0 = datetime.fromisoformat(d["time"][0].replace("Z","+00:00"))
        t1 = datetime.fromisoformat(d["time"][-1].replace("Z","+00:00"))
        dur_min = (t1 - t0).total_seconds() / 60
    except Exception:
        dur_min = float("nan")

    stats = [
        ("distance",        f"{total_dist:.1f} km"),
        ("duration",        f"{dur_min:.0f} min"),
        ("mean wind",       f"{np.nanmean(d['wind_spd']):.1f} m/s"),
        ("peak gust",       f"{np.nanmax(d['gust_spd']):.1f} m/s"),
        ("mean headwind",   f"{np.nanmean(hw):.1f} m/s"),
        ("max headwind",    f"{np.nanmax(hw):.1f} m/s"),
        ("max tailwind",    f"{abs(np.nanmin(hw)):.1f} m/s"),
        ("max crosswind",   f"{np.nanmax(np.abs(d['crosswind'])):.1f} m/s"),
        ("mean temp",       f"{np.nanmean(d['temp']):.1f} °C"),
        ("mean density",    f"{np.nanmean(d['density']):.4f} kg/m³"),
    ]
    for i, (label, val) in enumerate(stats):
        y = 0.94 - i * 0.086
        ax_txt.text(0.04, y, label, color="#507090", fontsize=9,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.96, y, val, color="#c0d0e0", fontsize=9,
                    fontweight="bold", ha="right", transform=ax_txt.transAxes)

    ax_txt.set_title("summary", color="#8090a0", fontsize=9, loc="left")

    # ── Time series: wind speed + gust ────────────────────────────────────────
    ax1 = fig.add_axes([0.06, 0.19, 0.88, 0.13], facecolor="#07111c")
    ax1.fill_between(dist, d["wind_spd"], d["gust_spd"],
                     alpha=0.15, color="#ba7517")
    ax1.plot(dist, d["wind_spd"], color="#378add", linewidth=1.2, label="wind spd")
    ax1.plot(dist, d["gust_spd"], color="#ba7517", linewidth=1.0,
             linestyle="--", label="gust")
    ax1.set_ylabel("m/s", color="#607080", fontsize=8)
    ax1.set_title("wind speed  /  gust", color="#8090a0", fontsize=9, loc="left")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.3,
               labelcolor="#c0d0e0", facecolor="#0d1117")
    ax1.tick_params(colors="#607080", labelbottom=False)
    for sp in ax1.spines.values(): sp.set_edgecolor("#1a3050")
    ax1.grid(axis="y", color="#1a3050", linewidth=0.5)
    ax1.set_xlim(dist[0], dist[-1])

    # ── Time series: headwind + crosswind ─────────────────────────────────────
    ax2 = fig.add_axes([0.06, 0.04, 0.88, 0.13], facecolor="#07111c")
    ax2.axhline(0, color="#2a3a4a", linewidth=0.8)
    ax2.fill_between(dist, hw, 0,
                     where=(hw >= 0), alpha=0.12, color="#d85a30")
    ax2.fill_between(dist, hw, 0,
                     where=(hw < 0),  alpha=0.12, color="#378add")
    ax2.plot(dist, hw, color="#d85a30", linewidth=1.2, label="headwind")
    ax2.plot(dist, d["crosswind"], color="#1d9e75", linewidth=1.0,
             linestyle="--", label="crosswind")
    ax2.set_xlabel("distance (km)", color="#607080", fontsize=8)
    ax2.set_ylabel("m/s", color="#607080", fontsize=8)
    ax2.set_title("headwind (+) / tailwind (−)  ·  crosswind",
                  color="#8090a0", fontsize=9, loc="left")
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.3,
               labelcolor="#c0d0e0", facecolor="#0d1117")
    ax2.tick_params(colors="#607080")
    for sp in ax2.spines.values(): sp.set_edgecolor("#1a3050")
    ax2.grid(axis="y", color="#1a3050", linewidth=0.5)
    ax2.set_xlim(dist[0], dist[-1])

    plt.rcParams["axes.labelcolor"] = "#607080"
    plt.rcParams["xtick.color"]     = "#607080"
    plt.rcParams["ytick.color"]     = "#607080"

    out_path = csv_path.replace(".csv", "_viz.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print(f"  Saved → {out_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize traversal CSV")
    p.add_argument("csv", help="Path to traversal CSV")
    p.add_argument("--map", choices=["topo","satellite","street","none"],
                   default="topo")
    args = p.parse_args()
    plot_traversal(args.csv, map_style=args.map)
