"""
plot_traversal.py
-----------------
Drop this in /output alongside traversal.csv.

Usage:
    python3 plot_traversal.py                        # reads traversal.csv
    python3 plot_traversal.py my_path.csv            # explicit file
    python3 plot_traversal.py my_path.csv --save     # save PNG instead of showing

Plots:
    Top    : wind speed (solid) and gust delta (dotted) vs. time
    Bottom : temperature vs. time
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Weatherboy traversal CSV.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="traversal.csv",
        help="Path to traversal CSV (default: traversal.csv)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNG alongside CSV instead of displaying interactively",
    )
    return parser.parse_args()


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse time column — try sim_time first, then elapsed_s
    if "sim_time" in df.columns:
        df["t"] = pd.to_datetime(df["sim_time"], utc=True)
    elif "elapsed_s" in df.columns:
        df["t"] = pd.to_timedelta(df["elapsed_s"], unit="s")
    else:
        df["t"] = df.index

    # Derive wind speed from NED components if not present
    if "wind_speed_ms" not in df.columns:
        if "wind_n_ms" in df.columns and "wind_e_ms" in df.columns:
            df["wind_speed_ms"] = (df["wind_n_ms"] ** 2 + df["wind_e_ms"] ** 2) ** 0.5
        else:
            raise KeyError("No wind speed column found. Expected 'wind_speed_ms' or 'wind_n_ms'/'wind_e_ms'.")

    # Derive gust delta = gust speed - mean wind speed
    if "gust_delta_ms" not in df.columns:
        if "gust_n_ms" in df.columns and "gust_e_ms" in df.columns:
            gust_speed = (df["gust_n_ms"] ** 2 + df["gust_e_ms"] ** 2) ** 0.5
            df["gust_delta_ms"] = (gust_speed - df["wind_speed_ms"]).clip(lower=0)
        else:
            raise KeyError("No gust columns found. Expected 'gust_n_ms'/'gust_e_ms'.")

    # Temperature — prefer Celsius
    if "temp_c" in df.columns:
        df["temp"] = df["temp_c"]
        temp_label = "Temperature (°C)"
    elif "temp_k" in df.columns:
        df["temp"] = df["temp_k"] - 273.15
        temp_label = "Temperature (°C)"
    else:
        raise KeyError("No temperature column found. Expected 'temp_c' or 'temp_k'.")

    return df, temp_label


def plot(df: pd.DataFrame, temp_label: str, csv_path: Path, save: bool):
    fig, (ax_wind, ax_temp) = plt.subplots(
        2, 1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
        facecolor="white",
    )

    for ax in (ax_wind, ax_temp):
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333", labelsize=9)
        ax.yaxis.label.set_color("#333333")
        ax.xaxis.label.set_color("#333333")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.grid(True, color="#e0e0e0", linewidth=0.6, linestyle="--")

    t = df["t"]

    # ── Top: wind speed + gust delta ──────────────────────────────────────
    ax_wind.plot(t, df["wind_speed_ms"], color="#1565c0", linewidth=1.4,
                 label="Wind speed (m/s)", solid_capstyle="round")
    ax_wind.plot(t, df["gust_delta_ms"], color="#c62828", linewidth=1.1,
                 linestyle="dotted", label="Gust delta (m/s)", dash_capstyle="round")
    ax_wind.set_ylabel("m/s", fontsize=9, color="#333333")
    ax_wind.legend(
        loc="upper right", fontsize=8,
        facecolor="white", edgecolor="#cccccc", labelcolor="#333333",
    )

    # ── Bottom: temperature ───────────────────────────────────────────────
    ax_temp.plot(t, df["temp"], color="#2e7d32", linewidth=1.4,
                 solid_capstyle="round")
    ax_temp.set_ylabel(temp_label, fontsize=9, color="#333333")

    # ── X-axis formatting ─────────────────────────────────────────────────
    if isinstance(t.iloc[0], pd.Timestamp):
        ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%Mz"))
        ax_temp.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30, ha="right")
        ax_temp.set_xlabel("Time (UTC)", fontsize=9, color="#333333")
    elif isinstance(t.iloc[0], pd.Timedelta):
        t_min = df["t"].dt.total_seconds() / 60
        ax_wind.lines[0].set_xdata(t_min)
        ax_wind.lines[1].set_xdata(t_min)
        ax_temp.lines[0].set_xdata(t_min)
        ax_wind.set_xlim(t_min.iloc[0], t_min.iloc[-1])
        ax_temp.set_xlim(t_min.iloc[0], t_min.iloc[-1])
        ax_temp.set_xlabel("Elapsed time (min)", fontsize=9, color="#333333")
    else:
        ax_temp.set_xlabel("Sample index", fontsize=9, color="#333333")

    # ── Title ─────────────────────────────────────────────────────────────
    title = csv_path.stem.replace("_", " ").title()
    fig.suptitle(f"Weatherboy · {title}", color="#111111", fontsize=11, y=0.98)

    plt.tight_layout()

    if save:
        out = csv_path.with_suffix(".png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
    else:
        plt.show()


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    df, temp_label = load(csv_path)
    plot(df, temp_label, csv_path, args.save)


if __name__ == "__main__":
    main()