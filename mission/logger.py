"""
mission/logger.py

Write ForcingRecord lists to CSV.

CSV column order follows ForcingRecord field declaration order.
One header row, one data row per sample. Missing/None values written as empty.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from mission.forcing import ForcingRecord, csv_header, record_to_row


def write_csv(records: list[ForcingRecord], output_path: str) -> None:
    """
    Write traversal records to a CSV file.

    Creates parent directories if they do not exist.
    Overwrites any existing file at output_path.
    """
    if not records:
        print("  [WARNING] No records to write.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header())
        for rec in records:
            writer.writerow(record_to_row(rec))

    print(f"  Wrote {len(records)} records → {output_path}")


def print_summary(records: list[ForcingRecord]) -> None:
    """Print a quick statistical summary of a traversal to stdout."""
    if not records:
        return

    speeds   = [r.wind_speed_ms  for r in records]
    gusts    = [r.gust_speed_ms  for r in records]
    headwds  = [r.headwind_ms    for r in records]
    cats     = [r.flight_category for r in records if r.flight_category]

    def _fmt(vals):
        return f"min={min(vals):.1f}  mean={sum(vals)/len(vals):.1f}  max={max(vals):.1f}"

    t_start = records[0].time
    t_end   = records[-1].time
    elapsed = (t_end - t_start).total_seconds() / 60

    print(f"\n  ── Traversal Summary ────────────────────────────────")
    print(f"  Samples       : {len(records)}")
    print(f"  Duration      : {elapsed:.1f} min  ({t_start:%H:%MZ} → {t_end:%H:%MZ})")
    print(f"  Distance      : {records[-1].dist_from_start_km:.1f} km")
    print(f"  Wind speed    : {_fmt(speeds)} m/s")
    print(f"  Gust speed    : {_fmt(gusts)} m/s")
    print(f"  Headwind      : {_fmt(headwds)} m/s")
    if cats:
        from collections import Counter
        dist = Counter(cats)
        cat_str = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  Flight cat    : {cat_str}")
    print(f"  ─────────────────────────────────────────────────────\n")
