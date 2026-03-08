#!/usr/bin/env python3
"""
python create_ar_csv.py

Generate CSV index files for Active Region (AR) segmentation datasets.
Scans for .h5 mask files and creates CSV indices matching the format
expected by dataset.py (columns: timestamp, file_path, present).

Author: [Rohit Lal] (modified by Kang Yang)
Date: [2025-03-02]
"""

import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def fetch_h5_files(directory, start_date, end_date):
    """
    Recursively find all .h5 files that fall within the specified date range.
    Returns a set of relative file paths.
    """
    pattern = re.compile(r"(\d{8})_(\d{4})\.h5")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    matching_files = set()

    for filepath in sorted(Path(directory).rglob("*.h5")):
        match = pattern.match(filepath.name)
        if not match:
            continue
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            ts = pd.Timestamp(
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
                f"{time_str[:2]}:{time_str[2:]}:00"
            )
        except ValueError:
            continue

        if start <= ts <= end:
            rel_path = f"data/{ts.year}/{ts.month:02d}/{filepath.name}"
            matching_files.add(rel_path)

    return matching_files


def create_csv_index(mask_dir, start_date, end_date, interval_minutes, csv_output):
    """
    Create a CSV index for AR segmentation .h5 files.
    """
    print(f"Scanning for .h5 files in {mask_dir} ...")
    existing_files = fetch_h5_files(mask_dir, start_date, end_date)
    print(f"Found {len(existing_files)} .h5 files")

    time_intervals = pd.date_range(
        start=start_date, end=end_date, freq=f"{interval_minutes}min"
    )
    print(f"Generated {len(time_intervals)} time intervals ({interval_minutes}min)")

    records = []
    matched = 0
    for t in tqdm(time_intervals, desc="Building index"):
        fname = t.strftime("%Y%m%d_%H%M") + ".h5"
        rel_path = f"data/{t.year}/{t.month:02d}/{fname}"
        present = 1.0 if rel_path in existing_files else 0.0
        if present:
            matched += 1
        records.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": rel_path if present else "",
            "present": present,
        })

    df = pd.DataFrame(records)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_output, index=False)
    print(f"Saved to {csv_output}")
    print(f"Matched: {matched}/{len(time_intervals)} ({matched/len(time_intervals)*100:.1f}%)")
    return df


def main():
    cwd = Path(__file__).parent.resolve()
    mask_dir = cwd / "assets" / "surya-bench-ar-segmentation"
    output_dir = cwd / "assets"

    print("Generating AR segmentation CSV index ...")
    print("Date range: 2010-01-01 to 2024-12-31")
    print("Interval: 12 minutes")

    create_csv_index(
        mask_dir=mask_dir,
        start_date="2010-01-01",
        end_date="2024-12-31",
        interval_minutes=12,
        csv_output=output_dir / "ar_index_12min.csv",
    )


if __name__ == "__main__":
    main()