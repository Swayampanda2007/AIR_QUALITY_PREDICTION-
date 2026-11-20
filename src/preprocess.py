"""Prepare station-level observations and one-hour-ahead PM2.5 targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "processed" / "combined_air_quality.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "processed" / "air_quality_preprocessed.csv"
FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "station",
]
TARGET_COLUMN = "PM2.5_next_hour"


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create a target from the next hourly record at the same station."""
    required_columns = {"datetime", *FEATURE_COLUMNS}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Combined dataset is missing required columns: {sorted(missing_columns)}")

    data = data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    data = data.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Shift within each station so a target can never come from another station.
    station_groups = data.groupby("station", sort=False)
    data[TARGET_COLUMN] = station_groups["PM2.5"].shift(-1)
    next_datetime = station_groups["datetime"].shift(-1)

    # A later row is not a next-hour target if the station has a timestamp gap.
    is_next_hour = next_datetime.eq(data["datetime"] + pd.Timedelta(hours=1))
    data.loc[~is_next_hour, TARGET_COLUMN] = float("nan")
    prepared = data.dropna(subset=[TARGET_COLUMN]).copy()

    if prepared.empty:
        raise ValueError("No rows have a valid next-hour PM2.5 target.")

    return prepared[["datetime", *FEATURE_COLUMNS, TARGET_COLUMN]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Combined dataset not found: {args.input}. Run src\\load_data.py first.")

    source_data = pd.read_csv(args.input)
    prepared = prepare_data(source_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)

    print(f"Prepared dataset shape: {prepared.shape[0]:,} rows x {prepared.shape[1]} columns")
    print("\nMissing values by column (not imputed):")
    missing_report = pd.DataFrame(
        {
            "missing_count": prepared.isna().sum(),
            "missing_percentage": prepared.isna().mean().mul(100).round(2),
        }
    )
    print(missing_report.to_string())
    print(f"\n{TARGET_COLUMN} statistics:")
    print(prepared[TARGET_COLUMN].describe().to_string())
    print("\nFirst five records:")
    print(prepared.head().to_string(index=False))
    print(f"\nSaved preprocessed data to {args.output}")


if __name__ == "__main__":
    main()
