"""Load and combine the station-level raw air-quality CSV files."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "processed" / "combined_air_quality.csv"
EXPECTED_STATION_FILES = 12
TIME_COLUMNS = ["year", "month", "day", "hour"]


def load_raw_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read usable station CSVs, reporting files that are missing or unreadable."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in raw data directory: {raw_dir}")
    if len(files) != EXPECTED_STATION_FILES:
        warnings.warn(
            f"Expected {EXPECTED_STATION_FILES} station CSV files, but found {len(files)} in {raw_dir}.",
            RuntimeWarning,
            stacklevel=2,
        )

    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            warnings.warn(f"Skipping unreadable file {path.name}: {error}", RuntimeWarning, stacklevel=2)
            continue

        missing = set(TIME_COLUMNS + ["station"]) - set(frame.columns)
        if missing:
            warnings.warn(
                f"Skipping {path.name}; missing required columns: {sorted(missing)}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        frame = frame.drop(columns=["No"], errors="ignore")
        try:
            frame["datetime"] = pd.to_datetime(frame[TIME_COLUMNS], errors="raise")
        except (TypeError, ValueError) as error:
            warnings.warn(f"Skipping {path.name}; invalid date/time values: {error}", RuntimeWarning, stacklevel=2)
            continue
        frames.append(frame)

    if not frames:
        raise ValueError(f"No readable station CSV files with the required columns were found in {raw_dir}.")

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["station", "datetime"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = load_raw_data(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Combined dataset shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Stations included: {data['station'].nunique()}")
    print(f"Saved combined dataset to {args.output}")


if __name__ == "__main__":
    main()
