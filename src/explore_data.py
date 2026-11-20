"""Summarize the combined dataset and save the requested exploratory plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from load_data import DEFAULT_OUTPUT, PROJECT_ROOT

DEFAULT_GRAPHS = PROJECT_ROOT / "graphs"


def explore(data: pd.DataFrame, graph_dir: Path) -> None:
    """Print dataset diagnostics and create four reproducible plots."""
    required = {"PM2.5", "PM10", "station"}
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise ValueError(f"Combined data is missing required columns: {sorted(missing_columns)}")

    graph_dir.mkdir(parents=True, exist_ok=True)

    print("\nFirst five records:")
    print(data.head())
    print("\nColumn names and data types:")
    print(data.dtypes.to_string())

    missing_report = pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_percentage": data.isna().mean().mul(100).round(2),
        }
    )
    print("\nMissing values:")
    print(missing_report.to_string())
    print("\nStatistical summary:")
    print(data.describe(include="all").to_string())
    print(f"\nDuplicate records: {data.duplicated().sum():,}")
    print("\nRecords per station:")
    print(data["station"].value_counts(dropna=False).to_string())

    # Histograms show the measured distributions without altering missing values.
    for column, filename, title in [
        ("PM2.5", "pm25_distribution.png", "PM2.5 distribution"),
        ("PM10", "pm10_distribution.png", "PM10 distribution"),
    ]:
        plt.figure(figsize=(9, 5))
        sns.histplot(data=data, x=column, bins=50, color="steelblue")
        plt.title(title)
        plt.xlabel(f"{column} concentration")
        plt.ylabel("Number of records")
        plt.tight_layout()
        plt.savefig(graph_dir / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="station", y="PM2.5", showfliers=False, color="lightblue")
    plt.title("PM2.5 comparison across monitoring stations")
    plt.xlabel("Monitoring station")
    plt.ylabel("PM2.5 concentration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(graph_dir / "pm25_by_station.png", dpi=150)
    plt.close()

    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Correlation heatmap of numerical variables")
    plt.tight_layout()
    plt.savefig(graph_dir / "numeric_correlation_heatmap.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPHS)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Combined dataset not found: {args.input}. Run src\\load_data.py first."
        )
    data = pd.read_csv(args.input, parse_dates=["datetime"])
    explore(data, args.graph_dir)
    print(f"\nSaved four graphs to {args.graph_dir}")


if __name__ == "__main__":
    main()
