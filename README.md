# Air Quality Prediction

This project combines, explores, and prepares hourly air-quality measurements
from 12 Beijing monitoring stations, then compares models for one-hour-ahead
PM2.5 prediction using a chronological holdout.

## Project layout

```text
dataset/
  raw/          Original station CSV files (12 expected)
  processed/    Combined and preprocessed datasets (generated)
src/            Data loading, exploration, preprocessing, and experiment modules
results/        Model comparison, predictions, and experiment report (generated)
models/         Best trained model pipeline (generated)
graphs/         Exploratory and experiment plots (generated)
research_paper/ Research material
requirements.txt
README.md
```

## Setup

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run from the VS Code terminal

Open the project root folder in VS Code, choose **Terminal > New Terminal**, and
run:

```powershell
python src\load_data.py
python src\explore_data.py
python src\preprocess.py
```

The first command reads the CSV files in `dataset/raw/`, retains each record's
station, combines usable files, and writes the result to
`dataset/processed/combined_air_quality.csv`. Warnings identify missing or
unreadable station files; the original CSVs are not changed.

The second command reads that combined dataset, prints the requested record,
column, missing-value, summary-statistic, duplicate, and station-count reports,
then saves these graphs in `graphs/`:

- `pm25_distribution.png`
- `pm10_distribution.png`
- `pm25_by_station.png`
- `numeric_correlation_heatmap.png`

The third command sorts by station and time, then pairs each row with PM2.5 from
the same station's following hour. It excludes rows without a valid next-hour
target and saves `dataset/processed/air_quality_preprocessed.csv`. Missing
feature values are reported and retained; imputation and train/test splitting
are deferred to the modeling stage to avoid data leakage.

To run validation, preprocessing, training, comparison, and report generation in
one command from the project root:

```powershell
python main.py
```

The experiment uses the earliest 80% of timestamps for training and the latest
20% for testing. Imputation and station encoding are fitted inside each model's
pipeline on training data only. `results/model_comparison.csv` contains the
holdout metrics, while `results/experiment_report.txt` records the split,
features, configurations, results, and limitations. The best trained model and
its fitted preprocessing pipeline are saved together in
`models/best_pm25_pipeline.joblib`.
