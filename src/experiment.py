"""Validation, chronological model comparison, evaluation, and reporting."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import FEATURE_COLUMNS, TARGET_COLUMN, prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "combined_air_quality.csv"
PREPROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "air_quality_preprocessed.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FRACTION = 0.8
RANDOM_STATE = 42
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column != "station"]
NONNEGATIVE_FEATURES = {
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "RAIN",
    "WSPM",
}
MODEL_NAMES = [
    "Linear Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Persistence Baseline",
]
TREE_MODEL_NAMES = ["Random Forest Regressor", "Gradient Boosting Regressor"]


def clean_and_validate_source(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate identifiers and timestamps, and mark invalid measurements missing."""
    required = {"datetime", *FEATURE_COLUMNS}
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise ValueError(f"Combined dataset is missing required columns: {sorted(missing_columns)}")

    data = data.copy()
    diagnostics = {
        "source_rows": len(data),
        "invalid_datetime_rows": 0,
        "missing_station_rows": 0,
        "duplicate_station_timestamps": 0,
        "invalid_measurements": {},
        "irregular_intervals": 0,
        "missing_hour_slots": 0,
    }

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    invalid_datetime = data["datetime"].isna()
    diagnostics["invalid_datetime_rows"] = int(invalid_datetime.sum())

    station_values = data["station"].astype("string").str.strip()
    invalid_station = station_values.isna() | station_values.eq("")
    diagnostics["missing_station_rows"] = int(invalid_station.sum())
    data["station"] = station_values.astype(object)
    data = data.loc[~(invalid_datetime | invalid_station)].copy()

    duplicate_count = int(data.duplicated(["station", "datetime"]).sum())
    diagnostics["duplicate_station_timestamps"] = duplicate_count
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate station/datetime observations; "
            "resolve them before modeling."
        )
    if data.empty:
        raise ValueError("No rows with valid datetime and station values remain.")

    # Invalid sensor values are retained as missing so model pipelines can impute them
    # using training data only. No high measurements are clipped or removed.
    for column in NUMERIC_FEATURES:
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        non_numeric = original.notna() & numeric.isna()
        non_finite = pd.Series(np.isinf(numeric.astype(float)), index=data.index)
        negative = numeric.lt(0) if column in NONNEGATIVE_FEATURES else pd.Series(False, index=data.index)
        invalid = non_numeric | non_finite | negative.fillna(False)
        diagnostics["invalid_measurements"][column] = {
            "non_numeric": int(non_numeric.sum()),
            "non_finite": int(non_finite.sum()),
            "negative": int(negative.sum()) if column in NONNEGATIVE_FEATURES else 0,
        }
        data[column] = numeric.mask(invalid)

    data = data.sort_values(["station", "datetime"]).reset_index(drop=True)
    intervals = data.groupby("station", sort=False)["datetime"].diff()
    one_hour = pd.Timedelta(hours=1)
    irregular = intervals.notna() & intervals.ne(one_hour)
    long_gaps = intervals.gt(one_hour)
    missing_slots = ((intervals[long_gaps] / one_hour).astype(int) - 1).clip(lower=0)
    diagnostics["irregular_intervals"] = int(irregular.sum())
    diagnostics["missing_hour_slots"] = int(missing_slots.sum())

    diagnostics["usable_source_rows"] = len(data)
    diagnostics["station_count"] = int(data["station"].nunique())
    diagnostics["source_start"] = data["datetime"].min()
    diagnostics["source_end"] = data["datetime"].max()
    return data, diagnostics


def verify_next_hour_targets(source: pd.DataFrame, prepared: pd.DataFrame) -> int:
    """Confirm each prepared target equals the next hourly PM2.5 at its station."""
    ordered = source.sort_values(["station", "datetime"]).reset_index(drop=True)
    grouped = ordered.groupby("station", sort=False)
    expected_target = grouped["PM2.5"].shift(-1)
    next_datetime = grouped["datetime"].shift(-1)
    valid = (
        next_datetime.eq(ordered["datetime"] + pd.Timedelta(hours=1))
        & expected_target.notna()
    )
    expected = ordered.loc[valid, ["datetime", "station"]].reset_index(drop=True)
    expected[TARGET_COLUMN] = expected_target.loc[valid].to_numpy()
    actual = prepared[["datetime", "station", TARGET_COLUMN]].reset_index(drop=True)

    if len(expected) != len(actual):
        raise ValueError(
            f"Target validation failed: expected {len(expected):,} valid rows, "
            f"but preprocessing produced {len(actual):,}."
        )
    if not expected[["datetime", "station"]].equals(actual[["datetime", "station"]]):
        raise ValueError("Target validation failed: prepared rows do not match source station/time rows.")
    if not np.allclose(expected[TARGET_COLUMN], actual[TARGET_COLUMN], rtol=0, atol=1e-10):
        raise ValueError("Target validation failed: a target differs from the next-hour station value.")
    return len(actual)


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by target time so training labels end before the holdout begins."""
    target_times = data["datetime"] + pd.Timedelta(hours=1)
    unique_times = target_times.drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_times) < 2:
        raise ValueError("At least two distinct timestamps are required for a chronological split.")

    split_index = int(len(unique_times) * TRAIN_FRACTION)
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    cutoff = unique_times.iloc[split_index]
    train = data.loc[target_times < cutoff].copy()
    test = data.loc[target_times >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty training or test set.")

    for station in data["station"].unique():
        station_train = train.loc[train["station"] == station, "datetime"] + pd.Timedelta(hours=1)
        station_test = test.loc[test["station"] == station, "datetime"] + pd.Timedelta(hours=1)
        if station_train.empty or station_test.empty or station_train.max() >= station_test.min():
            raise ValueError(f"Chronological split is invalid for station {station!r}.")
    return train, test, cutoff


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build imputers and station encoding to be fitted inside each model pipeline."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, ["station"]),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """Create reproducible models with train-only preprocessing pipelines."""
    return {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=16,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics for a common holdout sample."""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def save_actual_vs_predicted(predictions: pd.DataFrame) -> None:
    model_columns = MODEL_NAMES
    sample = predictions.sample(n=min(10_000, len(predictions)), random_state=RANDOM_STATE)
    low = min(sample["actual"].min(), sample[model_columns].min().min())
    high = max(sample["actual"].max(), sample[model_columns].max().max())

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, name in zip(axes.flat, model_columns):
        axis.scatter(sample["actual"], sample[name], alpha=0.25, s=8)
        axis.plot([low, high], [low, high], color="red", linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("Actual next-hour PM2.5")
        axis.set_ylabel("Predicted next-hour PM2.5")
    figure.suptitle("Actual vs predicted PM2.5 (chronological holdout)")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "actual_vs_predicted_pm25.png", dpi=150)
    plt.close(figure)


def save_model_comparison(metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, metric, title in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["Mean absolute error", "Root mean squared error", "R² score"],
    ):
        axis.bar(metrics["Model"], metrics[metric], color="steelblue")
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.set_ylabel(metric)
    figure.suptitle("Model comparison on the same chronological holdout")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "model_comparison.png", dpi=150)
    plt.close(figure)


def save_residual_distribution(predictions: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, name in zip(axes.flat, MODEL_NAMES):
        residuals = predictions["actual"] - predictions[name]
        axis.hist(residuals, bins=60, color="slateblue", alpha=0.85)
        axis.axvline(0, color="red", linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("Actual - predicted PM2.5")
        axis.set_ylabel("Frequency")
    figure.suptitle("Holdout residual distributions")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "residual_distribution.png", dpi=150)
    plt.close(figure)


def save_residual_analysis(predictions: pd.DataFrame) -> None:
    sample = predictions.sample(n=min(10_000, len(predictions)), random_state=RANDOM_STATE)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, name in zip(axes.flat, MODEL_NAMES):
        residuals = sample["actual"] - sample[name]
        axis.scatter(sample[name], residuals, alpha=0.25, s=8)
        axis.axhline(0, color="red", linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("Predicted next-hour PM2.5")
        axis.set_ylabel("Residual (actual - predicted)")
    figure.suptitle("Residuals vs predictions")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "residual_analysis.png", dpi=150)
    plt.close(figure)


def save_tree_feature_importance(models: dict[str, Pipeline]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 8))
    for axis, name in zip(axes, TREE_MODEL_NAMES):
        pipeline = models[name]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["regressor"].feature_importances_
        top = (
            pd.Series(importances, index=feature_names)
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        axis.barh(top.index, top.values, color="darkseagreen")
        axis.set_title(name)
        axis.set_xlabel("Feature importance")
    figure.suptitle("Tree-based model feature importance (top 15)")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "tree_feature_importance.png", dpi=150)
    plt.close(figure)


def save_time_series(predictions: pd.DataFrame) -> str:
    """Plot all holdout hours for a deterministic representative station."""
    station = sorted(predictions["station"].unique())[0]
    station_data = predictions.loc[predictions["station"] == station].sort_values("target_time")

    figure, axis = plt.subplots(figsize=(15, 6))
    axis.plot(station_data["target_time"], station_data["actual"], label="Actual", linewidth=1.2)
    for name in MODEL_NAMES:
        axis.plot(station_data["target_time"], station_data[name], label=name, linewidth=0.9, alpha=0.85)
    axis.set_title(f"Next-hour PM2.5 predictions for {station}")
    axis.set_xlabel("Target hour")
    axis.set_ylabel("PM2.5")
    axis.legend()
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "time_series_predictions.png", dpi=150)
    plt.close(figure)
    return station


def write_report(
    diagnostics: dict,
    prepared: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cutoff: pd.Timestamp,
    evaluation_rows: int,
    metrics: pd.DataFrame,
    best_trained_name: str,
    best_overall_name: str,
    model_configurations: dict[str, str],
    time_series_station: str,
) -> None:
    """Write observed dataset facts, configurations, metrics, and limitations."""
    train_range = f"{train['datetime'].min()} to {train['datetime'].max()}"
    test_range = f"{test['datetime'].min()} to {test['datetime'].max()}"
    train_target_range = (
        f"{train['datetime'].min() + pd.Timedelta(hours=1)} to "
        f"{train['datetime'].max() + pd.Timedelta(hours=1)}"
    )
    test_target_range = (
        f"{test['datetime'].min() + pd.Timedelta(hours=1)} to "
        f"{test['datetime'].max() + pd.Timedelta(hours=1)}"
    )
    invalid_total = sum(
        sum(counts.values()) for counts in diagnostics["invalid_measurements"].values()
    )
    lines = [
        "Short-Term Air Quality Prediction: Experiment Report",
        "=" * 58,
        "",
        "Dataset details",
        f"Input file: {COMBINED_DATA_PATH.relative_to(PROJECT_ROOT)}",
        f"Source rows: {diagnostics['source_rows']:,}",
        f"Rows after valid datetime/station checks: {diagnostics['usable_source_rows']:,}",
        f"Stations: {diagnostics['station_count']}",
        f"Source date range: {diagnostics['source_start']} to {diagnostics['source_end']}",
        f"Prepared rows with verified next-hour targets: {len(prepared):,}",
        f"Invalid datetime rows removed: {diagnostics['invalid_datetime_rows']:,}",
        f"Rows with missing/blank station removed: {diagnostics['missing_station_rows']:,}",
        f"Duplicate station/datetime rows: {diagnostics['duplicate_station_timestamps']:,}",
        f"Non-hourly consecutive station intervals: {diagnostics['irregular_intervals']:,}",
        f"Missing hourly slots within station records: {diagnostics['missing_hour_slots']:,}",
        f"Invalid sensor cells marked missing: {invalid_total:,}",
        "No high measurements were clipped or removed.",
        "Invalid sensor cells by feature (non-numeric / non-finite / negative):",
        *[
            f"  {column}: {counts['non_numeric']} / {counts['non_finite']} / {counts['negative']}"
            for column, counts in diagnostics["invalid_measurements"].items()
        ],
        f"Rows without a valid next-hour target: {diagnostics['usable_source_rows'] - len(prepared):,}",
        "",
        "Target and leakage checks",
        f"Target: {TARGET_COLUMN} from PM2.5 exactly one hour later at the same station.",
        "Target alignment was recomputed against the combined source and verified.",
        "Datetime and target columns are excluded from model input features.",
        f"Input features: {', '.join(FEATURE_COLUMNS)}",
        "Numeric features use median imputation; station uses most-frequent imputation "
        "and one-hot encoding.",
        "Each imputer/encoder is inside the model pipeline and is fitted on training data only.",
        "",
        "Chronological train/test split",
        "One shared target-time cutoff is used for all stations; records are never shuffled.",
        "Training labels end before the first holdout target time.",
        f"Training fraction of unique target timestamps: {TRAIN_FRACTION:.0%}",
        f"Cutoff (first test target timestamp): {cutoff}",
        f"Training records: {len(train):,} (feature times {train_range}; target times {train_target_range})",
        f"Test records: {len(test):,} (feature times {test_range}; target times {test_target_range})",
        f"Common evaluation records (current PM2.5 observed): {evaluation_rows:,}",
        "",
        "Model configurations",
    ]
    lines.extend(f"- {name}: {configuration}" for name, configuration in model_configurations.items())
    lines.extend(
        [
            "",
            "Holdout metrics",
            "All models are evaluated on the same test rows; persistence requires observed current PM2.5.",
            metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
            "",
            f"Best trained model by lowest holdout RMSE: {best_trained_name}",
            f"Best overall result by lowest holdout RMSE (including persistence): {best_overall_name}",
            "The best trained model pipeline, including its fitted preprocessing, is saved in models/.",
            f"Time-series plot station: {time_series_station}",
            "",
            "Important limitations",
            "- The evaluation uses a single chronological holdout from this Beijing dataset; "
            "it is not external validation.",
            "- The model predicts one hour ahead using measurements available at the current hour. "
            "It does not forecast future weather or pollutant inputs.",
            "- Persistence metrics omit test rows where current PM2.5 is missing so every model "
            "is compared on the same examples.",
            "- Missing numeric feature values are imputed with training-set medians; this can "
            "underrepresent unusual conditions.",
            "- Internal timestamp gaps are reported, and pairs crossing a gap are excluded; "
            "missing data before the first or after the last recorded hour cannot be detected.",
            "- Feature importance is model-specific and does not establish causal relationships.",
        ]
    )
    (RESULTS_DIR / "experiment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment() -> None:
    """Run source validation, preprocessing, chronological modeling, and outputs."""
    if not COMBINED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Combined dataset not found: {COMBINED_DATA_PATH}. Run src\\load_data.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(COMBINED_DATA_PATH, low_memory=False)
    source, diagnostics = clean_and_validate_source(source)
    prepared = prepare_data(source)
    validated_targets = verify_next_hour_targets(source, prepared)
    prepared.to_csv(PREPROCESSED_DATA_PATH, index=False)

    print("Data validation")
    print(f"  Source rows: {diagnostics['source_rows']:,}")
    print(f"  Usable rows: {diagnostics['usable_source_rows']:,}")
    print(f"  Stations: {diagnostics['station_count']}")
    print(f"  Invalid datetime rows removed: {diagnostics['invalid_datetime_rows']:,}")
    print(f"  Missing/blank station rows removed: {diagnostics['missing_station_rows']:,}")
    print(f"  Duplicate station/datetime rows: {diagnostics['duplicate_station_timestamps']:,}")
    print(f"  Non-hourly intervals: {diagnostics['irregular_intervals']:,}")
    print(f"  Missing hourly slots: {diagnostics['missing_hour_slots']:,}")
    print("  Invalid sensor values (non-numeric / non-finite / negative):")
    for column, counts in diagnostics["invalid_measurements"].items():
        print(
            f"    {column}: {counts['non_numeric']} / {counts['non_finite']} / {counts['negative']}"
        )
    print(f"  Verified one-hour targets: {validated_targets:,}")
    print(f"  Rows without valid targets: {len(source) - len(prepared):,}")
    print(f"  Preprocessed data saved to {PREPROCESSED_DATA_PATH}")

    train, test, cutoff = chronological_split(prepared)
    feature_columns = list(FEATURE_COLUMNS)
    if "datetime" in feature_columns or TARGET_COLUMN in feature_columns:
        raise ValueError("Datetime and target must not be used as model input features.")

    x_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    x_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]
    evaluation_mask = x_test["PM2.5"].notna() & np.isfinite(x_test["PM2.5"].astype(float))
    if not evaluation_mask.any():
        raise ValueError("No test records have current PM2.5 values for a persistence comparison.")

    fitted_models = build_models()
    prediction_values: dict[str, np.ndarray] = {}
    metric_rows = []
    model_configurations = {
        "Linear Regression": "LinearRegression(fit_intercept=True); median-impute and standardize numeric features.",
        "Random Forest Regressor": (
            "100 trees, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1."
        ),
        "Gradient Boosting Regressor": (
            "100 estimators, learning_rate=0.05, max_depth=3, random_state=42."
        ),
        "Persistence Baseline": "Predict next-hour PM2.5 as current-hour PM2.5.",
    }

    print("\nTraining models")
    for name, pipeline in fitted_models.items():
        start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        prediction_values[name] = pipeline.predict(x_test)
        elapsed = time.perf_counter() - start
        scores = calculate_metrics(
            y_test.loc[evaluation_mask],
            prediction_values[name][evaluation_mask.to_numpy()],
        )
        metric_rows.append({"Model": name, **scores, "Evaluation Samples": int(evaluation_mask.sum())})
        print(f"  {name}: finished in {elapsed:.1f} seconds")

    # Persistence is only defined where the current-hour PM2.5 measurement exists.
    prediction_values["Persistence Baseline"] = x_test.loc[evaluation_mask, "PM2.5"].to_numpy(dtype=float)
    evaluation_actual = y_test.loc[evaluation_mask]
    persistence_scores = calculate_metrics(
        evaluation_actual,
        prediction_values["Persistence Baseline"],
    )
    metric_rows.append(
        {
            "Model": "Persistence Baseline",
            **persistence_scores,
            "Evaluation Samples": int(evaluation_mask.sum()),
        }
    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_overall_name = metrics.loc[metrics["RMSE"].idxmin(), "Model"]
    trained_metrics = metrics.loc[metrics["Model"].isin(list(fitted_models))]
    best_trained_name = trained_metrics.loc[trained_metrics["RMSE"].idxmin(), "Model"]
    best_model_path = MODELS_DIR / "best_pm25_pipeline.joblib"
    joblib.dump(fitted_models[best_trained_name], best_model_path)

    evaluated = test.loc[evaluation_mask, ["datetime", "station"]].copy().reset_index(drop=True)
    evaluated["target_time"] = evaluated["datetime"] + pd.Timedelta(hours=1)
    evaluated["actual"] = evaluation_actual.to_numpy()
    for name in fitted_models:
        evaluated[name] = prediction_values[name][evaluation_mask.to_numpy()]
    evaluated["Persistence Baseline"] = prediction_values["Persistence Baseline"]
    evaluated.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    save_actual_vs_predicted(evaluated)
    save_model_comparison(metrics)
    save_residual_distribution(evaluated)
    save_residual_analysis(evaluated)
    save_tree_feature_importance(fitted_models)
    time_series_station = save_time_series(evaluated)

    write_report(
        diagnostics=diagnostics,
        prepared=prepared,
        train=train,
        test=test,
        cutoff=cutoff,
        evaluation_rows=int(evaluation_mask.sum()),
        metrics=metrics,
        best_trained_name=best_trained_name,
        best_overall_name=best_overall_name,
        model_configurations=model_configurations,
        time_series_station=time_series_station,
    )

    print("\nModel comparison (same chronological holdout rows)")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest trained model by RMSE: {best_trained_name}")
    print(f"Best overall result by RMSE: {best_overall_name}")
    print(f"Saved best trained pipeline to {best_model_path}")
    print(f"Saved report to {RESULTS_DIR / 'experiment_report.txt'}")
    print(f"Saved graphs to {GRAPHS_DIR}")
