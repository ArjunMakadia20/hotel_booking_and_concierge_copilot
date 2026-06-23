"""
Reproducible modeling pipeline for Hotel Booking Demand (ADR regression).

Wraps the existing src modules into a fast, sampled, end-to-end flow that the
notebook and the runner script both call. Sampling and dropping high-cardinality
identifier columns keep training fast and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.data_loader import get_default_hotel_booking_path, load_hotel_booking_data
from src.data_processing import (
    clean_hotel_booking_data,
    engineer_features,
    prepare_features,
    split_data,
)
from src.modeling import (
    evaluate_regression,
    train_linear_regression,
    train_random_forest,
)

# High-cardinality identifiers / post-outcome columns that explode one-hot
# encoding and slow training without helping a "basic above-average" model.
DROP_COLUMNS = ["country", "agent", "company", "reservation_status"]

try:  # XGBoost preferred; fall back to GradientBoosting if unavailable
    from src.modeling import train_xgboost

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - exercised only when xgboost missing
    HAS_XGBOOST = False


def load_clean_engineer(
    csv_path: Path | str | None = None,
    sample_size: int | None = 25000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load, clean, feature-engineer and (optionally) sample the dataset."""
    path = Path(csv_path) if csv_path else get_default_hotel_booking_path()
    df = load_hotel_booking_data(path)
    df = clean_hotel_booking_data(df)
    df = engineer_features(df)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def build_splits(
    df: pd.DataFrame,
    target: str = "adr",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Encode features and split into train/test."""
    X, y = prepare_features(df, target_column=target)
    return split_data(X, y, test_size=test_size, random_state=random_state)


def train_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train the candidate regressors and return (models, metrics_dataframe)."""
    models: dict[str, Any] = {
        "LinearRegression": train_linear_regression(X_train, y_train),
        "RandomForest": train_random_forest(
            X_train, y_train, n_estimators=100, max_depth=20, random_state=random_state
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = train_xgboost(X_train, y_train, random_state=random_state)
    else:
        gb = GradientBoostingRegressor(random_state=random_state)
        gb.fit(X_train, y_train)
        models["GradientBoosting"] = gb

    rows = []
    for name, model in models.items():
        m = evaluate_regression(model, X_test, y_test)
        rows.append({"model": name, "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"]})

    metrics_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return models, metrics_df


def save_comparison(metrics_df: pd.DataFrame, reports_dir: Path) -> Path:
    """Persist the model comparison table to reports/model_comparison.csv."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "model_comparison.csv"
    metrics_df.to_csv(out, index=False)
    return out


# Tree-based models eligible for TreeExplainer-based SHAP.
TREE_MODELS = {"RandomForest", "XGBoost", "GradientBoosting"}


def select_best_tree_model(
    metrics_df: pd.DataFrame, models: dict[str, Any]
) -> tuple[str, Any]:
    """Return the best tree-based model (by R2) for SHAP explainability."""
    tree_rows = metrics_df[metrics_df["model"].isin(TREE_MODELS)]
    best_name = tree_rows.iloc[0]["model"]
    return best_name, models[best_name]
