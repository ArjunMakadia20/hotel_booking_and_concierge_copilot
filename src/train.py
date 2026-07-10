"""
Official training entry point for the cancellation model.

Trains XGBoost (the model selected via ``run_pipeline.py`` /
``reports/classification_model_comparison.csv``) on the ~80k-record stratified
split, evaluates it, saves the serving pipeline to ``models/``, and logs the run
to MLflow.

Runs TWO experiments under one MLflow experiment, as requested:
    - "baseline"           : current production config (n_estimators=300,
                              max_depth=6, learning_rate=0.1) -> models/best_cancellation_model.pkl
    - "tuned_max_depth_10"  : same config with ONE parameter changed
                              (max_depth 6 -> 10) -> models/xgboost_tuned_max_depth10.pkl
                              (saved separately; does NOT replace the production model)

Run:
    python -m src.train
    mlflow ui   # http://127.0.0.1:5000 — compare the two runs side by side
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import get_default_hotel_booking_path, load_hotel_booking_data
from src.data_processing import (
    build_preprocessor,
    clean_hotel_booking_data,
    prepare_xy,
    remove_invalid_rows,
    split_data,
)
from src.modeling import evaluate_classification, save_model

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
EXPERIMENT_NAME = "hotel_cancellation_classification"

# The two experiments requested by the supervisor: "baseline" is the current
# production configuration; "tuned" changes exactly one hyperparameter (max_depth).
RUN_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "model_filename": "best_cancellation_model.pkl",
    },
    "tuned_max_depth_10": {
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.1,
        "model_filename": "xgboost_tuned_max_depth10.pkl",
    },
}


def _load_split(train_size: int = 80000, random_state: int = 42):
    """Load, clean and split the dataset once so both runs train on identical data."""
    df = clean_hotel_booking_data(load_hotel_booking_data(get_default_hotel_booking_path()))
    df = remove_invalid_rows(df)
    X, y = prepare_xy(df)
    return split_data(X, y, train_size=train_size, random_state=random_state)


def _log_confusion_matrix(cm, run_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Cancelled", "Cancelled"],
                yticklabels=["Not Cancelled", "Cancelled"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_name}", fontweight="bold")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def train_one_run(
    run_name: str,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train + evaluate one XGBoost configuration and log it as an MLflow run."""
    preprocessor = build_preprocessor(X_train)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )

    with mlflow.start_run(run_name=run_name):
        start = time.perf_counter()
        model.fit(X_train_enc, y_train)
        train_time = time.perf_counter() - start

        metrics = evaluate_classification(model, X_test_enc, y_test)

        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"],
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "train_time_sec": train_time,
            }
        )
        _log_confusion_matrix(metrics["confusion_matrix"], run_name)

        # Serving pipeline = preprocessor + model, so predict.py / api.py can call
        # predict_proba directly on raw booking input.
        # serialization_format="pickle": mlflow's default "skops" format rejects
        # XGBoost's Booster/XGBClassifier as "untrusted" types.
        serving = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        mlflow.sklearn.log_model(serving, name="model", serialization_format="pickle")

        # Also persist to models/ for the app scripts (predict_offline.py, api.py).
        model_path = MODELS_DIR / config["model_filename"]
        save_model(serving, model_path)
        if run_name == "baseline":
            save_model(preprocessor, MODELS_DIR / "preprocessor.pkl")

        print(
            f"  [{run_name}] acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
            f"roc_auc={metrics['roc_auc']:.3f} ({train_time:.1f}s) -> {model_path.name}"
        )

        return {
            "run_name": run_name,
            "model_path": str(model_path),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "train_time_sec": round(train_time, 3),
        }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading & splitting data (~80k train)...")
    X_train, X_test, y_train, y_test = _load_split()
    print(f"train={len(X_train):,}  test={len(X_test):,}")

    print("\nTraining + logging MLflow runs:")
    results = [
        train_one_run(name, cfg, X_train, X_test, y_train, y_test)
        for name, cfg in RUN_CONFIGS.items()
    ]

    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== Experiment comparison ===")
    print(comparison.to_string(index=False))
    print(
        f"\nMLflow experiment: '{EXPERIMENT_NAME}'. Run `mlflow ui` and open "
        "http://127.0.0.1:5000 to compare runs visually."
    )


if __name__ == "__main__":
    main()
