"""
End-to-end pipeline for Hotel Booking **Cancellation** classification.

Runs the whole workflow:
    load -> EDA -> clean/preprocess -> stratified 80k split -> train 5 models ->
    evaluate -> save reports/plots -> persist best model + preprocessor.

Call ``run_pipeline()`` (also exposed via the root ``run_pipeline.py`` script).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

from src.data_loader import get_default_hotel_booking_path, load_hotel_booking_data
from src.data_processing import (
    build_preprocessor,
    clean_hotel_booking_data,
    prepare_xy,
    split_data,
)
from src.eda_analysis import iqr_outlier_report, run_eda_analysis
from src.modeling import (
    comparison_dataframe,
    save_model,
    select_best_model,
    train_and_evaluate,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Friendly model name -> file slug for per-model report/plot filenames.
SLUG = {
    "LogisticRegression": "logistic_regression",
    "DecisionTree": "decision_tree",
    "RandomForest": "random_forest",
    "XGBoost": "xgboost",
    "MLP": "mlp",
}


def _save_confusion_matrix(name: str, cm, reports_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Cancelled", "Cancelled"],
                yticklabels=["Not Cancelled", "Cancelled"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / f"confusion_matrix_{SLUG[name]}.png", dpi=200, bbox_inches="tight")
    plt.close()


def _save_classification_reports(results: dict[str, dict[str, Any]], reports_dir: Path) -> None:
    """Write a single consolidated markdown report of every model."""
    lines = ["# Classification Reports — Cancellation Models\n"]
    for name, m in results.items():
        lines.append(f"## {name}\n")
        lines.append(f"- Accuracy: {m['accuracy']:.4f}")
        lines.append(f"- Precision: {m['precision']:.4f}")
        lines.append(f"- Recall: {m['recall']:.4f}")
        lines.append(f"- F1-score: {m['f1']:.4f}")
        lines.append(f"- ROC-AUC: {m['roc_auc']:.4f}")
        lines.append(f"- Train time: {m['train_time_sec']}s\n")
        lines.append("```")
        lines.append(m["report"].rstrip())
        lines.append("```\n")
    (reports_dir / "classification_reports.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    csv_path: Path | str | None = None,
    train_size: int = 80000,
    run_eda: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute the full cancellation-classification workflow."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load & clean -----------------------------------------------------
    path = Path(csv_path) if csv_path else get_default_hotel_booking_path()
    df_raw = load_hotel_booking_data(path)
    df = clean_hotel_booking_data(df_raw)
    print(f"Loaded {len(df_raw):,} rows -> cleaned {df.shape}")

    # --- EDA --------------------------------------------------------------
    if run_eda:
        run_eda_analysis(df, REPORTS_DIR)
        outliers = iqr_outlier_report(df)
        outliers.to_csv(REPORTS_DIR / "outlier_report.csv", index=False)
        print("Outlier report (IQR):")
        print(outliers.to_string(index=False))

    # --- Features / split -------------------------------------------------
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, train_size=train_size, random_state=random_state
    )
    print(f"\nSplit: train={len(X_train):,}  test={len(X_test):,}  "
          f"cancel rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    # --- Preprocess (fit on train only; NO scaling) -----------------------
    preprocessor = build_preprocessor(X_train)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    print(f"Encoded features: {X_train_enc.shape[1]}")

    # --- Train + evaluate 5 models ---------------------------------------
    print("\nTraining models:")
    fitted, results = train_and_evaluate(
        X_train_enc, y_train, X_test_enc, y_test, random_state=random_state
    )

    # --- Save comparison + per-model artefacts ---------------------------
    comp = comparison_dataframe(results)
    comp.to_csv(REPORTS_DIR / "classification_model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(comp.to_string(index=False))

    _save_classification_reports(results, REPORTS_DIR)
    for name, m in results.items():
        _save_confusion_matrix(name, m["confusion_matrix"], REPORTS_DIR)

    # --- Select best, build serveable pipeline, persist -------------------
    best_name, best_model = select_best_model(results, fitted)
    print(f"\nBest model (ROC-AUC): {best_name}")

    # Chain the already-fitted preprocessor + model so the API can call
    # predict/predict_proba on RAW booking input directly.
    serving = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
    save_model(serving, MODELS_DIR / "best_cancellation_model.pkl")
    save_model(preprocessor, MODELS_DIR / "preprocessor.pkl")
    print(f"Saved -> models/best_cancellation_model.pkl  (+ preprocessor.pkl)")

    return {
        "comparison": comp,
        "results": results,
        "best_name": best_name,
        "serving_pipeline": serving,
    }


if __name__ == "__main__":
    run_pipeline()
