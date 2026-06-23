"""
SHAP Explainability Module (Requirement 1)
Selects the best-performing regression model for ADR prediction and generates
SHAP explainability artifacts: summary bar plot, beeswarm plot, and a ranked
list of the top features with business-oriented descriptions.

Reuses the existing src modules (data_loader, data_processing, modeling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.modeling import (
    evaluate_regression,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)

# Match the visual conventions used by eda_analysis.py
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, Any, dict[str, dict[str, float]]]:
    """Train the candidate regressors and return the best one by test R2.

    Returns (best_name, best_model, all_metrics).
    """
    candidates = {
        "LinearRegression": train_linear_regression(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    metrics = {
        name: evaluate_regression(model, X_test, y_test)
        for name, model in candidates.items()
    }

    best_name = max(metrics, key=lambda name: metrics[name]["R2"])
    return best_name, candidates[best_name], metrics


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> shap.Explanation:
    """Compute SHAP values for a fitted tree model on a sample of X.

    A sample is used so the beeswarm plot stays legible and computation is fast
    on the ~119k-row dataset. TreeExplainer is used for tree-based models.
    """
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample, check_additivity=False)
    return shap_values


def top_features(shap_values: shap.Explanation, n: int = 10) -> pd.DataFrame:
    """Return the top-n features ranked by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranking = (
        pd.DataFrame(
            {"feature": shap_values.feature_names, "mean_abs_shap": mean_abs}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    ranking.index = ranking.index + 1
    return ranking


def plot_shap_bar(shap_values: shap.Explanation, save_path: Path, max_display: int = 15) -> Path:
    """Generate and save the SHAP feature-importance bar plot (mean |SHAP|)."""
    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP value|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_path / "shap_feature_importance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_shap_beeswarm(shap_values: shap.Explanation, save_path: Path, max_display: int = 15) -> Path:
    """Generate and save the SHAP summary beeswarm plot (impact + direction)."""
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Summary: Feature Impact on Predicted ADR", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_path / "shap_summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out
