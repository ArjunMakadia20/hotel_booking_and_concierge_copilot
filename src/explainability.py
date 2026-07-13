"""
SHAP explainability for the native-categorical XGBoost cancellation model.

The canonical served artefact is a ``preprocessor + XGBClassifier`` pipeline
(``models/xgboost_leakage_removed.pkl``); the same functions also explain the earlier
``models/xgboost_categorical_fix.pkl`` reference model. SHAP is a model-level
explanation, so it runs against the inner ``XGBClassifier`` on the *transformed*
feature matrix produced by the fitted preprocessor; the one-hot columns, numeric
passthroughs and the two native-categorical ID columns (``agent``/``company``) are all
fed to ``shap.TreeExplainer`` with the same dtypes XGBoost trained on.

All explanations are computed on the held-out TEST split only — SHAP never touches the
training data and never refits the model. Two importance views are provided: a
per-transformed-feature ranking (drives the beeswarm) and an aggregated ranking that
sums the one-hot columns back to their source feature, which is what the domain
sanity-check reads (does ``deposit_type`` rank high, where do ``agent``/``company``
land, is any post-booking feature a dominant driver).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.data_processing import (
    CATEGORICAL_COLUMNS,
    NATIVE_CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)

logger = logging.getLogger(__name__)

_TRANSFORMER_PREFIXES = ("cat__", "num__", "native__")


def transform_test_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return the fitted preprocessor's transformed feature frame for ``X``.

    The categorical/native columns keep their trained dtypes because the preprocessor
    was built with ``set_output('pandas')``; this frame is exactly what the inner
    ``XGBClassifier`` consumes, so SHAP values line up with the model's own features.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if not isinstance(transformed, pd.DataFrame):
        raise TypeError(
            "preprocessor did not return a DataFrame; native categorical dtypes "
            "cannot be preserved for SHAP"
        )
    return transformed


def _source_feature(transformed_name: str) -> str:
    """Map a transformed column name back to its original booking feature.

    ``num__lead_time`` -> ``lead_time``; ``native__agent`` -> ``agent``;
    ``cat__deposit_type_No Deposit`` -> ``deposit_type`` (the longest matching
    categorical column is used so multi-word names resolve correctly).
    """
    name = transformed_name
    for prefix in _TRANSFORMER_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if transformed_name.startswith("num__") or transformed_name.startswith("native__"):
        return name
    candidates = [c for c in CATEGORICAL_COLUMNS if name.startswith(f"{c}_")]
    if candidates:
        return max(candidates, key=len)
    return name


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[shap.TreeExplainer, shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for the pipeline's XGBoost model on ``X`` (test split).

    Uses ``shap.TreeExplainer`` in its tree-path-dependent mode (no background data,
    exact for tree ensembles, and able to follow XGBoost's native categorical splits).
    Returns the explainer, the ``Explanation`` object (log-odds for the cancelled
    class) and the transformed feature frame the values are aligned to.
    """
    model = pipeline.named_steps["model"]
    transformed = transform_test_features(pipeline, X)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)
    logger.info(
        "computed SHAP values for %d rows x %d features",
        explanation.values.shape[0],
        explanation.values.shape[1],
    )
    return explainer, explanation, transformed


def transformed_feature_ranking(explanation: shap.Explanation) -> pd.DataFrame:
    """Rank transformed features by mean absolute SHAP value (most important first)."""
    mean_abs = np.abs(explanation.values).mean(axis=0)
    ranking = pd.DataFrame(
        {"feature": list(explanation.feature_names), "mean_abs_shap": mean_abs}
    )
    return ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def original_feature_ranking(explanation: shap.Explanation) -> pd.DataFrame:
    """Aggregate SHAP importance back to original booking features.

    One-hot columns of the same source (e.g. every ``deposit_type_*``) are summed, so
    a categorical feature is ranked as a single field. This is the view used for the
    domain sanity-check against expected drivers.
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    sources = [_source_feature(name) for name in explanation.feature_names]
    agg = (
        pd.DataFrame({"source": sources, "mean_abs_shap": mean_abs})
        .groupby("source", as_index=False)["mean_abs_shap"]
        .sum()
    )
    return agg.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_beeswarm(
    explanation: shap.Explanation,
    save_path: Path,
    max_display: int = 20,
    title: str = "SHAP summary — categorical_fix_xgb (test split)",
) -> Path:
    """Save the global beeswarm summary plot (top ``max_display`` features).

    ``title`` names the model in the plot heading; pass the canonical model's name
    when explaining ``xgboost_leakage_removed.pkl`` so the figure is labelled correctly.
    """
    fig = plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_title(title, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved beeswarm -> %s", save_path.name)
    return save_path


def plot_dependence(
    explanation: shap.Explanation,
    feature: str,
    save_path: Path,
) -> Path:
    """Save a SHAP dependence (scatter) plot for a single transformed feature."""
    fig = plt.figure()
    shap.plots.scatter(explanation[:, feature], show=False)
    ax = plt.gca()
    ax.set_title(f"SHAP dependence — {feature}", fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved dependence plot (%s) -> %s", feature, save_path.name)
    return save_path


def plot_waterfall(
    explanation: shap.Explanation,
    index: int,
    save_path: Path,
    max_display: int = 15,
) -> Path:
    """Save a per-row waterfall plot explaining one prediction."""
    fig = plt.figure()
    shap.plots.waterfall(explanation[index], max_display=max_display, show=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved waterfall (row %d) -> %s", index, save_path.name)
    return save_path


def select_representative_examples(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, dict[str, Any]]:
    """Pick one confident-correct and one confident-wrong test row for waterfalls.

    Returns positional indices (aligned to ``X`` order, i.e. the SHAP ``Explanation``)
    plus the true label, predicted label and predicted cancellation probability, so
    the notebook/report can state exactly what each waterfall explains.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    true = y.to_numpy()
    confidence = np.abs(proba - 0.5)

    correct = np.where(pred == true)[0]
    wrong = np.where(pred != true)[0]
    if len(correct) == 0 or len(wrong) == 0:
        raise ValueError("need both a correct and an incorrect prediction to select")

    correct_idx = int(correct[np.argmax(confidence[correct])])
    wrong_idx = int(wrong[np.argmax(confidence[wrong])])

    def describe(i: int) -> dict[str, Any]:
        return {
            "index": i,
            "true_label": int(true[i]),
            "predicted_label": int(pred[i]),
            "predicted_proba": float(proba[i]),
        }

    return {"correct": describe(correct_idx), "misclassified": describe(wrong_idx)}
