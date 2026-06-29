"""
Classification models for hotel booking **cancellation** prediction.

Five models are trained and compared:

1. Logistic Regression  -> linear baseline (high bias, low variance)
2. Decision Tree        -> non-linear, low bias / high variance if unpruned
3. Random Forest        -> bagged trees, reduces variance
4. XGBoost              -> boosted trees, reduces bias + controls variance
5. MLP Classifier       -> deep-learning model, learns complex non-linear patterns

Each model exposes ``predict`` (class) and ``predict_proba`` (cancellation
probability). Models are trained on the **one-hot encoded** feature matrix produced
by ``data_processing.build_preprocessor``.

Scaling note: only the MLP is wrapped with a ``StandardScaler`` (inside its own
pipeline), because gradient-descent / neural nets are sensitive to feature scale.
The tree-based models and the comparison data are left unscaled, honouring the
"no normalization" requirement everywhere else.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_models(random_state: int = 42) -> dict[str, Any]:
    """Return the five candidate classifiers (untrained)."""
    return {
        # No feature scaling (project rule) makes lbfgs converge slowly, so we
        # give the linear baseline more iterations.
        "LogisticRegression": LogisticRegression(
            max_iter=3000, random_state=random_state, n_jobs=-1
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        # Scaling lives INSIDE the MLP pipeline only — see module docstring.
        "MLP": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=60,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_classification(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Compute the full classification metric set for a fitted model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "report": classification_report(
            y, y_pred, target_names=["Not Cancelled", "Cancelled"], zero_division=0
        ),
    }


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Train all five models, timing each, and evaluate on the test set.

    Returns ``(fitted_models, results)`` where ``results[name]`` holds the metric
    dict plus ``train_time_sec``.
    """
    models = build_models(random_state=random_state)
    fitted: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        metrics = evaluate_classification(model, X_test, y_test)
        metrics["train_time_sec"] = round(train_time, 3)

        fitted[name] = model
        results[name] = metrics
        print(
            f"  {name:<20} acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f} "
            f"({train_time:.1f}s)"
        )

    return fitted, results


def comparison_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a sortable comparison table (best ROC-AUC first)."""
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "train_time_sec": m["train_time_sec"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def select_best_model(
    results: dict[str, dict[str, Any]], fitted: dict[str, Any]
) -> tuple[str, Any]:
    """Pick the best model by ROC-AUC (best probability ranking)."""
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    return best_name, fitted[best_name]


def save_model(model: Any, path: Path) -> Path:
    """Persist a model/pipeline to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    """Load a persisted model/pipeline from disk."""
    return joblib.load(path)
