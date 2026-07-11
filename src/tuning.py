"""
Hyperparameter analysis for the cancellation XGBoost classifier.

Two tools, both operating strictly on the TRAINING split — the held-out test set is
never seen during any search or cross-validation:

1. ``run_max_depth_validation_curve`` / ``plot_validation_curve`` — a bias/variance
   sweep of ``max_depth`` via ``sklearn.model_selection.validation_curve``. The
   estimator is the full ``preprocessor + XGBClassifier`` pipeline, so one-hot
   encoding is refit inside every CV fold and no validation leakage occurs.
2. ``run_optuna_search`` — a TPE Optuna study jointly tuning the main XGBoost
   hyperparameters, optimising mean CV ROC-AUC on the training split, then a single
   held-out test evaluation of the best configuration, logged to MLflow.

The baseline configuration in ``src/train.py`` is left untouched: the tuned model is
saved additively (``models/xgboost_optuna_tuned.pkl``) and logged as its own MLflow
run under the existing ``hotel_cancellation_classification`` experiment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, validation_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import build_categorical_preprocessor, build_preprocessor
from src.modeling import evaluate_classification, load_model, save_model

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
EXPERIMENT_NAME = "hotel_cancellation_classification"

BASE_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "verbosity": 0,
}


def build_xgb_pipeline(
    X: pd.DataFrame,
    random_state: int = 42,
    n_jobs: int = 1,
    native_categorical: bool = False,
    **overrides: Any,
) -> Pipeline:
    """Return a ``preprocessor + XGBClassifier`` pipeline on raw booking features.

    The preprocessor is rebuilt (unfitted) from ``X`` so it refits inside each CV
    fold. ``overrides`` replace individual XGBoost hyperparameters over
    ``BASE_XGB_PARAMS``; ``n_jobs`` defaults to 1 so an outer parallel CV loop does
    not oversubscribe the cores.

    When ``native_categorical`` is True the agent/company ID columns are passed to
    XGBoost as native categoricals (``enable_categorical=True``, ``tree_method='hist'``)
    via :func:`build_categorical_preprocessor`, instead of the numeric passthrough
    used by the default one-hot preprocessor.
    """
    params = {**BASE_XGB_PARAMS, **overrides}
    if native_categorical:
        preprocessor = build_categorical_preprocessor(X)
        params = {**params, "enable_categorical": True, "tree_method": "hist"}
    else:
        preprocessor = build_preprocessor(X)
    model = XGBClassifier(random_state=random_state, n_jobs=n_jobs, **params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


TUNED_PARAM_KEYS = [
    "max_depth",
    "min_child_weight",
    "gamma",
    "reg_lambda",
    "reg_alpha",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
]


def metrics_comparison_table(metrics_by_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Tidy accuracy/precision/recall/F1/ROC-AUC/PR-AUC table across named runs."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = [{"model": name, **{k: m[k] for k in keys}} for name, m in metrics_by_name.items()]
    return pd.DataFrame(rows)


def extract_best_params(model_path: Path) -> dict[str, Any]:
    """Read the tuned XGBoost hyperparameters back out of a saved serving pipeline.

    Used so the categorical refit reuses the *exact* D4 Optuna configuration rather
    than a re-typed copy, isolating the encoding change in the comparison.
    """
    pipeline = load_model(model_path)
    xgb_params = pipeline.named_steps["model"].get_params()
    return {k: xgb_params[k] for k in TUNED_PARAM_KEYS}


def run_categorical_refit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: dict[str, Any],
    random_state: int = 42,
    experiment_name: str = EXPERIMENT_NAME,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    run_name: str = "categorical_fix_xgb",
) -> dict[str, Any]:
    """Refit the tuned XGBoost config with native categorical agent/company handling.

    ``X_train``/``X_test`` must come from :func:`prepare_xy_native_categorical` (agent
    and company as ``category`` dtype). The model uses the same ``best_params`` as the
    D4 tuned run, so the only difference from ``optuna_tuned_xgb`` is the ID encoding.
    Metrics, params and the confusion-matrix figure are logged to MLflow, and the
    pipeline is saved additively to ``models/xgboost_categorical_fix.pkl``.
    """
    pipeline = build_xgb_pipeline(
        X_train, random_state=random_state, n_jobs=-1, native_categorical=True, **best_params
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_classification(pipeline, X_test, y_test)

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    cm_fig = _confusion_matrix_figure(metrics["confusion_matrix"], run_name)
    cm_path = reports_dir / "confusion_matrix_categorical_fix.png"
    cm_fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    model_path = models_dir / "xgboost_categorical_fix.pkl"
    save_model(pipeline, model_path)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "encoding": "native_categorical_agent_company",
                "enable_categorical": True,
                "tree_method": "hist",
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
            }
        )
        mlflow.log_text(metrics["report"], "classification_report.txt")
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")
    plt.close(cm_fig)

    logger.info(
        "categorical_fix test: acc=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f -> %s",
        metrics["accuracy"], metrics["f1"], metrics["roc_auc"], metrics["pr_auc"],
        model_path.name,
    )

    return {
        "pipeline": pipeline,
        "test_metrics": metrics,
        "model_path": model_path,
        "confusion_matrix_path": cm_path,
        "best_params": best_params,
    }


def run_max_depth_validation_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    depths: Iterable[int] = range(2, 15),
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute train/validation ROC-AUC across ``max_depth`` on the training split.

    Uses stratified k-fold CV over the full pipeline (no test data involved). Returns
    the raw score arrays plus the depth at which mean validation ROC-AUC peaks.
    """
    pipeline = build_xgb_pipeline(X_train, random_state=random_state, n_jobs=1)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    depth_list = list(depths)

    train_scores, val_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name="model__max_depth",
        param_range=depth_list,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    val_mean = val_scores.mean(axis=1)
    best_idx = int(val_mean.argmax())
    result = {
        "param_name": "max_depth",
        "depths": depth_list,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "best_depth": depth_list[best_idx],
        "best_val_score": float(val_mean[best_idx]),
    }
    logger.info(
        "validation curve: best max_depth=%d (CV ROC-AUC=%.4f)",
        result["best_depth"],
        result["best_val_score"],
    )
    return result


def plot_validation_curve(
    result: dict[str, Any],
    save_path: Path = REPORTS_DIR,
    filename: str = "validation_curve_max_depth.png",
) -> Path:
    """Plot mean train/validation ROC-AUC vs ``max_depth`` with +/-1 std bands."""
    depths = result["depths"]
    train_mean = result["train_scores"].mean(axis=1)
    train_std = result["train_scores"].std(axis=1)
    val_mean = result["val_scores"].mean(axis=1)
    val_std = result["val_scores"].std(axis=1)
    best_depth = result["best_depth"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depths, train_mean, marker="o", color="steelblue", label="Train ROC-AUC")
    ax.fill_between(depths, train_mean - train_std, train_mean + train_std,
                    color="steelblue", alpha=0.15)
    ax.plot(depths, val_mean, marker="s", color="indianred", label="CV validation ROC-AUC")
    ax.fill_between(depths, val_mean - val_std, val_mean + val_std,
                    color="indianred", alpha=0.15)
    ax.axvline(best_depth, color="seagreen", linestyle="--", linewidth=1.5,
               label=f"Best depth = {best_depth}")

    ax.set_xlabel("max_depth", fontsize=11, fontweight="bold")
    ax.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    ax.set_title("Validation Curve — XGBoost max_depth (5-fold CV, training set only)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(list(depths))
    ax.legend()

    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    path = save_path / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", filename)
    return path


def _suggest_xgb_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Sample the joint XGBoost hyperparameter space for one Optuna trial."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 14),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _confusion_matrix_figure(cm: np.ndarray, run_name: str) -> plt.Figure:
    """Build a labelled confusion-matrix heatmap figure for logging."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Cancelled", "Cancelled"],
                yticklabels=["Not Cancelled", "Cancelled"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_name}", fontweight="bold")
    plt.tight_layout()
    return fig


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 50,
    cv_splits: int = 5,
    random_state: int = 42,
    experiment_name: str = EXPERIMENT_NAME,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Run a TPE Optuna study on the training split and evaluate the best config once.

    The objective is mean stratified-k-fold ROC-AUC on the training set. After the
    study, the best hyperparameters are refit on the full training set and evaluated a
    single time on the held-out test set (confusion matrix, per-class metrics,
    accuracy, ROC-AUC and PR-AUC). Best params, all trial results, final test metrics
    and the confusion-matrix figure are logged to MLflow. The tuned pipeline is saved
    additively as ``models/xgboost_optuna_tuned.pkl`` without touching the baseline.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_xgb_params(trial)
        pipeline = build_xgb_pipeline(
            X_train, random_state=random_state, n_jobs=-1, **params
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name="xgb_roc_auc"
    )
    logger.info("starting Optuna study: %d trials, %d-fold CV", n_trials, cv_splits)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Optuna best CV ROC-AUC=%.4f params=%s", study.best_value, study.best_params
    )

    best_pipeline = build_xgb_pipeline(
        X_train, random_state=random_state, n_jobs=-1, **study.best_params
    )
    best_pipeline.fit(X_train, y_train)
    metrics = evaluate_classification(best_pipeline, X_test, y_test)

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_csv = reports_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    cm_fig = _confusion_matrix_figure(metrics["confusion_matrix"], "optuna_tuned_xgb")
    cm_path = reports_dir / "confusion_matrix_optuna_tuned.png"
    cm_fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    model_path = models_dir / "xgboost_optuna_tuned.pkl"
    save_model(best_pipeline, model_path)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="optuna_tuned_xgb"):
        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "search": "optuna_tpe",
                "n_trials": n_trials,
                "cv_splits": cv_splits,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics(
            {
                "cv_best_roc_auc": study.best_value,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
            }
        )
        for trial in study.trials:
            if trial.value is not None:
                mlflow.log_metric("trial_cv_roc_auc", trial.value, step=trial.number)
        mlflow.log_artifact(str(trials_csv))
        mlflow.log_text(metrics["report"], "classification_report.txt")
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        mlflow.sklearn.log_model(
            best_pipeline, name="model", serialization_format="pickle"
        )
    plt.close(cm_fig)

    logger.info(
        "optuna_tuned test: acc=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f -> %s",
        metrics["accuracy"], metrics["f1"], metrics["roc_auc"], metrics["pr_auc"],
        model_path.name,
    )

    return {
        "study": study,
        "best_params": study.best_params,
        "best_cv_roc_auc": study.best_value,
        "test_metrics": metrics,
        "pipeline": best_pipeline,
        "model_path": model_path,
        "trials_csv": trials_csv,
        "confusion_matrix_path": cm_path,
    }
