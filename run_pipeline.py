"""
End-to-end runner: train & compare regression models for ADR prediction,
save metrics, and generate SHAP explainability for the best tree-based model.

Reproducible and fast (sampled). Run:  python run_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline import (
    build_splits,
    load_clean_engineer,
    save_comparison,
    select_best_tree_model,
    train_and_compare,
)
from src.shap_analysis import (
    compute_shap_values,
    plot_shap_bar,
    plot_shap_beeswarm,
    top_features,
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_engineer(sample_size=25000)
    X_train, X_test, y_train, y_test = build_splits(df)
    print(f"Data: rows={len(df)}, features={X_train.shape[1]}, "
          f"train={X_train.shape[0]}, test={X_test.shape[0]}")

    models, metrics_df = train_and_compare(X_train, y_train, X_test, y_test)

    print("\n=== Model comparison (test set) ===")
    print(metrics_df.to_string(index=False))

    csv_path = save_comparison(metrics_df, REPORTS_DIR)
    print(f"\nSaved metrics -> {csv_path}")

    best_name, best_model = select_best_tree_model(metrics_df, models)
    print(f"Best tree-based model for SHAP: {best_name}")

    shap_values = compute_shap_values(best_model, X_test, sample_size=1500)
    bar_path = plot_shap_bar(shap_values, REPORTS_DIR)
    bee_path = plot_shap_beeswarm(shap_values, REPORTS_DIR)
    print(f"Saved SHAP plots -> {bee_path.name}, {bar_path.name}")

    print("\n=== Top 10 features by mean |SHAP| ===")
    print(top_features(shap_values, n=10).to_string())


if __name__ == "__main__":
    main()
