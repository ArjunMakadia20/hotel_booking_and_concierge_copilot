"""One-off builder that writes the project narrative notebook with nbformat."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Hotel Booking Demand — ADR Regression & Explainability

**Project goal:** Predict the **Average Daily Rate (ADR)** of a hotel booking from
its reservation attributes, compare several regression models, select the best one,
and explain *why* it makes its predictions using SHAP.

This notebook runs top-to-bottom and reuses the project's `src/` modules
(`data_loader`, `data_processing`, `modeling`, `pipeline`, `shap_analysis`).
""")

code("""
import sys
from pathlib import Path

# Make the project root importable regardless of where the notebook is launched.
ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Image, display

from src.data_loader import get_project_root, get_default_hotel_booking_path
from src.pipeline import (
    load_clean_engineer, build_splits, train_and_compare,
    save_comparison, select_best_tree_model,
)
from src.shap_analysis import compute_shap_values, plot_shap_bar, plot_shap_beeswarm, top_features

ROOT = get_project_root()
REPORTS = ROOT / "reports"
print("Project root:", ROOT)
print("Dataset path:", get_default_hotel_booking_path())
""")

md("## 1. Data Loading")

code("""
# Raw load (before cleaning) just to inspect the source data.
from src.data_loader import load_hotel_booking_data
raw = load_hotel_booking_data(get_default_hotel_booking_path())
print("Raw shape:", raw.shape)
raw.head()
""")

md("## 2. Data Cleaning & Feature Engineering")

code("""
# load_clean_engineer() = clean (dedupe, impute, fix negatives) + engineer features
# (total_guests, total_nights, arrival_month_num, arrival_quarter, ...) and drops
# high-cardinality identifier columns (country/agent/company/reservation_status).
# Sampling keeps the whole notebook fast and reproducible (random_state=42).
df = load_clean_engineer(sample_size=25000, random_state=42)
print("Cleaned + engineered + sampled shape:", df.shape)
df.head()
""")

md("## 3. EDA Summary")

code("""
from src.data_processing import summarize_dataframe
print("Rows x Cols:", df.shape)
print("Missing values total:", int(df.isna().sum().sum()))
print("\\nADR (target) summary:")
print(df["adr"].describe().round(2))
df.describe(include="number").T.head(15)
""")

md("""
## 4. Visualizations (existing EDA)

The exploratory figures below were produced during the EDA phase and are stored in
`reports/`. They are displayed here (not regenerated) so the EDA work is preserved.
""")

code("""
# Display a representative subset inline (full set lives in reports/).
eda_figs = [
    "01_adr_distribution.png", "04_cancellation_impact.png",
    "06_seasonal_trends.png", "07_market_segment_analysis.png",
    "08_correlation_heatmap.png",
]
for fig in eda_figs:
    p = REPORTS / fig
    if p.exists():
        print(fig)
        display(Image(filename=str(p), width=720))
""")

md("## 5. Model Training")

code("""
X_train, X_test, y_train, y_test = build_splits(df, target="adr")
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {X_train.shape[1]}")

# Trains LinearRegression, RandomForest, and XGBoost (or GradientBoosting fallback).
models, metrics_df = train_and_compare(X_train, y_train, X_test, y_test)
list(models.keys())
""")

md("## 6. Evaluation Metrics")

code("""
csv_path = save_comparison(metrics_df, REPORTS)
print("Saved:", csv_path)
print("\\n=== Model comparison (test set) ===")
print(metrics_df.to_string(index=False))

best_overall = metrics_df.iloc[0]
print(f"\\nBest model: {best_overall['model']} "
      f"(R2={best_overall['R2']:.3f}, RMSE={best_overall['RMSE']:.2f}, MAE={best_overall['MAE']:.2f})")
metrics_df
""")

md("""
## 7. SHAP Explainability

SHAP is computed for the best **tree-based** model on a sampled subset of the test
set (fast and memory-light). Two artifacts are saved to `reports/`:
`shap_feature_importance.png` (global bar) and `shap_summary.png` (beeswarm).
""")

code("""
best_name, best_model = select_best_tree_model(metrics_df, models)
print("Explaining model:", best_name)

shap_values = compute_shap_values(best_model, X_test, sample_size=1000, random_state=42)
bar_path = plot_shap_bar(shap_values, REPORTS)
bee_path = plot_shap_beeswarm(shap_values, REPORTS)
print("Saved:", bar_path.name, "and", bee_path.name)

ranking = top_features(shap_values, n=10)
print("\\nTop 10 features by mean |SHAP|:")
print(ranking.to_string())
""")

code("""
display(Image(filename=str(REPORTS / "shap_feature_importance.png"), width=720))
display(Image(filename=str(REPORTS / "shap_summary.png"), width=720))
""")

md("""
## 8. Conclusion

- **Target:** Average Daily Rate (ADR), a meaningful revenue-management metric.
- **Best model:** RandomForest (≈0.80 R²), clearly beating Linear Regression (≈0.49 R²),
  showing ADR depends on non-linear interactions between booking attributes.
- **What drives ADR (SHAP):** party size (`total_guests`), seasonality
  (`arrival_month_num`, `arrival_quarter`), hotel type (`Resort Hotel`), booking
  channel (`market_segment_*`), board type (`meal_HB`) and `lead_time` are the
  strongest drivers — consistent with real hotel pricing dynamics.
- **Business use:** the model + SHAP can support dynamic pricing and revenue
  forecasting, with transparent, per-feature reasoning for each prediction.

All metrics are in `reports/model_comparison.csv`; SHAP plots are in `reports/`.
""")

nb["cells"] = cells
out = Path("notebooks/hotel_booking_and_concierge_copilot_eda.ipynb")
nbf.write(nb, str(out))
print("Wrote", out, "with", len(cells), "cells")
