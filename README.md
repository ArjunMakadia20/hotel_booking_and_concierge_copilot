# Hotel Booking and Concierge Copilot

AI/ML internship project focused on hotel booking analytics and concierge copilot
development. The current milestone is an **end-to-end, explainable regression
pipeline that predicts a booking's Average Daily Rate (ADR)**.

## Project Overview

Hotels need to understand and forecast room pricing for revenue management. This
project treats **ADR** (a booking's average daily rate) as a regression target,
compares several models, selects the best, and explains *which booking attributes
drive price* using SHAP.

Workflow: load → clean → feature-engineer → EDA → train/compare models → evaluate →
explain (SHAP) → conclude. It is reproducible (`random_state=42`) and fast — it uses
row sampling so the whole project runs in a couple of minutes.

## Datasets

1. **Hotel Booking Demand Dataset** — `data/hotel_bookings.csv` (~119k bookings, 32
   columns, ~16 MB). **Target: `adr` (Average Daily Rate).** This is the dataset used
   for modeling.
2. TripAdvisor Hotel Reviews Dataset — `data/tripadvisor_hotel_reviews.csv` (reserved
   for future concierge/recommendation work).

> If `data/hotel_bookings.csv` is missing, the pipeline raises a clear
> `FileNotFoundError`. Place the CSV at that path to run the project.

## Steps Completed

1. Data loading (`src/data_loader.py`)
2. Cleaning — dedupe, impute sparse fields, fix invalid ADR (`src/data_processing.py`)
3. Feature engineering — `total_guests`, `total_nights`, `arrival_month_num`,
   `arrival_quarter`, reservation date parts; high-cardinality identifiers
   (`country`, `agent`, `company`, `reservation_status`) dropped to keep encoding lean
4. EDA with 9 saved visualizations in `reports/`
5. Model training & comparison (`src/pipeline.py`, `src/modeling.py`)
6. Evaluation metrics saved to `reports/model_comparison.csv`
7. SHAP explainability for the best tree model (`src/shap_analysis.py`)
8. Narrative notebook: `notebooks/hotel_booking_and_concierge_copilot_eda.ipynb`

## Models Trained

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Random Forest | Non-linear ensemble (100 trees) |
| XGBoost | Gradient-boosted trees (falls back to `GradientBoostingRegressor` if XGBoost is unavailable) |

## Evaluation Results

Test set (5,000 bookings held out from a 25,000-row reproducible sample):

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **RandomForest** | **0.798** | **23.28** | **15.68** |
| XGBoost | 0.790 | 23.74 | 16.94 |
| LinearRegression | 0.486 | 37.18 | 27.49 |

**Best model: Random Forest** — explains ~80% of ADR variance, far ahead of the
linear baseline, confirming ADR is driven by non-linear feature interactions.
Full numbers: `reports/model_comparison.csv`.

## SHAP Explanation

SHAP (TreeExplainer) was run on the best tree model over a sampled subset of the
test set. Artifacts:

- `reports/shap_feature_importance.png` — global importance (mean |SHAP|)
- `reports/shap_summary.png` — beeswarm (impact + direction per feature)

**Top ADR drivers:** `total_guests`, `arrival_month_num` (seasonality),
`hotel_Resort Hotel`, `arrival_quarter`, `market_segment_Online TA`, `meal_HB`,
and `lead_time`. These align with real hotel pricing: larger parties, peak-season
months, resort properties, and the booking channel most influence the nightly rate.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure the dataset is present at data/hotel_bookings.csv

# 3. Run the full pipeline (train, compare, save metrics + SHAP plots)
python run_pipeline.py

# 4. Or run the narrative notebook end-to-end
jupyter notebook notebooks/hotel_booking_and_concierge_copilot_eda.ipynb
```

Outputs land in `reports/` (metrics CSV, SHAP plots, EDA figures).

## Project Structure

```text
data/        hotel_bookings.csv, tripadvisor_hotel_reviews.csv
src/         data_loader, data_processing, modeling, pipeline, shap_analysis, eda_analysis
notebooks/   hotel_booking_and_concierge_copilot_eda.ipynb   (full runnable narrative)
reports/     EDA figures, model_comparison.csv, shap_summary.png, shap_feature_importance.png
run_pipeline.py   end-to-end runner
requirements.txt
```
