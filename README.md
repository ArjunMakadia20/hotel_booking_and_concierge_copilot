# Hotel Booking Cancellation Prediction

Predict whether a hotel booking will be **cancelled**, and return the **probability**
of cancellation, so hotels can act on at-risk reservations (overbooking, deposits,
follow-ups).

```json
{ "prediction": 1, "label": "Cancelled", "cancellation_probability": 0.87 }
```

> **Project pivot:** this repo previously predicted **ADR (Average Daily Rate)** as a
> regression target. The goal changed completely, so all ADR-regression code, plots,
> reports and the SHAP-for-ADR module were **removed** (not archived). `adr` is now
> just one input *feature*, no longer the target.

## Problem statement
Binary **classification**: each booking is either cancelled (`is_canceled = 1`) or not
(`is_canceled = 0`). We want both the predicted class **and** a calibrated probability,
so the output is actionable and threshold-tunable.

## Dataset
- **`data/hotel_bookings.csv`** — 119,390 bookings × 32 columns. The modeling dataset.
- `data/tripadvisor_hotel_reviews.csv` — reserved for future concierge/recommendation
  work; not used for cancellation modeling.

**Target:** `is_canceled` (0 = Not Cancelled, 1 = Cancelled).

## Why this is classification
The target is a discrete yes/no outcome and we need a **probability of cancellation**,
which is exactly what a probabilistic classifier (`predict_proba`) provides — not a
continuous value (that was the old ADR regression task).

## Data understanding & EDA
Full write-up in [`reports/eda_summary.md`](reports/eda_summary.md); plots in `reports/`.

- **Class balance:** 37.0% of bookings cancel (moderate imbalance) → accuracy alone is
  misleading; we emphasise recall / F1 / ROC-AUC.
- **Strongest patterns:** `deposit_type` (Non-Refund cancels **99.4%**), long
  `lead_time` (monotonic rise), few `total_of_special_requests`, prior
  `previous_cancellations`, certain `market_segment` / `distribution_channel`,
  City vs Resort hotel.
- **Plots generated** (in `reports/`): `cancellation_class_distribution.png`,
  `cancellation_rate_by_*` (hotel, lead_time, lead_time_group, arrival_month,
  market_segment, distribution_channel, deposit_type, customer_type,
  reserved/assigned_room_type, special_requests, previous_cancellations,
  parking_spaces), `missing_values.png`, `correlation_heatmap.png`,
  `outlier_boxplots.png`.

### Outlier analysis
Checked 13 key numeric fields with the **IQR rule** (`reports/outlier_report.csv`,
`outlier_boxplots.png`). Flagged values are mostly **legitimate rare cases** (large
parties, many booking changes, one extreme `adr`), so they are **reported, not
removed** — tree models are robust to them and the tails carry real signal. Only
impossible values were fixed (negative `adr` → 0).

## Preprocessing
- **Missing values:** `children`→0, `country`→`"UNKNOWN"`, `agent`/`company`→0 (ID
  codes kept numeric; `company` is 94% empty so it acts as a presence flag).
- **Leakage removed:** `reservation_status`, `reservation_status_date` (they reveal the
  final outcome and aren't available at prediction time).
- **Encoding:** one-hot for categoricals (`OneHotEncoder`, rare categories bucketed via
  `min_frequency`, unseen categories ignored at serving time).
- **No normalization** (supervisor rule). Numerics pass through unchanged. **Exception:**
  a `StandardScaler` is applied **only inside the MLP's own pipeline**, because neural
  nets are scale-sensitive — no other model or the saved data is scaled.
- **Split:** stratified, **~80,000 training records** (rest → test, ~39,390).

## Models trained
| Model | Type | Bias/Variance note |
|-------|------|--------------------|
| Logistic Regression | Linear (baseline) | High bias, low variance |
| Decision Tree | Non-linear | Low bias, high variance if unpruned |
| Random Forest | Non-linear ensemble (bagging) | Reduces variance |
| XGBoost | Non-linear ensemble (boosting) | Reduces bias, controls variance |
| MLP Classifier | Deep learning (non-linear) | Flexible, can overfit; scaled inputs |

## Evaluation metrics
Accuracy, Precision, Recall, F1, **ROC-AUC** (primary), Confusion Matrix, and full
Classification Report — saved to `reports/classification_model_comparison.csv`,
`reports/classification_reports.md`, and `reports/confusion_matrix_*.png`.
Why not accuracy alone? With 37% cancellations, a "never cancels" model scores ~63%
while being useless — see [`reports/modeling_concepts.md`](reports/modeling_concepts.md).

## Model comparison (test set, 39,390 bookings)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|-------|----------|-----------|--------|-----|---------|-----------|
| **XGBoost** | **0.882** | 0.861 | **0.813** | **0.836** | **0.954** | 1.2s |
| Random Forest | 0.879 | 0.891 | 0.767 | 0.825 | 0.953 | 2.0s |
| MLP | 0.863 | 0.839 | 0.781 | 0.809 | 0.939 | 18.7s |
| Decision Tree | 0.848 | 0.838 | 0.732 | 0.782 | 0.927 | 0.9s |
| Logistic Regression | 0.806 | 0.798 | 0.638 | 0.709 | 0.874 | 14.3s |

**Best model: XGBoost** — best ROC-AUC & F1, best probability ranking, and one of the
fastest. Random Forest is a close second; Logistic Regression generalizes worst.

## Linear vs non-linear conclusion
**Non-linear.** No numeric feature is strongly *linearly* correlated with the target
(all |corr| < 0.3), the dominant driver is categorical with threshold behaviour, and
every non-linear model beats the linear Logistic Regression baseline by a wide margin
(ROC-AUC 0.93–0.95 vs 0.87). Cancellation is driven by **interactions and thresholds**.

## Bias & variance (summary)
Logistic Regression **underfits** (high bias). A single Decision Tree **overfits** (high
variance). Random Forest **averages** many trees to cut variance; XGBoost **boosts** to
cut bias while regularizing variance — and **generalizes** best here. The MLP is
flexible but slower and needs scaled inputs. Full discussion in
[`reports/modeling_concepts.md`](reports/modeling_concepts.md).

## Probability prediction
The classifier's `predict_proba` gives P(cancel). We report the class (threshold 0.5)
**and** the probability, so downstream systems can choose their own cut-off (e.g. only
act when P > 0.7). ROC-AUC was the selection metric precisely because it measures
probability-ranking quality.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA + train/compare all 5 models + save best model  (one command)
python run_pipeline.py
#   -> reports/  (EDA plots, classification_model_comparison.csv,
#                 classification_reports.md, confusion_matrix_*.png, outlier_report.csv)
#   -> models/   (best_cancellation_model.pkl, preprocessor.pkl)

# 3. Explore interactively
jupyter notebook notebooks/hotel_booking_and_concierge_copilot_eda.ipynb

# 4. Single prediction from Python
python -m src.predict

# 5. Serve the model with FastAPI
uvicorn src.api:app --reload
```

### FastAPI usage
```bash
# health
curl http://127.0.0.1:8000/health
# -> {"status":"ok"}

# predict (send only the fields you know; the rest use sensible defaults)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "hotel": "Resort Hotel", "lead_time": 350, "adults": 2,
  "deposit_type": "Non Refund", "previous_cancellations": 1,
  "total_of_special_requests": 0
}'
# -> {"prediction":1,"label":"Cancelled","cancellation_probability":...}
```
Interactive docs at `http://127.0.0.1:8000/docs`.

## Project structure
```text
data/        hotel_bookings.csv, tripadvisor_hotel_reviews.csv   (raw — untouched)
src/         data_loader, data_processing, eda_analysis, modeling, pipeline,
             predict (inference helper), api (FastAPI)
notebooks/   hotel_booking_and_concierge_copilot_eda.ipynb
reports/     EDA plots, classification_model_comparison.csv, classification_reports.md,
             confusion_matrix_*.png, outlier_report.csv, eda_summary.md, modeling_concepts.md
models/      best_cancellation_model.pkl, preprocessor.pkl
run_pipeline.py   end-to-end runner
requirements.txt
```

## Future improvements
- Probability **calibration** (Platt / isotonic) and threshold tuning to business cost.
- Hyperparameter search (Optuna) for XGBoost; class-weighting experiments.
- Richer feature engineering (total nights/guests, ADR-per-person, country grouping).
- SHAP explainability rebuilt for the **classification** model (the old ADR SHAP was
  removed). Group rare countries; add monitoring for data drift.
- Integrate with the concierge dataset for a combined booking-assistant product.
