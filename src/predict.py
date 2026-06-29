"""
Inference helper for the cancellation model.

Loads the saved serving pipeline (preprocessor + best model) and turns a partial
booking dict into a prediction:

    {"prediction": 1, "label": "Cancelled", "cancellation_probability": 0.87}

Callers (CLI, FastAPI) only need to pass the fields they know; everything else is
filled from ``FEATURE_DEFAULTS`` so the one-hot preprocessor always receives the
full column set it was fitted on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.modeling import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_cancellation_model.pkl"

# Sensible neutral defaults for any field the caller omits. Values reflect the
# most common / lowest-risk booking profile in the dataset.
FEATURE_DEFAULTS: dict[str, Any] = {
    # categorical
    "hotel": "City Hotel",
    "arrival_date_month": "August",
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    # numeric
    "lead_time": 80,
    "arrival_date_year": 2016,
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "agent": 0,
    "company": 0,
}

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


@lru_cache(maxsize=1)
def _get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python run_pipeline.py` first."
        )
    return load_model(MODEL_PATH)


def _build_row(booking: dict[str, Any]) -> pd.DataFrame:
    """Merge user input over defaults into a single-row feature DataFrame."""
    row = {**FEATURE_DEFAULTS, **{k: v for k, v in booking.items() if v is not None}}
    return pd.DataFrame([{c: row[c] for c in FEATURE_COLUMNS}])


def predict_cancellation(booking: dict[str, Any]) -> dict[str, Any]:
    """Predict cancellation class + probability for one booking."""
    model = _get_model()
    X = _build_row(booking)
    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)
    return {
        "prediction": prediction,
        "label": "Cancelled" if prediction == 1 else "Not Cancelled",
        "cancellation_probability": round(proba, 4),
    }


if __name__ == "__main__":
    # Quick smoke test with a high-risk profile (long lead time, non-refundable).
    example = {
        "hotel": "Resort Hotel",
        "lead_time": 350,
        "adults": 2,
        "deposit_type": "Non Refund",
        "previous_cancellations": 1,
        "total_of_special_requests": 0,
    }
    print(predict_cancellation(example))
