"""
Preprocessing for Hotel Booking *Cancellation* classification.

Target: ``is_canceled`` (0 = not cancelled, 1 = cancelled).

Key decisions (per project brief):
- Drop **leakage** columns ``reservation_status`` and ``reservation_status_date``.
  They record the *final outcome* of the booking, so keeping them would let the
  model "cheat" and would never be available at prediction time.
- Encode categoricals with **one-hot encoding** (``OneHotEncoder``). Rare
  categories are bucketed (``min_frequency``) so high-cardinality fields like
  ``country`` do not explode the feature space, and unseen categories at serving
  time are handled gracefully (``handle_unknown='ignore'``).
- **No global normalization** (supervisor requirement). Numeric features are
  passed through unchanged. Scaling is applied *only* inside the MLP model's own
  pipeline (see ``modeling.py``), never here.
- ``agent``/``company`` are ID codes -> filled with 0 and treated as numeric so
  they do not blow up one-hot encoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

TARGET = "is_canceled"

# Columns that leak the outcome — must never be features.
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# ID-style numeric codes (filled with 0, kept numeric rather than one-hot).
ID_NUMERIC_COLUMNS = ["agent", "company"]

# Categorical features encoded via one-hot.
CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

# Numeric features (passed through; adr kept as an input feature, no longer target).
NUMERIC_COLUMNS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "agent",
    "company",
]

# Numeric columns most worth checking for outliers (used by EDA, reported only).
OUTLIER_COLUMNS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary of the DataFrame for EDA reporting."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }


def clean_hotel_booking_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset for cancellation classification.

    Handles missing values and removes leakage columns. Does NOT drop rows on the
    target and does NOT scale anything.
    """
    df = df.copy()

    # --- Missing values ---------------------------------------------------
    # `children`: a handful of NaNs -> 0 (no children recorded).
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0).astype(int)

    # `country`: categorical -> explicit UNKNOWN bucket.
    if "country" in df.columns:
        df["country"] = df["country"].fillna("UNKNOWN").astype(str)

    # `agent`/`company`: ID codes, mostly-missing `company` -> 0 means "none".
    for col in ID_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Guard against impossible negative ADR (data-entry artefacts).
    if "adr" in df.columns:
        df.loc[df["adr"] < 0, "adr"] = 0

    # --- Remove leakage ---------------------------------------------------
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    return df


def prepare_xy(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into feature matrix X and target vector y."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    y = df[target].astype(int)
    feature_cols = [c for c in (CATEGORICAL_COLUMNS + NUMERIC_COLUMNS) if c in df.columns]
    X = df[feature_cols].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the one-hot + passthrough preprocessor (fit happens in the pipeline).

    No scaling here — numerics pass through untouched. Rare categories are bucketed
    and unseen categories ignored, which keeps the encoded matrix bounded and makes
    the saved preprocessor safe to reuse at serving time.
    """
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in NUMERIC_COLUMNS if c in X.columns]

    one_hot = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=0.01,  # bucket categories appearing in <1% of rows
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("cat", one_hot, categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int | float = 80000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Uses ~80,000 training records when the dataset is large enough (supervisor
    request); otherwise falls back to a 80/20 stratified split.
    """
    if isinstance(train_size, int) and train_size >= len(X):
        train_size = 0.8  # dataset smaller than requested -> proportion fallback

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
