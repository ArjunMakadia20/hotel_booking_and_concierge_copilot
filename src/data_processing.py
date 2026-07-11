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
- ``agent``/``company`` are high-cardinality ID codes. For the legacy five-model
  comparison they are filled with 0 and passed through numerically. For the tuned
  XGBoost path they are recast as ``category`` dtype (``NONE`` for missing) and fed
  to XGBoost's native categorical support, so their integer IDs are never read as
  ordered magnitudes. See ``prepare_xy_native_categorical`` /
  ``build_categorical_preprocessor``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET = "is_canceled"

# Columns that leak the outcome — must never be features.
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# ID-style numeric codes (filled with 0, kept numeric rather than one-hot).
ID_NUMERIC_COLUMNS = ["agent", "company"]

# High-cardinality ID columns handled as native XGBoost categoricals (never numeric).
NATIVE_CATEGORICAL_COLUMNS = ["agent", "company"]

# Domain thresholds for targeted removal of impossible rows (see remove_invalid_rows).
# These are NOT statistical outlier bounds — rare-but-valid extremes are kept.
MIN_VALID_ADR = 0.0
MAX_VALID_ADR = 5000.0
MAX_VALID_ADULTS = 20

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


def _invalid_row_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the boolean mask for each domain-invalid-row rule, keyed by rule name.

    Each mask flags rows that cannot represent a real booking. Rules may overlap
    (a zero-occupancy row can also have non-positive ADR); callers combine them
    with a logical OR to get the set of rows to drop.
    """
    occupancy = (
        df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)
    )
    return {
        "adr_non_positive": df["adr"] <= MIN_VALID_ADR,
        "adr_extreme_high": df["adr"] > MAX_VALID_ADR,
        "zero_occupancy": occupancy == 0,
        "implausible_adults": df["adults"] > MAX_VALID_ADULTS,
    }


def invalid_row_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report how many rows each removal rule matches, plus the unique total.

    Purely descriptive (nothing is dropped) so the counts can be displayed in the
    EDA notebook before removal is applied.
    """
    masks = _invalid_row_masks(df)
    union = pd.Series(False, index=df.index)
    rows = []
    for name, mask in masks.items():
        union |= mask
        rows.append({"rule": name, "rows_matched": int(mask.sum())})
    rows.append({"rule": "total_unique", "rows_matched": int(union.sum())})
    return pd.DataFrame(rows)


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop domain-impossible bookings before splitting, keeping valid rare extremes.

    This is targeted, domain-driven cleaning — NOT a blanket IQR trim. Rare-but-valid
    extremes (long ``lead_time``, high ``previous_cancellations``, many
    ``booking_changes``, long ``days_in_waiting_list``) are deliberately preserved
    because they carry genuine cancellation signal. Four rules are applied:

    - ``adr_non_positive``: zero or negative average daily rate (data errors).
    - ``adr_extreme_high``: a single impossibly high ADR (the ~5,400 record).
    - ``zero_occupancy``: no adults, children or babies (nonsensical booking).
    - ``implausible_adults``: dozens of adults on one row (data-entry errors).

    The count removed by each rule and the deduplicated total are logged.
    """
    df = df.copy()
    n_start = len(df)

    masks = _invalid_row_masks(df)
    union = pd.Series(False, index=df.index)
    for name, mask in masks.items():
        union |= mask
        logger.info("remove_invalid_rows: rule '%s' matched %d rows", name, int(mask.sum()))

    cleaned = df.loc[~union].reset_index(drop=True)
    n_removed = int(union.sum())
    logger.info(
        "remove_invalid_rows: removed %d of %d rows (%.2f%%); %d remain",
        n_removed,
        n_start,
        100.0 * n_removed / max(n_start, 1),
        len(cleaned),
    )
    return cleaned


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


def _to_native_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Recast the ID columns to ``category`` dtype with an explicit ``NONE`` label.

    ``clean_hotel_booking_data`` fills missing agent/company with 0; here 0 becomes
    the ``NONE`` category (no agent / not a company booking) and every other ID
    becomes its own string category. Returns a copy; other columns are untouched.
    """
    df = df.copy()
    for col in NATIVE_CATEGORICAL_COLUMNS:
        if col in df.columns:
            codes = df[col].astype(int)
            labels = codes.astype(str).where(codes != 0, "NONE")
            df[col] = labels.astype("category")
    return df


def prepare_xy_native_categorical(
    df: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature/target split with agent/company typed as native categoricals.

    Identical feature set to :func:`prepare_xy`, except the ID columns are excluded
    from the numeric block and returned as ``category`` dtype so XGBoost's
    ``enable_categorical`` handles them without treating IDs as ordered magnitudes.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    y = df[target].astype(int)
    numeric = [c for c in NUMERIC_COLUMNS if c not in NATIVE_CATEGORICAL_COLUMNS]
    feature_cols = [
        c
        for c in (CATEGORICAL_COLUMNS + numeric + NATIVE_CATEGORICAL_COLUMNS)
        if c in df.columns
    ]
    X = _to_native_categoricals(df[feature_cols].copy())
    return X, y


def build_categorical_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Preprocessor for the native-categorical XGBoost path.

    Mirrors :func:`build_preprocessor` exactly for the low-cardinality categoricals
    (one-hot, rare-bucketed) and the true numerics (passthrough); the only difference
    is that agent/company pass through as their ``category`` dtype rather than as
    numeric columns. ``set_output('pandas')`` keeps the category dtype intact through
    the transformer so ``XGBClassifier(enable_categorical=True)`` can consume it.
    """
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [
        c
        for c in NUMERIC_COLUMNS
        if c in X.columns and c not in NATIVE_CATEGORICAL_COLUMNS
    ]
    native = [c for c in NATIVE_CATEGORICAL_COLUMNS if c in X.columns]

    one_hot = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=0.01,
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("cat", one_hot, categorical),
            ("num", "passthrough", numeric),
            ("native", "passthrough", native),
        ],
        remainder="drop",
    ).set_output(transform="pandas")


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
