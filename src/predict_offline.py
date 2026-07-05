"""
Offline (batch) prediction script.

Loads the saved cancellation model and scores a CSV of new bookings, writing an
output CSV with ``prediction``, ``label`` and ``cancellation_probability`` columns
appended. This is the offline counterpart to the online FastAPI service
(``src/api.py``): the API scores one booking per HTTP request in real time, this
script scores an entire file at once.

Run:
    python -m src.predict_offline --input new_bookings.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.predict import predict_batch


def run(input_path: Path, output_path: Path, model_path: Path | None = None) -> pd.DataFrame:
    """Score every row of ``input_path`` and write the result to ``output_path``."""
    df = pd.read_csv(input_path)
    scored = predict_batch(df, model_path=model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score new bookings for cancellation risk.")
    parser.add_argument("--input", required=True, help="CSV of new bookings to score.")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional path to a specific model .pkl (defaults to models/best_cancellation_model.pkl).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model) if args.model else None

    scored = run(input_path, output_path, model_path=model_path)

    n_cancel = int(scored["prediction"].sum())
    print(f"Scored {len(scored):,} bookings -> {output_path}")
    print(f"Predicted cancellations: {n_cancel:,} ({n_cancel / len(scored) * 100:.1f}%)")
    print(f"Average cancellation probability: {scored['cancellation_probability'].mean():.4f}")


if __name__ == "__main__":
    main()
