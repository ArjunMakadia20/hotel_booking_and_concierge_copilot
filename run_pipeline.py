"""
End-to-end runner for the Hotel Booking **Cancellation** classification project.

Trains and compares five classifiers (Logistic Regression, Decision Tree, Random
Forest, XGBoost, MLP), saves EDA plots, classification reports, confusion-matrix
plots and the model comparison table, then persists the best model.

Run:  python run_pipeline.py
"""

from __future__ import annotations

from src.pipeline import run_pipeline


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
