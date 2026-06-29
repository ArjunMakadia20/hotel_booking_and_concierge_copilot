"""
Cancellation-focused Exploratory Data Analysis.

Generates a set of plots saved into ``reports/`` that explain *what drives hotel
booking cancellations* (target = ``is_canceled``):

- class distribution & class imbalance
- cancellation rate by every important categorical / grouped feature
- lead-time relationship (continuous, binned)
- missing values
- correlation heatmap (numeric features vs. is_canceled)
- outlier boxplots (IQR-based, reported not removed)

A companion ``reports/eda_summary.md`` is written by ``write_eda_summary``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_processing import OUTLIER_COLUMNS

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _rate_by_category(
    df: pd.DataFrame,
    column: str,
    save_path: Path,
    filename: str,
    title: str | None = None,
    order: list | None = None,
    rotate: int = 0,
) -> None:
    """Bar chart of cancellation rate (%) for each value of a categorical column."""
    if column not in df.columns:
        print(f"  skip {filename} (no column '{column}')")
        return

    grp = df.groupby(column, observed=True)["is_canceled"].agg(["mean", "count"])
    if order is not None:
        grp = grp.reindex([o for o in order if o in grp.index])
    else:
        grp = grp.sort_values("mean", ascending=False)
    rates = grp["mean"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(rates)), rates.values, color="indianred",
                  edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel("Cancellation Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel(column, fontsize=11, fontweight="bold")
    ax.set_title(title or f"Cancellation Rate by {column}", fontsize=12, fontweight="bold")
    ax.axhline(df["is_canceled"].mean() * 100, color="navy", linestyle="--",
               linewidth=1.5, label="Overall rate")
    ax.legend()
    for bar, v in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.0f}%",
                ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {filename}")


# --------------------------------------------------------------------------- #
# Individual plots
# --------------------------------------------------------------------------- #
def plot_class_distribution(df: pd.DataFrame, save_path: Path) -> None:
    counts = df["is_canceled"].value_counts().sort_index()
    labels = ["Not Cancelled (0)", "Cancelled (1)"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts.values, color=["seagreen", "indianred"],
                  edgecolor="black", alpha=0.85)
    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + total * 0.005,
                f"{v:,}\n({v / total * 100:.1f}%)", ha="center", fontweight="bold")
    ax.set_ylabel("Number of Bookings", fontsize=11, fontweight="bold")
    ax.set_title("Target Class Distribution (is_canceled)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "cancellation_class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved cancellation_class_distribution.png")


def plot_lead_time(df: pd.DataFrame, save_path: Path) -> None:
    """Cancellation behaviour vs. lead_time (continuous): overlaid distributions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cap = df["lead_time"].quantile(0.99)
    for label, color, val in [("Not Cancelled", "seagreen", 0), ("Cancelled", "indianred", 1)]:
        data = df.loc[df["is_canceled"] == val, "lead_time"].clip(upper=cap)
        ax.hist(data, bins=50, alpha=0.55, label=label, color=color, edgecolor="none")
    ax.set_xlabel("Lead Time (days, capped at 99th pct)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Bookings", fontsize=11, fontweight="bold")
    ax.set_title("Lead Time Distribution by Cancellation Status", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path / "cancellation_rate_by_lead_time.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved cancellation_rate_by_lead_time.png")


def plot_lead_time_group(df: pd.DataFrame, save_path: Path) -> None:
    """Cancellation rate by lead-time bucket (clear monotonic trend)."""
    bins = [-1, 7, 30, 90, 180, 365, np.inf]
    labels = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]
    grp = df.assign(lead_group=pd.cut(df["lead_time"], bins=bins, labels=labels))
    _rate_by_category(grp, "lead_group", save_path,
                      "cancellation_rate_by_lead_time_group.png",
                      title="Cancellation Rate by Lead-Time Group (days)",
                      order=labels)


def plot_missing_values(df: pd.DataFrame, save_path: Path) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        pct = (missing / len(df) * 100)
        ax.barh(range(len(missing)), pct.values, color="coral", edgecolor="black")
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index)
        ax.invert_yaxis()
        ax.set_xlabel("Missing (%)", fontsize=11, fontweight="bold")
        for i, v in enumerate(pct.values):
            ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_title("Missing Values by Feature (raw data)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "missing_values.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved missing_values.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    # Order numeric features by absolute correlation with the target.
    if "is_canceled" in corr.columns:
        order = corr["is_canceled"].abs().sort_values(ascending=False).index.tolist()
        corr = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features, ordered by |corr| with target)",
                 fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(save_path / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved correlation_heatmap.png")


def plot_outlier_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    cols = [c for c in OUTLIER_COLUMNS if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightsteelblue"))
        ax.set_title(col, fontsize=10, fontweight="bold")
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Outlier Boxplots (IQR) — reported, not auto-removed",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "outlier_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved outlier_boxplots.png")


def iqr_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-feature IQR outlier count/percentage table."""
    rows = []
    for col in [c for c in OUTLIER_COLUMNS if c in df.columns]:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        rows.append({
            "feature": col,
            "outliers": int(mask.sum()),
            "outlier_pct": round(mask.mean() * 100, 2),
            "lower_bound": round(lo, 2),
            "upper_bound": round(hi, 2),
        })
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda_analysis(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Generate and save all cancellation-focused EDA plots into ``reports/``."""
    output_dir = output_dir or Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CANCELLATION EDA — generating plots")
    print("=" * 60)

    plot_class_distribution(df, output_dir)
    _rate_by_category(df, "hotel", output_dir, "cancellation_rate_by_hotel_type.png",
                      title="Cancellation Rate by Hotel Type")
    plot_lead_time(df, output_dir)
    plot_lead_time_group(df, output_dir)
    _rate_by_category(df, "arrival_date_month", output_dir,
                      "cancellation_rate_by_arrival_month.png",
                      title="Cancellation Rate by Arrival Month",
                      order=MONTH_ORDER, rotate=45)
    _rate_by_category(df, "market_segment", output_dir,
                      "cancellation_rate_by_market_segment.png",
                      title="Cancellation Rate by Market Segment", rotate=45)
    _rate_by_category(df, "distribution_channel", output_dir,
                      "cancellation_rate_by_distribution_channel.png",
                      title="Cancellation Rate by Distribution Channel", rotate=30)
    _rate_by_category(df, "deposit_type", output_dir,
                      "cancellation_rate_by_deposit_type.png",
                      title="Cancellation Rate by Deposit Type")
    _rate_by_category(df, "customer_type", output_dir,
                      "cancellation_rate_by_customer_type.png",
                      title="Cancellation Rate by Customer Type")
    _rate_by_category(df, "reserved_room_type", output_dir,
                      "cancellation_rate_by_reserved_room_type.png",
                      title="Cancellation Rate by Reserved Room Type")
    _rate_by_category(df, "assigned_room_type", output_dir,
                      "cancellation_rate_by_assigned_room_type.png",
                      title="Cancellation Rate by Assigned Room Type")
    _rate_by_category(df, "total_of_special_requests", output_dir,
                      "cancellation_rate_by_special_requests.png",
                      title="Cancellation Rate by Number of Special Requests")
    _rate_by_category(df, "previous_cancellations", output_dir,
                      "cancellation_rate_by_previous_cancellations.png",
                      title="Cancellation Rate by Previous Cancellations")
    _rate_by_category(df, "required_car_parking_spaces", output_dir,
                      "cancellation_rate_by_parking_spaces.png",
                      title="Cancellation Rate by Required Car Parking Spaces")
    plot_missing_values(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_outlier_boxplots(df, output_dir)

    print("=" * 60)
    print(f"EDA plots saved to: {output_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_hotel_booking_data, get_default_hotel_booking_path
    from src.data_processing import clean_hotel_booking_data

    df_raw = load_hotel_booking_data(get_default_hotel_booking_path())
    run_eda_analysis(clean_hotel_booking_data(df_raw))
