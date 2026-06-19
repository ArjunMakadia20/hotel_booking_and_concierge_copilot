from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Return the repository root path for the hotel booking copilot project."""
    return Path(__file__).resolve().parent.parent


def get_data_dir(root: Optional[Path] = None) -> Path:
    """Return the project data directory path."""
    root_path = root or get_project_root()
    return root_path / "data"


def load_hotel_booking_data(csv_path: Path | str) -> pd.DataFrame:
    """Load hotel booking data from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def load_tripadvisor_reviews_data(csv_path: Path | str) -> pd.DataFrame:
    """Load TripAdvisor hotel reviews data from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def get_default_hotel_booking_path(root: Optional[Path] = None) -> Path:
    """Return the default CSV path for the hotel booking dataset."""
    return get_data_dir(root) / "hotel_bookings.csv"


def get_default_tripadvisor_reviews_path(root: Optional[Path] = None) -> Path:
    """Return the default CSV path for the TripAdvisor reviews dataset."""
    return get_data_dir(root) / "tripadvisor_hotel_reviews.csv"
