"""
Deprecated entry point — kept for backwards compatibility.

The original version trained on the full ~119k-row dataset with all high-cardinality
columns one-hot encoded (~900 features), which made Random Forest training hang.
It has been superseded by the fast, sampled, reproducible pipeline in
``run_pipeline.py``. This shim simply delegates there so existing references keep
working.
"""

from __future__ import annotations

from run_pipeline import main

if __name__ == "__main__":
    main()
