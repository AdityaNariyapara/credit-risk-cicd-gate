"""Dataset loading (Task 2 / Topic 4).

The CSV itself is tracked by DVC, not plain git. This module only reads the
working copy that `dvc pull` / `dvc checkout` places on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

EXPECTED_COLUMNS = [
    "Customer_ID", "Age", "Income_Bracket", "Credit_Limit", "Current_Balance",
    "Monthly_Spend", "Num_Transactions_Monthly", "Payment_Delay_Days",
    "Tenure_Months", "High_Risk_Customer",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `dvc pull` (or `dvc checkout`) "
            f"to materialise the DVC-tracked data."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} missing expected columns: {sorted(missing)}")
    return df


def split_X_y(df: pd.DataFrame, target: str, id_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features-with-raw-columns, target). Customer_ID is dropped so it
    can never be used as a predictor."""
    y = df[target].astype(int)
    X = df.drop(columns=[target, id_column], errors="ignore")
    return X, y
