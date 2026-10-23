"""Tests for the data validation gate."""
import numpy as np
import pandas as pd
import pytest

from credit_risk.config import DataConfig, DataValidationConfig
from credit_risk.validation import validate_dataset
from credit_risk.data import EXPECTED_COLUMNS


def _good(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Customer_ID": [f"CC-{i:05d}" for i in range(n)],
        "Age": rng.integers(20, 80, n),
        "Income_Bracket": rng.choice(["Low", "Medium", "High"], n),
        "Credit_Limit": rng.uniform(1000, 20000, n),
        "Current_Balance": rng.uniform(0, 8000, n),
        "Monthly_Spend": rng.uniform(50, 2000, n),
        "Num_Transactions_Monthly": rng.integers(1, 40, n),
        "Payment_Delay_Days": rng.integers(0, 15, n).astype(float),
        "Tenure_Months": rng.integers(1, 120, n),
        "High_Risk_Customer": rng.integers(0, 2, n),
    })


DATA = DataConfig(version="v1", path="x")
VCFG = DataValidationConfig()


def test_good_data_passes():
    assert validate_dataset(_good(), DATA, VCFG).ok


def test_too_many_missing_fails():
    df = _good()
    df.loc[df.index[:100], "Credit_Limit"] = np.nan  # 50% missing
    res = validate_dataset(df, DATA, VCFG)
    assert not res.ok
    assert any("missing" in i for i in res.issues)


def test_out_of_range_fails():
    df = _good()
    df.loc[df.index[0], "Age"] = 300
    res = validate_dataset(df, DATA, VCFG)
    assert not res.ok
    assert any("Age" in i for i in res.issues)


def test_non_binary_target_fails():
    df = _good()
    df.loc[df.index[0], "High_Risk_Customer"] = 5
    res = validate_dataset(df, DATA, VCFG)
    assert not res.ok
    assert any("target" in i for i in res.issues)


def test_missing_column_fails():
    df = _good().drop(columns=["Credit_Limit"])
    res = validate_dataset(df, DATA, VCFG)
    assert not res.ok
    assert any("missing expected columns" in i for i in res.issues)


def test_too_few_rows_fails():
    res = validate_dataset(_good(10), DATA, VCFG)
    assert not res.ok
    assert any("too few rows" in i for i in res.issues)


def test_real_v1_within_missing_limit():
    # v1 has ~3% missing in three columns, under the 10% default limit.
    from credit_risk.data import load_dataset
    import os
    path = "data/credit_card_customers_v1.csv"
    if not os.path.exists(path):
        pytest.skip("v1 CSV not materialised (run dvc pull)")
    res = validate_dataset(load_dataset(path), DATA, VCFG)
    assert res.ok, res.issues
