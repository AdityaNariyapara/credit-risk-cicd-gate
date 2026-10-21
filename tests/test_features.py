"""Unit tests for feature logic, including edge cases (Task 4)."""
import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FeatureConfig
from credit_risk.features import FeatureEngineer


def _row(**kw):
    base = dict(Age=30, Income_Bracket="Medium", Credit_Limit=1000.0,
                Current_Balance=500.0, Monthly_Spend=200.0,
                Num_Transactions_Monthly=10, Payment_Delay_Days=0.0,
                Tenure_Months=24)
    base.update(kw)
    return base


@pytest.fixture
def cfg():
    return FeatureConfig()


def test_utilisation_ratio(cfg):
    fe = FeatureEngineer(cfg)
    out = fe.fit_transform(pd.DataFrame([_row(Credit_Limit=1000, Current_Balance=500)]))
    assert out["credit_utilisation"].iloc[0] == pytest.approx(0.5)


def test_zero_credit_limit_does_not_divide_by_zero(cfg):
    # Edge case required by the brief: Credit_Limit == 0.
    fe = FeatureEngineer(cfg)
    fe.fit(pd.DataFrame([_row(Credit_Limit=1000)]))
    out = fe.transform(pd.DataFrame([_row(Credit_Limit=0, Current_Balance=800)]))
    val = out["credit_utilisation"].iloc[0]
    assert np.isfinite(val)
    assert val == 0.0  # guarded to 0, not inf/NaN


def test_high_utilisation_flag_uses_configured_cutoff():
    cfg = FeatureConfig(utilisation_high_cutoff=0.5)
    fe = FeatureEngineer(cfg)
    fe.fit(pd.DataFrame([_row()]))
    out = fe.transform(pd.DataFrame([
        _row(Credit_Limit=1000, Current_Balance=600),   # util 0.6 -> flag 1
        _row(Credit_Limit=1000, Current_Balance=400),   # util 0.4 -> flag 0
    ]))
    assert list(out["high_utilisation_flag"]) == [1, 0]


def test_late_payer_flag_threshold():
    cfg = FeatureConfig(payment_delay_flag_days=5)
    fe = FeatureEngineer(cfg)
    fe.fit(pd.DataFrame([_row()]))
    out = fe.transform(pd.DataFrame([
        _row(Payment_Delay_Days=5), _row(Payment_Delay_Days=4)]))
    assert list(out["late_payer_flag"]) == [1, 0]


def test_avg_transaction_value_zero_transactions(cfg):
    fe = FeatureEngineer(cfg)
    fe.fit(pd.DataFrame([_row()]))
    out = fe.transform(pd.DataFrame([_row(Num_Transactions_Monthly=0, Monthly_Spend=300)]))
    assert np.isfinite(out["avg_transaction_value"].iloc[0])
    assert out["avg_transaction_value"].iloc[0] == 0.0


def test_missing_values_imputed_with_train_median(cfg):
    train = pd.DataFrame([_row(Credit_Limit=1000), _row(Credit_Limit=3000)])
    fe = FeatureEngineer(cfg).fit(train)
    out = fe.transform(pd.DataFrame([_row(Credit_Limit=np.nan, Current_Balance=500)]))
    # median of [1000, 3000] = 2000
    assert out["Credit_Limit"].iloc[0] == pytest.approx(2000.0)


def test_income_bracket_ordinal_encoding(cfg):
    fe = FeatureEngineer(cfg)
    fe.fit(pd.DataFrame([_row()]))
    out = fe.transform(pd.DataFrame([
        _row(Income_Bracket="Low"), _row(Income_Bracket="Medium"),
        _row(Income_Bracket="High")]))
    assert list(out["Income_Bracket_ord"]) == [0, 1, 2]


def test_transform_columns_stable_on_new_data(cfg):
    # Inference data missing a whole age band must still yield identical columns.
    train = pd.DataFrame([_row(Age=20), _row(Age=30), _row(Age=50), _row(Age=70)])
    fe = FeatureEngineer(cfg).fit(train)
    out = fe.transform(pd.DataFrame([_row(Age=30)]))
    assert list(out.columns) == fe.output_columns_


def test_customer_id_not_in_features(cfg):
    fe = FeatureEngineer(cfg)
    out = fe.fit_transform(pd.DataFrame([_row()]))
    assert not any("Customer_ID" in c or "ID" == c for c in out.columns)
