"""Cleaning and feature engineering (Task 4 / Topic 11).

`FeatureEngineer` is a fit/transform component: everything it needs to apply to
*new* data (imputation medians, encoding maps, the fixed output columns) is
learned on the training split in `fit` and re-applied verbatim in `transform`.
Training and inference therefore share one code path, so there is no train/serve
skew and no target leakage (medians come from train only; the target is never
read here).

Engineered features
-------------------
  credit_utilisation    ratio    Current_Balance / Credit_Limit
  high_utilisation_flag binary   credit_utilisation >= utilisation_high_cutoff
  late_payer_flag       binary   Payment_Delay_Days >= payment_delay_flag_days
  age_band              binned   Age bucketed into configured bands (one-hot)
  avg_transaction_value own      Monthly_Spend / Num_Transactions_Monthly
See docs/feature_table.md for the formula + reason table.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import FeatureConfig

# Income_Bracket is ordinal, so encode it with an order rather than one-hot.
INCOME_ORDER: Dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}
IMPUTE_COLUMNS = ["Credit_Limit", "Current_Balance", "Payment_Delay_Days"]


class FeatureEngineer:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.medians_: Dict[str, float] = {}
        self.output_columns_: List[str] = []
        self._fitted = False

    # -- learn state from training data only -------------------------------
    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        self.medians_ = {c: float(X[c].median()) for c in IMPUTE_COLUMNS if c in X}
        feat = self._build(X)
        self.output_columns_ = list(feat.columns)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform called before fit")
        feat = self._build(X)
        # Guarantee identical columns/order as at fit time (new data may miss a band).
        feat = feat.reindex(columns=self.output_columns_, fill_value=0)
        return feat

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # -- shared clean + engineer path --------------------------------------
    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Impute missing numeric values with train medians (robust to the
        #    skew in financial columns; indicator-free to keep the model simple).
        for col, med in self.medians_.items():
            if col in df:
                df[col] = df[col].fillna(med)

        cl = df["Credit_Limit"].to_numpy(dtype=float)
        bal = df["Current_Balance"].to_numpy(dtype=float)
        spend = df["Monthly_Spend"].to_numpy(dtype=float)
        ntx = df["Num_Transactions_Monthly"].to_numpy(dtype=float)
        delay = df["Payment_Delay_Days"].to_numpy(dtype=float)

        out = pd.DataFrame(index=df.index)

        # Raw numeric predictors kept as-is.
        out["Age"] = df["Age"].to_numpy(dtype=float)
        out["Credit_Limit"] = cl
        out["Current_Balance"] = bal
        out["Monthly_Spend"] = spend
        out["Num_Transactions_Monthly"] = ntx
        out["Payment_Delay_Days"] = delay
        out["Tenure_Months"] = df["Tenure_Months"].to_numpy(dtype=float)

        # Income_Bracket -> ordinal.
        out["Income_Bracket_ord"] = (
            df["Income_Bracket"].map(INCOME_ORDER).fillna(1).astype(int)
        )

        # 2. Engineered: ratio. Guard the zero-credit-limit edge case.
        safe_cl = np.where(cl > 0, cl, np.nan)
        util = np.divide(bal, safe_cl, out=np.zeros_like(bal), where=~np.isnan(safe_cl))
        util = np.nan_to_num(util, nan=0.0, posinf=0.0, neginf=0.0)
        out["credit_utilisation"] = util

        # 3. Engineered: binary flag from the configured utilisation cut-off.
        out["high_utilisation_flag"] = (util >= self.cfg.utilisation_high_cutoff).astype(int)

        # 4. Engineered: binary late-payer flag from the configured threshold.
        out["late_payer_flag"] = (delay >= self.cfg.payment_delay_flag_days).astype(int)

        # 5. Engineered (own design): average transaction value. Very low spend
        #    spread over many transactions signals cash-flow stress; a few large
        #    transactions behaves differently for risk.
        safe_ntx = np.where(ntx > 0, ntx, np.nan)
        atv = np.divide(spend, safe_ntx, out=np.zeros_like(spend), where=~np.isnan(safe_ntx))
        out["avg_transaction_value"] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. Engineered: binned Age -> one-hot over the fixed configured labels.
        band = pd.cut(
            df["Age"], bins=self.cfg.age_bins, labels=self.cfg.age_labels, right=False
        )
        for label in self.cfg.age_labels:
            out[f"age_band_{label}"] = (band == label).astype(int)

        return out
