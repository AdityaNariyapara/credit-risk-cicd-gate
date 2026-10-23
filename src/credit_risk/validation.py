"""Data validation gate (Tasks 2 and 6).

A second gate that runs before training. It rejects a batch whose schema,
missing-value share or value ranges are unacceptable, so a bad dataset cannot
silently produce a bad model. This mirrors a real retraining pipeline, where the
data gate protects the model gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .config import DataConfig, DataValidationConfig
from .data import EXPECTED_COLUMNS


@dataclass
class ValidationResult:
    ok: bool
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return "Data validation PASSED: schema, missing-value share and ranges all acceptable."
        return "Data validation FAILED:\n- " + "\n- ".join(self.issues)


def validate_dataset(df: pd.DataFrame, data: DataConfig,
                     vcfg: DataValidationConfig) -> ValidationResult:
    issues: List[str] = []

    # 1. Row count.
    if len(df) < vcfg.min_rows:
        issues.append(f"too few rows: {len(df)} < min_rows {vcfg.min_rows}")

    # 2. Schema: every expected column present.
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing expected columns: {missing_cols}")

    # 3. Missing-value share per column.
    for col in df.columns:
        share = float(df[col].isna().mean())
        if share > vcfg.max_missing_share:
            issues.append(
                f"column {col} has {share:.1%} missing, above the "
                f"{vcfg.max_missing_share:.0%} limit")

    # 4. Target is binary 0/1.
    if data.target in df.columns:
        vals = set(pd.unique(df[data.target].dropna()))
        if not vals.issubset({0, 1}):
            issues.append(f"target {data.target} has non-binary values: {sorted(vals)}")

    # 5. Numeric sanity ranges.
    for col, (lo, hi) in vcfg.ranges.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            below = int((s < lo).sum())
            above = int((s > hi).sum())
            if below or above:
                issues.append(
                    f"column {col} has {below} values below {lo} and "
                    f"{above} above {hi}")

    return ValidationResult(ok=not issues, issues=issues)
